from flask import Flask, render_template_string, Response
import cv2
import numpy as np
import math
import time

app = Flask(__name__)

# ---------------------------------------------------------------------------------------
# 1. YOLO Initialization
# ---------------------------------------------------------------------------------------
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
vehicle_classes = ['car', 'bus', 'truck', 'motorbike']

# If you want a live camera feed, set video_path = 0 or another index.
# For a file, set video_path = 'traffic.mp4' or 'test.mp4'.
video_path = 'nithu.mp4'

# ---------------------------------------------------------------------------------------
# 2. Perspective Transformation Setup
# ---------------------------------------------------------------------------------------
# Define points for perspective transformation
# Adjust these points based on your drone-angle video
src_points = np.float32([[100, 700], [1200, 700], [300, 200], [1000, 200]])  # Source points from the video
output_width, output_height = 800, 600  # Desired output resolution
dst_points = np.float32([[0, output_height], [output_width, output_height], [0, 0], [output_width, 0]])

# Compute the homography matrix
homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# ---------------------------------------------------------------------------------------
# 3. Naive Data Structures for Tracking Two Lines
# ---------------------------------------------------------------------------------------
"""
objects[object_id] = {
    "center": (cx, cy),
    "lost_frames": int,
    "matched": bool,
    "state": str,        # "aboveA", "betweenA_B", "crossedB"
    "time_lineA": float, # when crossing line A
    "time_lineB": float, # when crossing line B
    "speed_status": str, # "traffic" or "normal"
    "speed_px_sec": float # speed in pixels/second
}
"""
objects = {}
next_object_id = 0
cross_count = 0

# Thresholds / Constants
MAX_LOST_FRAMES = 40
DIST_THRESHOLD = 50

# If crossing from line A to line B is slower than this (px/sec), it's traffic
SPEED_THRESHOLD = 2.0  # px/s

def match_detections_to_objects(detections, lineA_y, lineB_y):
    """
    1. Match bounding box centers in 'detections' to existing 'objects' by proximity.
    2. Update states (aboveA -> betweenA_B -> crossedB).
    3. Compute speed (in px/s) when crossing B and update "traffic" or "normal."
    """
    global objects, next_object_id, cross_count

    # Mark all as unmatched this frame
    for obj_id in objects:
        objects[obj_id]["matched"] = False

    # Attempt to match each detection
    for (cx, cy) in detections:
        best_obj_id = None
        best_dist = float('inf')

        for obj_id, info in objects.items():
            old_cx, old_cy = info["center"]
            dist = math.hypot(cx - old_cx, cy - old_cy)
            if dist < best_dist and dist < DIST_THRESHOLD:
                best_dist = dist
                best_obj_id = obj_id

        if best_obj_id is not None:
            # Update existing object
            info = objects[best_obj_id]
            old_cy = info["center"][1]
            old_state = info["state"]

            info["center"] = (cx, cy)
            info["lost_frames"] = 0
            info["matched"] = True

            # If "aboveA" -> crosses line A => "betweenA_B"
            if old_state == "aboveA":
                if old_cy < lineA_y <= cy:
                    info["state"] = "betweenA_B"
                    info["time_lineA"] = time.time()

            # If "betweenA_B" -> crosses line B => "crossedB"
            elif old_state == "betweenA_B":
                if old_cy < lineB_y <= cy:
                    info["state"] = "crossedB"
                    info["time_lineB"] = time.time()

                    crossing_time = info["time_lineB"] - info["time_lineA"]
                    distance_px = lineB_y - lineA_y  # vertical distance in px
                    # speed in px/s
                    speed_px_sec = 0.0
                    if crossing_time > 0:
                        speed_px_sec = distance_px / crossing_time
                    info["speed_px_sec"] = speed_px_sec

                    # If speed < threshold => traffic, else normal
                    if speed_px_sec < SPEED_THRESHOLD:
                        info["speed_status"] = "traffic"
                    else:
                        info["speed_status"] = "normal"

                    cross_count += 1

        else:
            # New object
            start_state = "aboveA"
            if cy >= lineA_y:
                start_state = "betweenA_B"
            objects[next_object_id] = {
                "center": (cx, cy),
                "lost_frames": 0,
                "matched": True,
                "state": start_state,
                "time_lineA": 0.0,
                "time_lineB": 0.0,
                "speed_status": "",
                "speed_px_sec": 0.0
            }
            next_object_id += 1

    # Remove unmatched objects if lost for too many frames
    to_remove = []
    for obj_id, info in objects.items():
        if not info["matched"]:
            info["lost_frames"] += 1
            if info["lost_frames"] > MAX_LOST_FRAMES:
                to_remove.append(obj_id)
    for r in to_remove:
        del objects[r]

def generate_frames():
    global objects, cross_count

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply perspective transformation
        transformed_frame = cv2.warpPerspective(frame, homography_matrix, (output_width, output_height))

        height, width = transformed_frame.shape[:2]
        lineA_y = height // 2
        lineB_y = int(height * 0.8)

        # YOLO detection
        blob = cv2.dnn.blobFromImage(transformed_frame, 0.00392, (416, 416),
                                     (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] in vehicle_classes:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w // 2
                    y = center_y - h // 2

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Create detections for matching
        detections = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                confidence = confidences[i]

                # Draw bounding box
                color = (0, 255, 0)
                cv2.rectangle(transformed_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(transformed_frame, f"{label} {int(confidence*100)}%",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

                # Center for naive tracking
                cx = x + w // 2
                cy = y + h // 2
                detections.append((cx, cy))

        # Update objects with detections
        match_detections_to_objects(detections, lineA_y, lineB_y)

        # Draw lines
        cv2.line(transformed_frame, (0, lineA_y), (width, lineA_y), (255, 0, 0), 2)   # Blue line A
        cv2.line(transformed_frame, (0, lineB_y), (width, lineB_y), (0, 0, 255), 2)  # Red line B

        # Show how many have crossed line B
        cv2.putText(transformed_frame, f"Count Crossed B: {cross_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

        # Show speed & status for objects that have crossed B
        for obj_id, info in objects.items():
            if info["state"] == "crossedB":
                cx, cy = info["center"]
                speed_px_sec = info["speed_px_sec"]
                speed_status = info["speed_status"]
                # e.g.: "traffic (1.50px/s)" or "normal (3.12px/s)"
                text = f"{speed_status} ({speed_px_sec:.2f}px/s)"
                color_label = (0, 0, 255) if speed_status == "traffic" else (0, 255, 0)

                # Draw the text near the vehicle center
                cv2.putText(transformed_frame, text, (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            color_label, 2)

        # Stream the frame
        ret, buffer = cv2.imencode('.jpg', transformed_frame)
        if not ret:
            continue
        yield (  b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/')
def index():
    """
    Main HTML page.
    """
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>YOLO Vehicle Counting</title>
        <style>
            /* Reset default margin & padding */
            * {
              margin: 0;
              padding: 0;
              box-sizing: border-box;
            }
            body {
              font-family: Arial, sans-serif;
              text-align: center;
              color: #fff;
              background: linear-gradient(-45deg, red, blue, black, white);
              background-size: 400% 400%;
              animation: bgAnimation 10s ease infinite;
              padding: 20px;
              overflow-x: hidden;
            }
            @keyframes bgAnimation {
              0% { background-position: 0% 50%; }
              50% { background-position: 100% 50%; }
              100% { background-position: 0% 50%; }
            }
            h1 {
              font-size: 2.5rem;
              margin-bottom: 10px;
              text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            }
            p {
              margin-bottom: 20px;
            }
            .video-container {
              width: 400px;
              margin: 0 auto;
              border: 4px solid #fff;
              border-radius: 10px;
              overflow: hidden;
              background-color: #222;
            }
            .video-container img {
              width: 100%;
              display: block;
            }
        </style>
    </head>
    <body>
        <h1>YOLO Vehicle Counting</h1>
        <p>Below is the live video stream with bounding boxes and counts.</p>
        <div class="video-container">
        <img src="{{ url_for('video_feed') }}" alt="Video Feed">
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
