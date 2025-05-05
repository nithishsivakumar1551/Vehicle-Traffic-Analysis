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

# We'll consider these as vehicles
vehicle_classes = ['car', 'bus', 'truck', 'motorbike']

# If you want a live camera feed, set video_path = 0 or another index
# For a file, set video_path = 'traffic.mp4'
video_path = 'test.mp4'  

# ---------------------------------------------------------------------------------------
# 2. Tracking data structure (naive approach with two lines)
# ---------------------------------------------------------------------------------------
"""
objects[object_id] = {
    "center": (cx, cy),
    "lost_frames": int,
    "matched": bool,
    "state": str,        # "aboveA", "betweenA_B", "crossedB", "done"
    "time_lineA": float, # when crossed line A
    "time_lineB": float, # when crossed line B
    "speed_status": str  # "traffic" or "normal"
}
"""
objects = {}
next_object_id = 0
cross_count = 0

MAX_LOST_FRAMES = 40
DIST_THRESHOLD = 50

# Speed thresholds (naive, just a time difference threshold)
# E.g. if crossing from line A to line B took more than 3 seconds => "traffic"
TIME_THRESHOLD = 35.0

def match_detections_to_objects(detections, lineA_y, lineB_y):
    """
    1. Match bounding box centers in 'detections' to existing 'objects'.
    2. Update states (aboveA -> betweenA_B -> crossedB).
    3. Compute "traffic" or "normal" when crossing line B.
    """
    global objects, next_object_id, cross_count

    # Mark all objects as unmatched
    for obj_id in objects:
        objects[obj_id]["matched"] = False

    # Match each detection
    for (cx, cy) in detections:
        best_obj_id = None
        best_dist = float('inf')

        # Find closest object within DIST_THRESHOLD
        for obj_id, info in objects.items():
            old_cx, old_cy = info["center"]
            dist = math.hypot(cx - old_cx, cy - old_cy)
            if dist < best_dist and dist < DIST_THRESHOLD:
                best_dist = dist
                best_obj_id = obj_id

        if best_obj_id is not None:
            # Update existing object
            info = objects[best_obj_id]
            old_cx, old_cy = info["center"]
            old_state = info["state"]

            info["center"] = (cx, cy)
            info["lost_frames"] = 0
            info["matched"] = True

            # State transitions
            # 1) If it was "aboveA", check if it now crosses lineA
            if old_state == "aboveA":
                if old_cy < lineA_y <= cy:
                    # Crossed line A now
                    info["state"] = "betweenA_B"
                    info["time_lineA"] = time.time()
            
            # 2) If it's "betweenA_B", check if it crosses line B
            elif old_state == "betweenA_B":
                if old_cy < lineB_y <= cy:
                    # Crossed line B
                    info["state"] = "crossedB"
                    info["time_lineB"] = time.time()
                    # Evaluate time difference => traffic or normal
                    crossing_time = info["time_lineB"] - info["time_lineA"]
                    if crossing_time > TIME_THRESHOLD:
                        info["speed_status"] = "traffic"
                    else:
                        info["speed_status"] = "normal"
                    cross_count += 1

        else:
            # Create new object
            start_state = "aboveA"
            # if the center is already below lineA, then it's "betweenA_B" or "crossedB" 
            # but typically let's assume it's "aboveA" if it's not too far down
            if cy >= lineA_y:
                start_state = "betweenA_B"
            objects[next_object_id] = {
                "center": (cx, cy),
                "lost_frames": 0,
                "matched": True,
                "state": start_state,
                "time_lineA": 0.0,
                "time_lineB": 0.0,
                "speed_status": ""
            }
            next_object_id += 1

    # For unmatched objects, increment lost_frames
    # remove if beyond MAX_LOST_FRAMES
    to_remove = []
    for obj_id, info in objects.items():
        if not info["matched"]:
            info["lost_frames"] += 1
            if info["lost_frames"] > MAX_LOST_FRAMES:
                to_remove.append(obj_id)

    for obj_id in to_remove:
        del objects[obj_id]

def generate_frames():
    global objects, next_object_id, cross_count

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]

        # Define two lines (for demonstration):
        #  - line A in the middle
        #  - line B near the bottom
        lineA_y = height // 2
        lineB_y = int(height * 0.8)  # 80% down the frame

        # YOLO detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416),
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
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        detections = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]

                # Draw bounding box
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {int(confidence * 100)}%",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

                cx = x + w // 2
                cy = y + h // 2
                detections.append((cx, cy))

        # Match detections -> objects, update states
        match_detections_to_objects(detections, lineA_y, lineB_y)

        # Draw lines
        cv2.line(frame, (0, lineA_y), (width, lineA_y), (255, 0, 0), 2)  # blue line A
        cv2.line(frame, (0, lineB_y), (width, lineB_y), (0, 0, 255), 2)  # red line B

        # Display overall cross_count
        cv2.putText(frame, f"Count Crossed B: {cross_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

        # For demonstration: show traffic or normal for each active object
        # We'll display a small label near the object's center
        for obj_id, info in objects.items():
            if info["state"] == "crossedB":
                cx, cy = info["center"]
                speed_status = info["speed_status"]
                cv2.putText(frame, speed_status, (cx - 20, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255) if speed_status == "traffic" else (0,255,0),
                            2)

        # Encode frame & yield
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    """
    Simple HTML to display the feed.
    """
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Two-Line Speed / Traffic Demo</title>
    </head>
    <body style="text-align:center; margin:0; background:#f0f0f0;">
        <h1>Two-Line Vehicle Speed Estimation (Naive)</h1>
        <p>Red line is near the bottom. If crossing from the blue line to the red line takes too long => 'traffic'</p>
        <img src="{{ url_for('video_feed') }}" alt="Video Feed">
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
