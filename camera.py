from flask import Flask, render_template_string, Response
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLO classes
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Vehicle classes in COCO
vehicle_classes = ['car', 'bus', 'truck', 'motorbike']

# Initialize YOLO
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# To capture from the built-in laptop camera, use index 0
# If this doesn't work on your system, try 1 or 2
cap = cv2.VideoCapture(0)

def generate_frames():
    """
    Generator function: 
    - Reads frames from the webcam
    - Runs YOLO to detect vehicles
    - Draws bounding boxes and yields frames as JPEG for streaming
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO Preprocessing
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 
            0.00392, 
            (416, 416), 
            (0, 0, 0), 
            swapRB=True, 
            crop=False
        )
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        # Parse YOLO output
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Only consider confidence above a threshold
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

        # Non-Max Suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]

                # Draw bounding box & label
                color = (0, 255, 0)  # green
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame, 
                    f"{label} {int(confidence*100)}%", 
                    (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color, 
                    2
                )

        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # Yield the frame in MIME multipart format (MJPEG)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """
    Simple HTML page to display the live video feed from the webcam
    """
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Real-Time Vehicle Detection</title>
    </head>
    <body style="text-align:center; background:#fafafa; margin:0;">
        <h1>Vehicle Detection - Laptop Camera</h1>
        <p>This page shows real-time detections from your built-in webcam.</p>
        <img src="{{ url_for('video_feed') }}" alt="Live Webcam Feed" />
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/video_feed')
def video_feed():
    """
    Route that streams the MJPEG frames from generate_frames()
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown')
def shutdown():
    """
    Optional route to shut down the app and release the camera.
    """
    cap.release()
    return "Camera released and app shutdown."

if __name__ == '__main__':
    # Run the Flask app
    # Visit http://127.0.0.1:5000/ in your browser to see the live feed
    app.run(debug=True)
