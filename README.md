# Vehicle Traffic Analysis

A Python-based project designed to analyze vehicle traffic using computer vision techniques. The system utilizes YOLO (You Only Look Once) for object detection to monitor and analyze traffic patterns from video footage.

## Features

* **Vehicle Detection**: Employs YOLO for real-time detection of vehicles in video streams.
* **Traffic Analysis**: Analyzes traffic flow and density based on detected vehicles.
* **Drone Angle Detection**: Includes functionality to detect and adjust for drone camera angles in traffic footage.
* **Web Interface**: Provides a simple web interface (`index.html`) to display analysis results.

## Project Structure

```bash
Vehicle-Traffic-Analysis/
├── 2lines.py
├── camera.py
├── index.html
├── traffic.mp4
└── yolo_drone_angle_detection.py
```



* `2lines.py`: Script to define and process two lines for traffic analysis.
* `camera.py`: Handles video capture and preprocessing.
* `index.html`: Frontend interface to display traffic analysis results.
* `traffic.mp4`: Sample traffic video used for analysis.
* `yolo_drone_angle_detection.py`: Script to detect vehicles and adjust for drone angles using YOLO.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/nithishsivakumar1551/Vehicle-Traffic-Analysis.git
   cd Vehicle-Traffic-Analysis
   ```



2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```



3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```



*Note: Ensure that `requirements.txt` is present in the repository with all necessary dependencies listed.*

## Usage

1. **Run the main analysis script**:

   ```bash
   python 2lines.py
   ```

## Requirements

* Python 3.6 or higher
* OpenCV
* NumPy
* YOLOv5 or compatible YOLO version
* Flask (if the web interface is powered by Flask)

*Note: Ensure all dependencies are listed in `requirements.txt`.*

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.
