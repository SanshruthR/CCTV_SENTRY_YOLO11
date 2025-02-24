# CCTV_SENTRY_YOLO12

![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv12-FF6200?style=for-the-badge&logo=ultralytics&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-0DAB76?style=for-the-badge&logo=gradio&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-FFEB3B?style=for-the-badge&logo=open-source-initiative&logoColor=303030)

![download](https://github.com/user-attachments/assets/1123f224-7984-4294-a6de-2a335e702816)




## Overview
CCTV_SENTRY_YOLO12 is an advanced object detection system built using YOLOv12 by Ultralytics. It provides real-time monitoring, object tracking, and line-crossing detection for IP camera streams. Hosted on Hugging Face Spaces, it enables users to easily interact with the model via a web interface.

## Features
- **Line-Crossing Detection**: Detects objects crossing user-defined lines.
- **Real-Time Object Detection**: Utilizes YOLOv12 for high-speed object tracking.
- **Interactive Interface**: Powered by Gradio for an intuitive user experience.
- **Customizable Classes**: Filter objects based on specific detection classes.
- **Detailed Visualization**: Annotated frames with bounding boxes, IDs, and counts.
  

https://github.com/user-attachments/assets/e29ad9df-b810-4308-b6a8-4ff81019edea



## Model Details
The system leverages the YOLOv12 model from Ultralytics for accurate and efficient object detection. Key technologies include:
- **OpenCV**: For video frame processing.
- **Gradio**: For creating an interactive user interface.
- **Ultralytics YOLO**: For state-of-the-art object detection and tracking.

## How It Works
1. Upload or provide the URL of an IP camera stream.
2. Draw a line on the first frame to set the detection boundary.
3. Select the object classes to monitor.
4. Watch real-time detections and line-crossing counts directly on the interface.
## Industrial and Commercial Applications


### **Parking Management**
 - Vehicle entry/exit tracking
 - Parking space occupancy monitoring
 - Unauthorized parking detection

### **Manufacturing**
 - Conveyor belt product counting
 - Quality control inspections
 - Real-time inventory tracking

### **Retail and Logistics**
 - Customer movement analysis
 - Stock level monitoring
 - Theft prevention systems

### **Transportation**
 - Vehicle tracking
 - Loading dock management
 - Traffic flow analysis

### **Security**
 - Perimeter surveillance
 - Restricted area monitoring
 - Crowd density estimation




## Usage
### Requirements
- Python 3.x
- Install dependencies:
  ```bash
  pip install ultralytics opencv-python-headless gradio numpy pillow
  ```

### Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/SanshruthR/CCTV_SENTRY_YOLO12.git
   
   ```
2. Navigate to the project directory:
   ```bash
   cd CCTV_SENTRY_YOLO12
   pip install -r requirements.txt
   ```
3. Start the application:
   ```bash
   python app.py
   ```

### Live Demo
Experience the project live on Hugging Face Spaces:  
[CCTV_SENTRY_YOLO12 on Hugging Face](https://huggingface.co/spaces/Sanshruth/CCTV_SENTRY_YOLO12)

## Deployment
* To create a live HLS stream for testing, refer to this GitHub repository:  
https://github.com/SanshruthR/mock-hls-server

* Use the sample video file for testing:  
https://videos.pexels.com/video-files/1169852/1169852-hd_1920_1080_30fps.mp4

## License
This project is licensed under the MIT License. 

