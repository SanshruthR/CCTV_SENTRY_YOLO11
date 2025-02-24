import multiprocessing
import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import logging
import math
import time
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store line coordinates and line equation
start_point = None
end_point = None
line_params = None  # Stores (slope, intercept) of the line

# Maximize CPU usage
cpu_cores = multiprocessing.cpu_count()
cv2.setNumThreads(cpu_cores)
logger.info(f"OpenCV using {cv2.getNumThreads()} threads out of {cpu_cores} available cores")

def extract_first_frame(stream_url):
    """
    Extracts the first available frame from the IP camera stream and returns it as a PIL image.
    """
    logger.info("Attempting to extract the first frame from the stream...")
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        logger.error("Error: Could not open stream.")
        return None, "Error: Could not open stream."

    ret, frame = cap.read()
    cap.release()

    if not ret:
        logger.error("Error: Could not read the first frame.")
        return None, "Error: Could not read the first frame."

    # Convert the frame to a PIL image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    logger.info("First frame extracted successfully.")
    return pil_image, "First frame extracted successfully."

def update_line(image, evt: gr.SelectData):
    """
    Updates the line based on user interaction (click and drag).
    """
    global start_point, end_point, line_params

    # If it's the first click, set the start point and show it on the image
    if start_point is None:
        start_point = (evt.index[0], evt.index[1])

        # Draw the start point on the image
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            (start_point[0] - 5, start_point[1] - 5, start_point[0] + 5, start_point[1] + 5),
            fill="blue", outline="blue"
        )

        return image, f"Line Coordinates:\nStart: {start_point}, End: None"

    # If it's the second click, set the end point and draw the line
    end_point = (evt.index[0], evt.index[1])

    # Calculate the slope (m) and intercept (b) of the line: y = mx + b
    if start_point[0] != end_point[0]:  # Avoid division by zero
        slope = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
        intercept = start_point[1] - slope * start_point[0]
        line_params = (slope, intercept, start_point, end_point)  # Store slope, intercept, and points
    else:
        # Vertical line (special case)
        line_params = (float('inf'), start_point[0], start_point, end_point)

    # Draw the line and end point on the image
    draw = ImageDraw.Draw(image)
    draw.line([start_point, end_point], fill="red", width=2)
    draw.ellipse(
        (end_point[0] - 5, end_point[1] - 5, end_point[0] + 5, end_point[1] + 5),
        fill="green", outline="green"
    )

    # Return the updated image and line info
    line_info = f"Line Coordinates:\nStart: {start_point}, End: {end_point}\nLine Equation: y = {line_params[0]:.2f}x + {line_params[1]:.2f}"

    # Reset the points for the next interaction
    start_point = None
    end_point = None

    return image, line_info

def reset_line():
    """
    Resets the line coordinates.
    """
    global start_point, end_point, line_params
    start_point = None
    end_point = None
    line_params = None
    return None, "Line reset. Click to draw a new line."

def intersect(A, B, C, D):
    """
    Determines if two line segments AB and CD intersect.
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])

    def on_segment(A, B, C):
        if min(A[0], B[0]) <= C[0] <= max(A[0], B[0]) and min(A[1], B[1]) <= C[1] <= max(A[1], B[1]):
            return True
        return False

    # Check if the line segments intersect
    ccw1 = ccw(A, B, C)
    ccw2 = ccw(A, B, D)
    ccw3 = ccw(C, D, A)
    ccw4 = ccw(C, D, B)

    if ((ccw1 * ccw2 < 0) and (ccw3 * ccw4 < 0)):
        return True
    elif ccw1 == 0 and on_segment(A, B, C):
        return True
    elif ccw2 == 0 and on_segment(A, B, D):
        return True
    elif ccw3 == 0 and on_segment(C, D, A):
        return True
    elif ccw4 == 0 and on_segment(C, D, B):
        return True
    else:
        return False

def is_object_crossing_line(box, line_params):
    """
    Determines if an object's bounding box is fully intersected by the user-drawn line.
    """
    _, _, line_start, line_end = line_params

    # Get the bounding box coordinates
    x1, y1, x2, y2 = box

    # Define the four edges of the bounding box
    box_edges = [
        ((x1, y1), (x2, y1)),  # Top edge
        ((x2, y1), (x2, y2)),  # Right edge
        ((x2, y2), (x1, y2)),  # Bottom edge
        ((x1, y2), (x1, y1))   # Left edge
    ]

    # Count the number of intersections between the line and the bounding box edges
    intersection_count = 0
    for edge_start, edge_end in box_edges:
        if intersect(line_start, line_end, edge_start, edge_end):
            intersection_count += 1

    # Only count the object if the line intersects the bounding box at least twice
    return intersection_count >= 2

def draw_angled_line(image, line_params, color=(0, 255, 0), thickness=2):
    """
    Draws the user-defined line on the frame.
    """
    _, _, start_point, end_point = line_params
    cv2.line(image, start_point, end_point, color, thickness)

def process_video(confidence_threshold=0.5, selected_classes=None, stream_url=None, target_fps=30, model_name="yolov8n.pt"):
    """
    Processes the IP camera stream to count objects of the selected classes crossing the line.
    """
    global line_params

    errors = []

    if line_params is None:
        errors.append("Error: No line drawn. Please draw a line on the first frame.")
    if selected_classes is None or len(selected_classes) == 0:
        errors.append("Error: No classes selected. Please select at least one class to detect.")
    if stream_url is None or stream_url.strip() == "":
        errors.append("Error: No stream URL provided.")

    if errors:
        return None, "\n".join(errors)

    logger.info("Connecting to the IP camera stream...")
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        errors.append("Error: Could not open stream.")
        return None, "\n".join(errors)

    model = YOLO(model=model_name)
    crossed_objects = {}
    max_tracked_objects = 1000  # Maximum number of objects to track before clearing

    # Queue to hold frames for processing
    frame_queue = deque(maxlen=10)

    logger.info("Starting to process the stream...")
    last_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            errors.append("Error: Could not read frame from the stream.")
            break

        # Add frame to the queue
        frame_queue.append(frame)

        # Process frames in the queue
        if len(frame_queue) > 0:
            process_frame = frame_queue.popleft()

            # Perform object tracking with confidence threshold
            results = model.track(process_frame, persist=True, conf=confidence_threshold)

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()
                boxes = results[0].boxes.xyxy.cpu()
                confs = results[0].boxes.conf.cpu().tolist()

                for box, cls, t_id, conf in zip(boxes, clss, track_ids, confs):
                    if conf >= confidence_threshold and model.names[cls] in selected_classes:
                        # Check if the object crosses the line
                        if is_object_crossing_line(box, line_params) and t_id not in crossed_objects:
                            crossed_objects[t_id] = True

                            # Clear the dictionary if it gets too large
                            if len(crossed_objects) > max_tracked_objects:
                                crossed_objects.clear()

            # Visualize the results with bounding boxes, masks, and IDs
            annotated_frame = results[0].plot()

            # Draw the angled line on the frame
            draw_angled_line(annotated_frame, line_params, color=(0, 255, 0), thickness=2)

            # Display the count on the frame with a modern look
            count = len(crossed_objects)
            (text_width, text_height), _ = cv2.getTextSize(f"COUNT: {count}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            # Calculate the position for the middle of the top
            margin = 10  # Margin from the top
            x = (annotated_frame.shape[1] - text_width) // 2  # Center-align the text horizontally
            y = text_height + margin  # Top-align the text

            # Draw the black background rectangle
            cv2.rectangle(annotated_frame, (x - margin, y - text_height - margin), (x + text_width + margin, y + margin), (0, 0, 0), -1)

            # Draw the text
            cv2.putText(annotated_frame, f"COUNT: {count}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Yield the annotated frame to Gradio
            yield annotated_frame, ""

        # Calculate the time taken to process the frame
        current_time = time.time()
        elapsed_time = current_time - last_time
        last_time = current_time

        # Calculate the time to sleep to maintain the target FPS
        sleep_time = max(0, (1.0 / target_fps) - elapsed_time)
        time.sleep(sleep_time)

    cap.release()
    logger.info("Stream processing completed.")

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("<h1>Real-time monitoring, object tracking, and line-crossing detection for CCTV camera streams.</h1></center>")
    gr.Markdown("## https://github.com/SanshruthR/CCTV_SENTRY_YOLO12")
    
    # Step 1: Enter the IP Camera Stream URL
    ip="https://59d39900ebfb8.streamlock.net/Channels301/default.stream/chunklist_w1780413149.m3u8"
    stream_url = gr.Textbox(label="Enter IP Camera Stream URL", value=ip, visible=False)
    

    # Step 1: Extract the first frame from the stream
    gr.Markdown("### Step 1: Click on the frame to draw a line, the objects crossing it would be counted in real-time.")
    first_frame, status = extract_first_frame(stream_url.value)
    if first_frame is None:
        gr.Markdown(f"**Error:** {status}")
    else:
        # Image component for displaying the first frame
        image = gr.Image(value=first_frame, label="First Frame of Stream", type="pil")

        line_info = gr.Textbox(label="Line Coordinates", value="Line Coordinates:\nStart: None, End: None")
        image.select(update_line, inputs=image, outputs=[image, line_info])

        # Step 2: Select classes to detect
        gr.Markdown("### Step 2: Select Classes to Detect")
        model = YOLO(model="yolov8n.pt")  # Load the model to get class names
        class_names = list(model.names.values())  # Get class names
        selected_classes = gr.CheckboxGroup(choices=class_names, label="Select Classes to Detect")

        # Step 3: Adjust confidence threshold 
        gr.Markdown("### Step 3: Adjust Confidence Threshold (Optional)")
        confidence_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, label="Confidence Threshold")

        # Step 4: Set target FPS
        gr.Markdown("### Step 4: Set Target FPS (Optional)")
        target_fps = gr.Slider(minimum=1, maximum=120*4, value=60, label="Target FPS")

        # Step 5: Select YOLO model
        gr.Markdown("### Step 5: Select YOLO Model")
        model_name = gr.Dropdown(choices=["yolov8n.pt", "yolo11n.pt","yolo12n.pt"], label="Select YOLO Model", value="yolov8n.pt")

        # Process the stream
        process_button = gr.Button("Process Stream")

        # Output image for real-time frame rendering
        output_image = gr.Image(label="Processed Frame", streaming=True)

        # Error box to display warnings/errors
        error_box = gr.Textbox(label="Errors/Warnings", interactive=False)

        # Event listener for processing the video
        process_button.click(process_video, inputs=[confidence_threshold, selected_classes, stream_url, target_fps, model_name], outputs=[output_image, error_box])

# Launch the interface
demo.launch(debug=True)
