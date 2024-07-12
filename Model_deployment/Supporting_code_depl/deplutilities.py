import os
import cv2
import json
import math
import matplotlib.pyplot as plt


def show_image(img, title='Image'):
    plt.figure(figsize=(10, 8))
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None, image
    
    # Select the face with the largest area
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    
    # Draw rectangle around the largest face
    x, y, w, h = largest_face
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return largest_face, image

def crop_face(image, face_rect, padding=1.0):
    x, y, w, h = face_rect
    x_pad = int(padding * w)
    y_pad = int(padding * h)
    x1 = max(0, x - x_pad)
    y1 = max(0, y - y_pad)
    x2 = min(image.shape[1], x + w + x_pad)
    y2 = min(image.shape[0], y + h + y_pad)
    return image[y1:y2, x1:x2], (x1, y1)

def map_coordinates(pred, offset, cropped_shape, original_shape):
    mapped_pred = []
    x_offset, y_offset = offset
    crop_h, crop_w = cropped_shape
    orig_h, orig_w = original_shape
    
    for p in pred:
        cls, x_center, y_center, width, height = p
        x_center = int(x_center * crop_w + x_offset)
        y_center = int(y_center * crop_h + y_offset)
        width = int(width * crop_w)
        height = int(height * crop_h)
        
        mapped_pred.append({
            'marker': cls,
            'x': x_center,
            'y': y_center,
            'width': width,
            'height': height
        })
    
    return mapped_pred

def draw_circles(image, detections):
    for detection in detections:
        x_center = int(detection['x'])
        y_center = int(detection['y'])
        width = int(detection['width'])
        height = int(detection['height'])
        radius = int((width + height) / 4)

        if detection['marker'] == 0:  # Class 0: round marker
            color = (50, 255, 0) 
        elif detection['marker'] == 1:  # Class 1: square marker
            color = (0, 255, 50)
        else:
            color = (0, 0, 255)  # Green for any other class (just in case)

        cv2.circle(image, (x_center, y_center), radius, color, 2)

def process_image(model, image_path, output_path, face_crop_padding=1.0, apply_box=True):

    # Load the image
    image = cv2.imread(image_path)
    original_image = image.copy()  
    original_image_shape = image.shape[:2]

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Detect the largest face in the image
    face_rect, image_with_faces = detect_face(image)
    
    if face_rect is None:
        print("No face detected.")
        return
    
    # Optionally draw the face box
    if apply_box:
        x, y, w, h = face_rect
        cv2.rectangle(original_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Crop the image to retain only the face and some surrounding area
    cropped_image, offset = crop_face(original_image, face_rect, padding=face_crop_padding)
    cropped_image_shape = cropped_image.shape[:2]

    
    # Apply YOLOv8 on the cropped image
    results = model.predict(source=cropped_image)
    
    # Extract predictions and map them back to the original image
    detections = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            x_center, y_center, width, height = box.xywhn[0]
            detections.append([cls, x_center, y_center, width, height])
    
    mapped_detections = map_coordinates(detections, offset, cropped_image_shape, original_image_shape)
    
    # Draw circles around the detected markers
    draw_circles(original_image, mapped_detections)
    
    # Save the output image
    cv2.imwrite(output_path, original_image)
    print(f"Output saved to {output_path}")

    # Show the final image
    #show_image(original_image, 'Processed Image with Detected Faces and Markers')


def order_points(points):
    # Order the points in a consistent way to form a proper quadrilateral
    # Sort the points based on their x-coordinates
    points = sorted(points, key=lambda x: x[0])
    leftmost = points[:2]
    rightmost = points[2:]

    # Sort the leftmost points based on their y-coordinates to determine top-left and bottom-left
    leftmost = sorted(leftmost, key=lambda x: x[1])
    (tl, bl) = leftmost

    # Sort the rightmost points based on their y-coordinates to determine top-right and bottom-right
    rightmost = sorted(rightmost, key=lambda x: x[1])
    (tr, br) = rightmost

    return [tl, tr, br, bl]


def process_video(model, video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    processed_frames = 0
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        original_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        # Apply YOLOv8 on the frame
        detections = model.predict(source=frame)
        
        # Filter for round markers (class 0)
        round_markers = []
        for detection in detections:
            for box in detection.boxes:
                if int(box.cls) == 0:  # Class 0: round marker
                    x_center, y_center, width, height = box.xywhn[0]
                    x_center = int(x_center * frame_width)
                    y_center = int(y_center * frame_height)
                    round_markers.append((x_center, y_center))
        
        # If exactly 4 round markers are found
        if len(round_markers) == 4:
            ordered_markers = order_points(round_markers)
            results.append({
                'frame_number': frame_number,
                'coordinates': ordered_markers
            })
            processed_frames += 1

        print(f"Processed {processed_frames} frames out of {frame_number} frames read", end='\r')

    # Release the video capture object
    cap.release()

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the results to a file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {output_path}")


def map_coordinates_video(pred, frame_width, frame_height):
    """
    Maps normalized predictions to original image coordinates.
    
    Args:
        pred: List of predictions.
        frame_width: Width of the frame.
        frame_height: Height of the frame.
    
    Returns:
        Mapped predictions in the original image coordinates.
    """
    mapped_pred = []
    
    for p in pred:
        cls, x_center, y_center, width, height = p
        x_center = int(x_center * frame_width)
        y_center = int(y_center * frame_height)
        width = int(width * frame_width)
        height = int(height * frame_height)
        
        mapped_pred.append({
            'marker': cls,
            'x': x_center,
            'y': y_center,
            'width': width,
            'height': height
        })
    
    return mapped_pred

def visualize_video_with_detections(model, video_path, output_path):
 

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    results = []

    # Get the original video resolution
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (original_width, original_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        frame_height, frame_width = frame.shape[:2]

        # Apply YOLOv8 on the frame
        detections = model.predict(source=frame)
        
        # Extract and map predictions to original frame coordinates
        all_detections = []
        for detection in detections:
            for box in detection.boxes:
                cls = int(box.cls)
                x_center, y_center, width, height = box.xywhn[0]
                all_detections.append([cls, x_center, y_center, width, height])
        
        mapped_detections = map_coordinates_video(all_detections, frame_width, frame_height)
        
        # Draw circles around the detected markers
        draw_circles(frame, mapped_detections)
        
        # Write the frame to the output video
        out.write(frame)
        
        # Display the frame in a smaller window
        display_frame = cv2.resize(frame, (960, 720))  # Resize for display purposes
        cv2.imshow('Video with Detections', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Output video saved to {output_path}")


def angle_between_vectors(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag_v1 = math.sqrt(sum(a ** 2 for a in v1))
    mag_v2 = math.sqrt(sum(a ** 2 for a in v2))
    # Clamp the value to avoid domain errors
    cos_theta = dot_product / (mag_v1 * mag_v2)
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    return math.degrees(math.acos(cos_theta))


def calculate_rectangle_angles(points):
    if len(points) != 4:
        raise ValueError("Four points are required to form a rectangle.")
    
    ordered_points = order_points(points)
    p1, p2, p3, p4 = ordered_points

    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    v3 = (p4[0] - p3[0], p4[1] - p3[1])
    v4 = (p1[0] - p4[0], p1[1] - p4[1])

    angle1 = angle_between_vectors(v1, v2)
    angle2 = angle_between_vectors(v2, v3)
    angle3 = angle_between_vectors(v3, v4)
    angle4 = angle_between_vectors(v4, v1)

    return [angle1, angle2, angle3, angle4]

def get_best_frames_from_json(json_file_path, num_best_frames=1):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    frame_scores = []

    for frame in data:
        markers = frame.get('coordinates', None)
        if markers is None or len(markers) != 4:
            continue  # Ensure exactly 4 markers

        points = [(m[0], m[1]) for m in markers]
        try:
            angles = calculate_rectangle_angles(points)
            angle_deviation = sum(abs(90 - angle) for angle in angles)
            frame_scores.append((angle_deviation, frame))
        except ValueError as e:
            print(f"Error in frame {frame['frame_number']}: {e}")
            continue

    frame_scores.sort(key=lambda x: x[0])
    best_frames = [frame for _, frame in frame_scores[:num_best_frames]]

    return best_frames

def plot_quadrilateral(frame):
    """
    Plot the quadrilateral defined by the coordinates in the frame.
    
    Parameters:
    frame (dict): The frame data including 'coordinates'.
    
    Returns:
    None
    """
    coords = frame['coordinates']
    polygon = plt.Polygon(coords, closed=True, edgecolor='r', fill=None)
    plt.gca().add_patch(polygon)
    plt.scatter(*zip(*coords), color='b')
    plt.text(coords[0][0], coords[0][1], f"Frame: {frame['frame_number']}", fontsize=12, ha='right')


def display_frame(image, frame_number):
    """
    Displays a frame using matplotlib.

    Parameters:
    image (numpy.ndarray): The frame image to display.
    frame_number (int): The frame number.
    """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Frame {frame_number}")
    plt.axis('off')  # Hide axes
    plt.show()
    

def extract_and_return_frames(video_path, output_folder, frame_numbers):
    """
    Extracts and returns specific frames from a video file as JPEG files.

    Parameters:
    video_path (str): Path to the MP4 video file.
    output_folder (str): Folder to save the extracted frames.
    frame_numbers (list of int): List of frame numbers to extract and save.

    Returns:
    dict: A dictionary where keys are frame numbers and values are the extracted frames.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check if all specified frame numbers are within the valid range
    for frame_number in frame_numbers:
        if frame_number < 0 or frame_number >= total_frames:
            vidcap.release()
            raise ValueError(f"Frame number {frame_number} is out of range. The video has {total_frames} frames.")

    frames = {}
    count = 0
    success, image = vidcap.read()

    while success:
        if count in frame_numbers:
            frame_filename = os.path.join(output_folder, f"frame{count}.jpg")
            cv2.imwrite(frame_filename, image)  # save specific frame as JPEG file
            frames[count] = image
            print(f"Saved frame {count} to {frame_filename}")
            if len(frames) == len(frame_numbers):
                break  # Exit the loop after saving all specified frames
        success, image = vidcap.read()
        count += 1

    vidcap.release()
    return frames