import os
import cv2
import random
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math



def show_image(img, title='Image'):
    """
    Display an image with a given title.

    Parameters:
    img (numpy.ndarray): The image to display. Can be a grayscale or color image.
    title (str): The title for the image.

    Returns:
    None
    """
    plt.figure(figsize=(10, 8))
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()



def apply_random_rotation(image, apply_rotation=True):
    """
    Apply a random rotation and scaling to the image with optional rotation control.

    Parameters:
    image (numpy.ndarray): The image to be processed.
    apply_rotation (bool): If False, the function returns the original image without rotation.

    Returns:
    numpy.ndarray: The rotated and scaled image.
    """
    if not apply_rotation:
        return image  # Return the original image if rotation is not to be applied
    
    # Apply a random rotation between -30 and 30 degrees
    angle = random.uniform(-30, 30)
    # Apply a random scaling factor along the x-axis between 0.8 and 1.2
    scale_x = random.uniform(0.8, 1.2)
    # Apply a random scaling factor along the y-axis between 0.8 and 1.2
    scale_y = random.uniform(0.8, 1.2)

    # Get the height and width of the image
    h, w = image.shape[:2]
    # Define the maximum dimension to create a larger canvas to prevent cropping
    max_dim = int(max(h, w) * 2)
    # Create a large canvas with the maximum dimension and 4 channels (RGBA)
    large_canvas = np.zeros((max_dim, max_dim, 4), dtype=np.uint8)
    # Define the center coordinates of the large canvas
    center_x, center_y = max_dim // 2, max_dim // 2
    # Calculate the offset to position the original image at the center of the canvas
    offset_x, offset_y = center_x - w // 2, center_y - h // 2
    
    # If the image has 3 channels (RGB), convert it to 4 channels (RGBA)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # Place the original image at the center of the large canvas
    large_canvas[offset_y:offset_y + h, offset_x:offset_x + w] = image

    # Define the center of rotation and scaling
    center = (center_x, center_y)
    # Create a scaling matrix
    scaling_matrix = cv2.getRotationMatrix2D(center, 0, 1.0)
    scaling_matrix[0, 0] *= scale_x  # Apply scaling along the x-axis
    scaling_matrix[1, 1] *= scale_y  # Apply scaling along the y-axis
    # Apply the scaling transformation
    elliptical_image = cv2.warpAffine(large_canvas, scaling_matrix, (max_dim, max_dim), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # Create a rotation matrix
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Apply the rotation transformation
    rotated_image = cv2.warpAffine(elliptical_image, rot_matrix, (max_dim, max_dim), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # Find the bounding box of the non-zero regions (i.e., the actual image) to crop the image without the extra black areas
    gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGRA2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)

    cropped_image = rotated_image[y:y+h, x:x+w]

    return cropped_image



def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """
    Overlay img_overlay on top of img at (x, y) and blend using alpha_mask.

    Parameters:
    img (numpy.ndarray): The background image.
    img_overlay (numpy.ndarray): The image to overlay.
    x (int): The x-coordinate where the overlay starts.
    y (int): The y-coordinate where the overlay starts.
    alpha_mask (numpy.ndarray): The alpha mask for blending.
    """
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, None] / 255.0
    img_crop[:] = (1.0 - alpha) * img_crop + alpha * img_overlay_crop


def add_markers_to_image(background, round_markers, square_markers, random_marker_rotation=True):
    """
    Add round and square markers to the background image.

    Parameters:
    background (numpy.ndarray): The background image.
    round_markers (list): List of round marker images.
    square_markers (list): List of square marker images.
    random_marker_rotation (bool): If True, apply random rotation to markers.

    Returns:
    tuple: Augmented image and marker positions.
    """
    augmented_image = background.copy()
    marker_positions = []
    bg_height, bg_width, _ = background.shape
    
    central_area_width = int(bg_width * 0.3)
    central_area_height = int(bg_height * 0.3)
    central_x = (bg_width - central_area_width) // 2
    central_y = (bg_height - central_area_height) // 2
    
    for marker_set, marker_class in zip([round_markers, square_markers], [0, 1]):
        for marker in marker_set:
            if marker.shape[2] == 3:
                marker = cv2.cvtColor(marker, cv2.COLOR_BGR2BGRA)
            
            marker_height_percent = random.uniform(0.01, 0.04)
            new_marker_height = int(bg_height * marker_height_percent)
            aspect_ratio = marker.shape[1] / marker.shape[0]
            new_marker_width = int(new_marker_height * aspect_ratio)
            
            resized_marker = cv2.resize(marker, (new_marker_width, new_marker_height), interpolation=cv2.INTER_AREA)
            
            rotated_marker = apply_random_rotation(resized_marker, apply_rotation=random_marker_rotation)
            h, w, _ = rotated_marker.shape
            
            if random.random() < 0.7:
                x = random.randint(central_x, central_x + central_area_width - w)
                y = random.randint(central_y, central_y + central_area_height - h)
            else:
                x = random.randint(0, bg_width - w)
                y = random.randint(0, bg_height - h)
            
            alpha_mask = rotated_marker[:, :, 3]
            overlay_image_alpha(augmented_image, rotated_marker[:, :, :3], x, y, alpha_mask)
            
            marker_positions.append({'marker': marker_class, 'x': x, 'y': y, 'width': w, 'height': h})
    
    return augmented_image, marker_positions



def apply_random_image_rotation(image):
    """
    Apply a random rotation to an image with a 20% probability.

    The function randomly rotates the image by 90, -90, or 180 degrees with a 20% chance.

    Parameters:
    image (numpy.ndarray): The image to be rotated.

    Returns:
    numpy.ndarray: The rotated image or the original image if no rotation is applied.
    """
    if random.random() < 0.2:
        angle = random.choice([90, -90, 180])
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == -90:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
    return image


def apply_random_brightness_shift(image):
    """
    Apply a random brightness shift to the image.

    Parameters:
    image (numpy.ndarray): The image to be processed.

    Returns:
    numpy.ndarray: The processed image with a random brightness shift applied.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness_shift = random.uniform(0.5, 1.5)
    hsv[:, :, 2] = cv2.convertScaleAbs(hsv[:, :, 2] * brightness_shift)
    brightened_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return brightened_image


# Specification of the main pipeline function
def create_dataset_from_images(n_total_images,
                               num_images_per_bg,
                               num_background_images,
                               round_markers,
                               square_markers,
                               background_folder,
                               output_folder,
                               random_marker_rotation=True,
                               random_image_rotation=True,
                               convert_to_bw=False,
                               apply_edge_detection=False,
                               apply_brightness_shift=False):
    """
    Create a dataset of images with markers.

    Parameters:
    n_total_images (int): Total number of images to create.
    num_images_per_bg (int): Number of images to create per background image.
    num_background_images (int): Number of background images to use.
    round_markers (list): List of round marker images.
    square_markers (list): List of square marker images.
    background_folder (str): Path to the folder containing background images.
    output_folder (str): Path to the folder to save the generated images.
    random_marker_rotation (bool): If True, apply random rotation to markers.
    random_image_rotation (bool): If True, apply random rotation to background images.
    convert_to_bw (bool): If True, convert images to black and white.
    apply_edge_detection (bool): If True, apply edge detection to images.
    apply_brightness_shift (bool): If True, apply random brightness shift to images.

    Returns:
    list: Dataset information containing file names and marker positions.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    dataset_info = []
    background_files = [os.path.join(background_folder, f) for f in os.listdir(background_folder) if f.lower().endswith('.png')]
    
    def pre_process_image(image, output_file_name):
        if convert_to_bw:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if apply_edge_detection:
            image = cv2.Canny(image, 100, 200)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if apply_brightness_shift:
            image = apply_random_brightness_shift(image)
        cv2.imwrite(output_file_name, image)
    
    for i in range(num_background_images):
        if n_total_images <= 0:
            break
        bg_file = random.choice(background_files)
        background_image = cv2.imread(bg_file)
        if background_image is None:
            continue

        if random_image_rotation is True:
            background_image = apply_random_image_rotation(background_image)
                
        output_file_name = f'{os.path.splitext(os.path.basename(bg_file))[0]}_background_{i}.png'
        output_path = os.path.join(output_folder, output_file_name)
        pre_process_image(background_image, output_path)
        
        image_info = {
            'file_name': os.path.basename(output_path),
            'markers': []
        }
        dataset_info.append(image_info)
        n_total_images -= 1

    while n_total_images > 0:
        for bg_file in background_files:
            if n_total_images <= 0:
                break
            
            background_image = cv2.imread(bg_file)
            
            if random_image_rotation is True:
                background_image = apply_random_image_rotation(background_image)
            
            if background_image is None:
                continue
            
            bg_height, bg_width, _ = background_image.shape

            for i in range(num_images_per_bg):
                num_round_markers = random.randint(1, 2)
                num_square_markers = random.randint(1, 2)
                selected_round_markers = random.choices(round_markers, k=num_round_markers)
                selected_square_markers = random.choices(square_markers, k=num_square_markers)
                augmented_image, marker_positions = add_markers_to_image(background_image, selected_round_markers, selected_square_markers,
                                                                         random_marker_rotation=random_marker_rotation)
                
                output_file_name = f'{os.path.splitext(os.path.basename(bg_file))[0]}_augmented_{i}.png'
                output_path = os.path.join(output_folder, output_file_name)
                pre_process_image(augmented_image, output_path)
                
                label_file_name = f'{os.path.splitext(os.path.basename(bg_file))[0]}_augmented_{i}.txt'
                label_path = os.path.join(output_folder, label_file_name)
                
                with open(label_path, 'w') as label_file:
                    for marker in marker_positions:
                        x_center = (marker['x'] + marker['width'] / 2) / bg_width
                        y_center = (marker['y'] + marker['height'] / 2) / bg_height
                        width = marker['width'] / bg_width
                        height = marker['height'] / bg_height
                        label_file.write(f"{marker['marker']} {x_center} {y_center} {width} {height}\n")
                
                image_info = {
                    'file_name': os.path.basename(output_path),
                    'markers': marker_positions
                }
                dataset_info.append(image_info)
                n_total_images -= 1

    return dataset_info