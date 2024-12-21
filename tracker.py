'''
----------------------------------------
WRITE COORDINATE ON IMAGES
----------------------------------------
'''

from ultralytics import YOLO
import os
import math
import json
import numpy as np
import cv2

# Load YOLO model
model = YOLO("yolov8x.pt")

# Calculate Euclidean distance between two points
def calculate_distance(point1, point2): 
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Main function to check for adults and track them
def check_adult_from_folder(folder_path, output_folder):
    # Read all image files from the folder and sort them
    frame_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
    human_coordinates = {}
    current_id = 0
    all_frame_centroids = []
    first_mean_found = False

    for frame_count, frame_file in enumerate(frame_files, start=1):
        # Read the frame
        img = cv2.imread(frame_file)
        if img is None:
            continue

        results = model.predict(img, verbose=False, classes=[0], conf=0.3)
        
        frame_centroids = []  # Collect centroids for the current frame
        for result in results:
            for box in result.boxes:
                # Bounding box coordinates
                x_min = box.xyxy[0][0].item()
                y_min = box.xyxy[0][1].item()
                x_max = box.xyxy[0][2].item()
                y_max = box.xyxy[0][3].item()

                # Calculate centroid
                centroid_x = int((x_min + x_max) / 2)
                centroid_y = int((y_min + y_max) / 2)
                new_centroid = (centroid_x, centroid_y)
                frame_centroids.append(new_centroid)

                # Assign centroid to a person ID
                assigned_id = None

                if assigned_id is not None:
                    human_coordinates[assigned_id].append(new_centroid)
                else:
                    human_coordinates[current_id] = [new_centroid]
                    current_id += 1

        # Calculate and print the first mean centroid as soon as found
        if frame_centroids and not first_mean_found:
            centroids_array = np.array(frame_centroids)
            first_frame_mean = np.mean(centroids_array, axis=0).tolist()
            print(f"First mean centroid found in frame {frame_count}: {first_frame_mean}")
            first_mean_found = True

        # Append centroids of the current frame to the overall list
        if frame_centroids:
            all_frame_centroids.extend(frame_centroids)

        # Annotate the image with person numbers and centroids
        for person_id, centroids in human_coordinates.items():
            if centroids:
                centroid = centroids[-1]
                cv2.circle(img, centroid, 5, (0, 255, 0), -1)  # Draw the centroid
                cv2.putText(img, f"Person_{person_id}", (centroid[0] + 10, centroid[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Label the person

        # Save the annotated image
        annotated_image_path = os.path.join(output_folder, f"annotated_frame_{frame_count}.jpg")
        cv2.imwrite(annotated_image_path, img)

        # Stop after processing 20 frames
        if frame_count >= 20:
            break

    # Calculate and print the mean of all centroids across all frames
    if all_frame_centroids:
        all_frame_centroids_array = np.array(all_frame_centroids)
        overall_mean = np.mean(all_frame_centroids_array, axis=0).tolist()
        print(f"Overall mean centroids across all frames: {overall_mean}")
    else:
        print("No centroids detected in any frame.")

def process_all_folders(base_folder, output_base_folder):
    # Iterate through all subdirectories in the base folder
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)

        # Check if the current path is a directory
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")

            # Create a corresponding output folder
            output_folder = os.path.join(output_base_folder, folder_name)
            os.makedirs(output_folder, exist_ok=True)

            # Process the folder
            check_adult_from_folder(folder_path, output_folder)



# Base input and output folder paths
base_input_folder = "/home/abhishek/adult_not_found_without_blindspot/"
base_output_folder = "/home/abhishek/output_blindspot/"

# Run the function to process all folders
process_all_folders(base_input_folder, base_output_folder)