import csv
import os
import numpy as np
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8x.pt")

# Save a single CSV file for all folders
def save_means_to_csv(csv_file_path, folder_name, first_mean, overall_mean):
    # Check if the CSV file exists; if not, write the header
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["Folder Name", "First Mean Centroid", "Overall Mean Centroid"])
        writer.writerow([folder_name, first_mean, overall_mean])

# Main function to check for adults and track them
def check_adult_from_folder(folder_path, output_folder, csv_file_path):
    # Read all image files from the folder and sort them
    frame_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
    human_coordinates = {}
    detection_counts = {}  # Track detection counts for each person ID
    current_id = 0
    all_frame_centroids = []
    overall_mean_centroid = None

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

                # # Check existing persons for close match
                # for person_id, centroids in human_coordinates.items():
                #     if centroids:
                #         last_centroid = centroids[-1]
                #         distance = np.linalg.norm(np.array(last_centroid) - np.array(new_centroid))
                #         if distance < 50:  # Threshold for matching centroids
                #             assigned_id = person_id
                #             break

                if assigned_id is not None:
                    human_coordinates[assigned_id].append(new_centroid)
                    detection_counts[assigned_id] += 1
                else:
                    human_coordinates[current_id] = [new_centroid]
                    detection_counts[current_id] = 1
                    current_id += 1

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

    # Determine the person with the maximum detections
    if detection_counts:
        max_detections_id = max(detection_counts, key=detection_counts.get)
        max_person_centroids = human_coordinates[max_detections_id]
        first_mean_centroid = np.mean(np.array(max_person_centroids), axis=0).tolist()
        print(f"First mean centroid for person with max detections: {first_mean_centroid}")
    else:
        first_mean_centroid = None
        print("No detections found.")

    # Calculate and save the overall mean centroid
    if all_frame_centroids:
        all_frame_centroids_array = np.array(all_frame_centroids)
        overall_mean_centroid = np.mean(all_frame_centroids_array, axis=0).tolist()
        print(f"Overall mean centroids across all frames: {overall_mean_centroid}")
    else:
        print("No centroids detected in any frame.")

    # Save the results to the single CSV file
    save_means_to_csv(csv_file_path, os.path.basename(folder_path), first_mean_centroid, overall_mean_centroid)

def process_all_folders(base_folder, output_base_folder, output_csv_file):
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
            check_adult_from_folder(folder_path, output_folder, output_csv_file)

# Base input and output folder paths
base_input_folder = "/home/abhishek/adult_not_found_without_blindspot/"
base_output_folder = "/home/abhishek/output_blindspot/"
output_csv_file = "/home/abhishek/output_blindspot/means.csv"

# Run the function to process all folders
process_all_folders(base_input_folder, base_output_folder, output_csv_file)
