import sys
import os
import glob
import dlib
from rembg import remove
import cv2
import numpy as np
import env
# Model paths
# env.pedictor_path = "shape_predictor_5_face_landmarks.dat"
# env.face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# Folder paths
# env.faces_folder_base = "entities"  # Base path for input folders
# env.output_folder_base = "entities_process"  # Base path for processed output folders

# Image quality assessment function
def judge_image_quality(image_path):
    """Evaluate image quality and return a score."""
    im = cv2.imread(image_path, 0)  # Read in grayscale
    blurScore = cv2.Laplacian(im, cv2.CV_64F).var()
    return str(round(blurScore))

# Load models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(env.pedictor_path)
facerec = dlib.face_recognition_model_v1(env.face_rec_model_path)

# print("Starting image processing...")

def image_process():
# Process each subfolder
    for subdir in next(os.walk(env.faces_folder_base))[1]:
        print(f"\nProcessing subfolder: {subdir}")
        faces_folder_path = os.path.join(env.faces_folder_base, subdir)
        output_folder_path = os.path.join(env.output_folder_base, subdir + "_process")

        # Ensure the output directory exists
        if not os.path.isdir(output_folder_path):
            os.makedirs(output_folder_path)
            print(f"Creating output directory: {output_folder_path}")

        descriptors = []
        images = []

        # Process each image
        for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):  # Supports multiple image formats
            print(f"Processing image: {f}")
            img = dlib.load_rgb_image(f)
            dets = detector(img, 1)
            print(f"Number of faces detected: {len(dets)}")

            for k, d in enumerate(dets):
                shape = sp(img, d)
                face_descriptor = facerec.compute_face_descriptor(img, shape)
                descriptors.append(face_descriptor)
                images.append((img, shape))

        # Perform clustering
        labels = dlib.chinese_whispers_clustering(descriptors, 0.5)
        num_classes = len(set(labels))
        print(f"Total number of clusters: {num_classes}")
        if num_classes == 0:
            return

        # Find the biggest cluster
        biggest_class_length = max([len([label for label in labels if label == i]) for i in range(num_classes)])
        biggest_class = [i for i in range(num_classes) if len([label for label in labels if label == i]) == biggest_class_length][0]
        print(f"Biggest cluster ID: {biggest_class}, contains {biggest_class_length} faces")

        # Save faces from the biggest cluster
        indices = [i for i, label in enumerate(labels) if label == biggest_class]
        for i, index in enumerate(indices):
            img, shape = images[index]
            file_path = os.path.join(output_folder_path, "face_" + str(i))
            dlib.save_face_chip(img, shape, file_path, size=450, padding=1)
            print(f"Saved face image to: {file_path}")

            # Image quality assessment and background removal
        for f in glob.glob(os.path.join(output_folder_path, "*.jpg")):
            print(f"Processing image: {f}")
            input_path = f

            # First, read the image and assess its quality
            quality_score = float(judge_image_quality(input_path))  # Ensure conversion to a floating-point number for comparison
            print(f"Image quality score: {quality_score}")

            # If image quality is less than 50, skip this image
            if quality_score < 50:
                print("Insufficient image quality, skipping.")
                os.remove(input_path)
                continue  # Skip the current iteration, do not execute the code below

            # Read the image content for background removal
            with open(input_path, 'rb') as i:
                input_image = i.read()

            # Remove the background
            output_image = remove(input_image)

            # Build a new output path, including the quality score
            output_path = input_path.replace(".jpg", f"-{quality_score}-noback.png")

            # Save the processed image to the new path
            with open(output_path, 'wb') as o:
                o.write(output_image)
                print(f"Background removal completed, saved to: {output_path}")

            # Delete the original image
            os.remove(input_path)

# print("\nAll images processed.")
