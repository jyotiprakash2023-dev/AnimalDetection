# collect_data.py

import cv2
import os

# Paths
input_folder = "images"               # Folder with Kaggle images
output_folder = "crop_image"               # Where cropped faces/animals will be saved
cascade_path = "haarcascade_frontalface_default.xml"  # Change to animal cascade if needed

# Load classifier
classifier = cv2.CascadeClassifier(cascade_path)

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Process each image in the dataset folder
count = 0
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        file_path = os.path.join(input_folder, file_name)
        image = cv2.imread(file_path)

        if image is None:
            print(f"Error reading {file_name}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            count += 1
            save_path = os.path.join(output_folder, f"face_{count}.jpg")
            cv2.imwrite(save_path, face)
            print(f"[{count}] Saved: {save_path}")

print("Done collecting from dataset.")