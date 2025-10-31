import cv2
import os

# Input video path
video_path = "source.mp4"
output_folder = "frames"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save frame as PNG
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
    cv2.imwrite(frame_filename, frame)
    
    frame_count += 1

cap.release()
print(f"Extracted {frame_count} frames to {output_folder}")
