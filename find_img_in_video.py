import cv2
import numpy as np

# Load template image
logo = cv2.imread("logo.PNG", 0)
logo_mod = cv2.GaussianBlur(logo, (3, 3), 0)

# Open video file
video_path = "source.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set up video writer for output
output_path = "result_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Define scale range and threshold
scales = np.logspace(np.log10(0.5), np.log10(1.5), 20)  # 20 scales from 50% to 150%
threshold = 0.8

frame_count = 0
match_count = 0

print(f"Processing video: {video_path}")
print(f"Total frames: {total_frames}")
print(f"Video size: {width}x{height}, FPS: {fps}")
print(f"Template image: logo.PNG")
print(f"Threshold: {threshold}\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale and apply Gaussian blur
    frame_mod = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    
    # Multi-scale template matching
    best_val = 0
    best_loc = None
    best_w, best_h = None, None
    best_scale = None
    
    for scale in scales:
        resized = cv2.resize(logo_mod, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = resized.shape
        
        # Skip if template larger than frame
        if h > frame_mod.shape[0] or w > frame_mod.shape[1]:
            continue
        
        result = cv2.matchTemplate(frame_mod, resized, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_scale = scale
            best_w, best_h = w, h
    
    # Draw bounding box if match found
    if best_val >= threshold:
        top_left = best_loc
        bottom_right = (top_left[0] + best_w, top_left[1] + best_h)
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
        match_count += 1
        if frame_count % 10 == 0:  # Print every 10th frame to reduce output
            print(f"Frame {frame_count}: Match found! Confidence: {best_val:.3f} at scale {best_scale:.2f}")
    
    # Write frame to output video
    out.write(frame)
    
    frame_count += 1
    
    # Progress indicator
    if frame_count % 30 == 0:
        progress = (frame_count / total_frames) * 100
        print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames processed)")

# Release resources
cap.release()
out.release()

print(f"\n✅ Processing complete!")
print(f"Total frames processed: {frame_count}")
print(f"Matches found: {match_count} ({match_count/frame_count*100:.1f}% of frames)")
print(f"Output video saved to: {output_path}")

