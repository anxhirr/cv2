import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# User settings
# -----------------------------
video_path = "source.mp4"
logo_path = "logo.PNG"
output_path = "result_video.mp4"
threshold = 0.8
scales = np.logspace(np.log10(0.5), np.log10(1.5), 20)  # 20 scales from 50% to 150%
frame_skip = 10  # process every 'frame_skip' frame (1 = every frame)

# -----------------------------
# Load template and preprocess
# -----------------------------
logo = cv2.imread(logo_path, 0)
logo_mod = cv2.GaussianBlur(logo, (3, 3), 0)

# Precompute resized templates
resized_templates = []
for scale in scales:
    resized = cv2.resize(logo_mod, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = resized.shape
    resized_templates.append((scale, resized, w, h))

# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Processing video: {video_path}")
print(f"Total frames: {total_frames}, FPS: {fps}, Size: {width}x{height}")
print(f"Threshold: {threshold}, Scales: {len(scales)}, Frame skip: {frame_skip}\n")

frame_count = 0
match_count = 0

# -----------------------------
# Process video
# -----------------------------
for _ in tqdm(range(total_frames), desc="Frames processed"):
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames if needed
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    frame_mod = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)

    best_val = 0
    best_loc = None
    best_w, best_h = None, None
    best_scale = None

    # Multi-scale template matching
    for scale, template, w, h in resized_templates:
        if h > frame_mod.shape[0] or w > frame_mod.shape[1]:
            continue

        result = cv2.matchTemplate(frame_mod, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_w, best_h = w, h
            best_scale = scale

    # Draw bounding box if match exceeds threshold
    if best_val >= threshold:
        match_count += 1
        top_left = best_loc
        bottom_right = (top_left[0] + best_w, top_left[1] + best_h)
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)

    out.write(frame)
    frame_count += 1

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
out.release()

print(f"\n✅ Processing complete!")
print(f"Total frames processed: {frame_count}")
print(f"Matches found: {match_count} ({match_count/frame_count*100:.1f}% of frames)")
print(f"Output video saved to: {output_path}")
