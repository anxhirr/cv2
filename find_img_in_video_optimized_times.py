import cv2
import numpy as np
from tqdm import tqdm
import time

# -----------------------------
# User settings
# -----------------------------
video_path = "source.mp4"
logo_path = "logo7.PNG"
threshold = 0.8
scales = np.logspace(np.log10(0.5), np.log10(1.5), 20)  # 20 scales from 50% to 150%
frame_skip = 10  # check every frame for accurate timing
search_region = None  # (y1, y2, x1, x2) or None

# -----------------------------
# Load template and preprocess
# -----------------------------
logo = cv2.imread(logo_path, 0)
logo_mod = cv2.GaussianBlur(logo, (3, 3), 0)

resized_templates = []
for scale in scales:
    resized = cv2.resize(logo_mod, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = resized.shape
    resized_templates.append((scale, resized, w, h))

# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Processing video: {video_path}")
print(f"Total frames: {total_frames}, FPS: {fps}")
print(f"Threshold: {threshold}, Scales: {len(scales)}, Frame skip: {frame_skip}\n")

frame_count = 0
detections = []
match_template_time = 0

# Track ongoing detection
detecting = False
start_frame = None

# -----------------------------
# Process video
# -----------------------------
for _ in tqdm(range(total_frames), desc="Frames processed"):
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    frame_gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)

    if search_region:
        y1, y2, x1, x2 = search_region
        frame_mod = frame_gray[y1:y2, x1:x2]
    else:
        frame_mod = frame_gray

    best_val = 0

    for scale, template, w, h in resized_templates:
        if h > frame_mod.shape[0] or w > frame_mod.shape[1]:
            continue
        start_time = time.perf_counter()
        result = cv2.matchTemplate(frame_mod, template, cv2.TM_CCOEFF_NORMED)
        match_template_time += time.perf_counter() - start_time
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best_val:
            best_val = max_val

    # Check detection status
    if best_val >= threshold:
        if not detecting:
            # New detection starts
            detecting = True
            start_frame = frame_count
    else:
        if detecting:
            # Detection ends
            end_frame = frame_count - 1
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = end_time - start_time
            detections.append((start_time, end_time, duration))
            detecting = False
            start_frame = None

    frame_count += 1

# Handle if video ends while detection is ongoing
if detecting:
    end_frame = frame_count - 1
    start_time = start_frame / fps
    end_time = end_frame / fps
    duration = end_time - start_time
    detections.append((start_time, end_time, duration))

cap.release()

# -----------------------------
# Print detection summary
# -----------------------------
print("\n✅ Detection summary:")
if detections:
    for idx, (start, end, dur) in enumerate(detections, 1):
        print(f"{idx}. Logo detected from {start:.2f}s to {end:.2f}s ({dur:.2f} seconds)")
else:
    print("No detections found.")
print(f"\nmatchTemplate time: {match_template_time:.2f}s ({match_template_time/frame_count*1000:.2f}ms per frame)")
