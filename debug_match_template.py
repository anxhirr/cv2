import cv2
import numpy as np
from tqdm import tqdm
import time
import os

# -----------------------------
# User settings
# -----------------------------
video_path = "source.mp4"
logo_path = "logo7.PNG"
output_debug_dir = "debug_frames"
threshold = 0.8
scales = np.logspace(np.log10(0.5), np.log10(1.5), 20)  # 20 scales
frame_skip = 10  # check every frame

# -----------------------------
# Prepare directories
# -----------------------------
os.makedirs(output_debug_dir, exist_ok=True)

# -----------------------------
# Load template
# -----------------------------
logo = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
logo_mod = cv2.GaussianBlur(logo, (3, 3), 0)

# Precompute all resized templates
resized_logos = []
for scale in scales:
    resized = cv2.resize(logo_mod, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = resized.shape
    resized_logos.append((scale, resized, w, h))

# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Processing video: {video_path}")
print(f"Total frames: {total_frames}, FPS: {fps}")
print(f"Threshold: {threshold}, Scales: {len(scales)}\n")

detections = []
match_template_time = 0
detecting = False
start_frame = None
last_best_scale_idx = len(scales) // 2

frames_to_check = range(0, total_frames, frame_skip)

# -----------------------------
# Process frames
# -----------------------------
for frame_num in tqdm(frames_to_check, desc="Frames processed"):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    best_val = 0
    best_loc = (0, 0)
    best_scale_idx = last_best_scale_idx

    scale_order = [last_best_scale_idx]
    for offset in range(1, len(scales)):
        if last_best_scale_idx + offset < len(scales):
            scale_order.append(last_best_scale_idx + offset)
        if last_best_scale_idx - offset >= 0:
            scale_order.append(last_best_scale_idx - offset)

    # -----------------------------
    # Try all scales
    # -----------------------------
    for idx in scale_order:
        scale, template, w, h = resized_logos[idx]
        if h > gray.shape[0] or w > gray.shape[1]:
            continue

        start_time = time.perf_counter()
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        match_template_time += time.perf_counter() - start_time

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_scale_idx = idx

        # Early stop if strong match
        if max_val > threshold + 0.1:
            break

    last_best_scale_idx = best_scale_idx

    # -----------------------------
    # Detection tracking
    # -----------------------------
    if best_val >= threshold:
        if not detecting:
            detecting = True
            start_frame = frame_num
    else:
        if detecting:
            end_frame = frame_num - frame_skip
            detections.append((start_frame / fps, end_frame / fps, (end_frame - start_frame) / fps))
            detecting = False

    # -----------------------------
    # Save debug frames (all of them)
    # -----------------------------
    scale, template, w, h = resized_logos[best_scale_idx]
    top_left = best_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 1️⃣ Full frame with box
    frame_debug = frame.copy()
    cv2.rectangle(frame_debug, top_left, bottom_right, (0, 0, 255), 2)
    text = f"Frame {frame_num}, scale={scale:.2f}, val={best_val:.3f}"
    cv2.putText(frame_debug, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(output_debug_dir, f"debug_frame_{frame_num}.png"), frame_debug)

    # 2️⃣ Heatmap
    heatmap = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_debug_dir, f"heatmap_{frame_num}.png"), heatmap)

    # 3️⃣ Template vs ROI
    roi = gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    if roi.size > 0:
        combined = np.hstack([
            cv2.resize(template, (w, h)),
            roi
        ])
        cv2.imwrite(os.path.join(output_debug_dir, f"template_vs_roi_{frame_num}.png"), combined)

# -----------------------------
# Wrap up
# -----------------------------
cap.release()

if detecting:
    end_frame = frame_num
    detections.append((start_frame / fps, end_frame / fps, (end_frame - start_frame) / fps))

# -----------------------------
# Print results
# -----------------------------
print("\n✅ Detection summary:")
if detections:
    for i, (start, end, dur) in enumerate(detections, 1):
        print(f"{i}. Logo detected from {start:.2f}s to {end:.2f}s ({dur:.2f} seconds)")
else:
    print("No detections found.")

print(f"\nmatchTemplate total time: {match_template_time:.2f}s")
print(f"All debug frames saved in '{output_debug_dir}/'")
