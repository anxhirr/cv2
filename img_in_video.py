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
frame_skip = 10 # check every frame for accurate timing

# -----------------------------
# Load template
# -----------------------------
logo = cv2.imread(logo_path, 0)
logo_mod = cv2.GaussianBlur(logo, (3, 3), 0)

# Pre-compute all resized templates ONCE
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
print(f"Checking every {frame_skip} frames = {total_frames//frame_skip} frames to process")
print(f"Threshold: {threshold}, Scales: {len(scales)}\n")

detections = []
match_template_time = 0
detecting = False
start_frame = None
last_best_scale_idx = len(scales) // 2  # Start with middle scale
best_values_log = []  # Track all best values

# Process only the frames we need
frames_to_check = range(0, total_frames, frame_skip)

for frame_num in tqdm(frames_to_check, desc="Frames processed"):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale (skip blur unless necessary)
    frame_mod = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_mod = cv2.GaussianBlur(frame_mod, (3, 3), 0)
    
    best_val = 0
    best_scale_idx = last_best_scale_idx
    
    # Smart scale search: check last successful scale first, then neighbors
    scale_order = [last_best_scale_idx]
    for offset in range(1, len(scales)):
        if last_best_scale_idx + offset < len(scales):
            scale_order.append(last_best_scale_idx + offset)
        if last_best_scale_idx - offset >= 0:
            scale_order.append(last_best_scale_idx - offset)
    
    for idx in scale_order:
        scale, template, w, h = resized_logos[idx]
        
        if h > frame_mod.shape[0] or w > frame_mod.shape[1]:
            continue
        
        start_time = time.perf_counter()
        result = cv2.matchTemplate(frame_mod, template, cv2.TM_CCOEFF_NORMED)
        match_template_time += time.perf_counter() - start_time
        
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_val:
            best_val = max_val
            best_scale_idx = idx
        
        # Early exit if we found a strong match
        if max_val > threshold + 0.1:
            break
    
    last_best_scale_idx = best_scale_idx
    best_values_log.append((frame_num, best_val))
    
    # Detection tracking
    if best_val >= threshold:
        if not detecting:
            detecting = True
            start_frame = frame_num
    else:
        if detecting:
            end_frame = frame_num - frame_skip
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = end_time - start_time
            detections.append((start_time, end_time, duration))
            detecting = False

# Handle ongoing detection at video end
if detecting:
    end_frame = frame_num
    start_time = start_frame / fps
    end_time = end_frame / fps
    duration = end_time - start_time
    detections.append((start_time, end_time, duration))

cap.release()

# -----------------------------
# Print results
# -----------------------------
print("\n✅ Detection summary:")
if detections:
    for idx, (start, end, dur) in enumerate(detections, 1):
        print(f"{idx}. Logo detected from {start:.2f}s to {end:.2f}s ({dur:.2f} seconds)")
else:
    print("No detections found.")

# Print best value statistics
print("\n📊 Best Match Value Statistics:")
if best_values_log:
    best_vals = [val for _, val in best_values_log]
    max_val = max(best_vals)
    min_val = min(best_vals)
    avg_val = np.mean(best_vals)
    
    print(f"Maximum match value: {max_val:.4f}")
    print(f"Minimum match value: {min_val:.4f}")
    print(f"Average match value: {avg_val:.4f}")
    print(f"Current threshold: {threshold}")
    
    # Show frame with best match
    best_frame_num, best_frame_val = max(best_values_log, key=lambda x: x[1])
    print(f"\n🎯 Best match found at frame {best_frame_num} ({best_frame_num/fps:.2f}s) with value: {best_frame_val:.4f}")
    
    # Show top 5 matches
    sorted_vals = sorted(best_values_log, key=lambda x: x[1], reverse=True)
    print("\nTop 5 matches:")
    for i, (frame, val) in enumerate(sorted_vals[:5], 1):
        print(f"  {i}. Frame {frame} ({frame/fps:.2f}s): {val:.4f}")

total_processed = len(frames_to_check)
print(f"\nmatchTemplate time: {match_template_time:.2f}s ({match_template_time/total_processed*1000:.2f}ms per frame)")