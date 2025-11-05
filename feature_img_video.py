#!/usr/bin/env python3
from cv2.typing import MatLike
import cv2, numpy as np, matplotlib.pyplot as plt
from typing import List, Sequence
import os, subprocess
from pathlib import Path
from matplotlib import pyplot

# ------------------- MATCHING CORE -------------------
RATIO = 0.7 
MIN_MATCH_COUNT = 10

def detect_and_compute(img: MatLike):
    sift = cv2.SIFT_create()
    kps, desc = sift.detectAndCompute(img, None)
    return kps, desc

def flann_knn_match(desc1: MatLike, desc2: MatLike):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=3)
    return matches

def lowe_ratio_filter(matches: Sequence[Sequence[cv2.DMatch]]):
    good = []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n[0], m_n[1]
        if m.distance < RATIO * n.distance:
            good.append(m)
    return good

def symmetric_check(matches12: List[cv2.DMatch], matches21: List[cv2.DMatch]):
    map12 = { (m.queryIdx, m.trainIdx): m for m in matches12 }
    map21 = { (m.trainIdx, m.queryIdx): m for m in matches21 } 
    mutual = []
    for (q, t), m in map12.items():
        if (q, t) in map21:
            mutual.append(m)
    return mutual

def filter_with_ransac(kps1, kps2, matches):
    if len(matches) < MIN_MATCH_COUNT:
        return [], None, None
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    mask = mask.ravel().tolist()
    return matches, H, mask

def draw_matches(i1,k1,i2,k2,ms,t):
    out = cv2.drawMatches(i1,k1,i2,k2,ms,None,matchColor=(0,255,0),flags=2)
    plt.figure(figsize=(14,7)); plt.title(t); plt.imshow(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
    plt.show()

# ----------------------------------------------------

def extract_frames_ffmpeg(video_path: str, output_dir: str):
    """
    Extract all frames from a video as grayscale PNGs using FFmpeg.
    Equivalent to IMREAD_GRAYSCALE for every frame.
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "format=gray",           # force grayscale frames
        f"{output_dir}/frame_%05d.png"  # numbered frames
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def video_generator_from_folder(folder: str):
    """
    Generator that yields frames (as grayscale) in order from a folder.
    """
    files = sorted(Path(folder).glob("frame_*.png"))
    for idx, path in enumerate(files):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        yield idx, gray, None  # consistent with previous structure

# ----------------------------------------------------

if __name__ == "__main__":
    video_path = "source.mp4"
    frames_dir = "frames_gray"

    # 1) Extract frames once using FFmpeg (grayscale)
    if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
        print(f"[FFmpeg] Extracting frames from {video_path} → {frames_dir}/")
        extract_frames_ffmpeg(video_path, frames_dir)
    else:
        print(f"[Info] Using existing extracted frames from {frames_dir}/")

    # 2) Load logo
    logo = cv2.imread("logo2.png", cv2.IMREAD_GRAYSCALE)
    kps1, desc1 = detect_and_compute(logo)

    # 3) Process frames
    for idx, gray, _ in video_generator_from_folder(frames_dir):
        kps2, desc2 = detect_and_compute(gray)

        # 1) KNN matches 1->2 and 2->1
        knn12 = flann_knn_match(desc1, desc2)
        knn21 = flann_knn_match(desc2, desc1)

        # 2) Lowe ratio test in both directions   
        good12 = lowe_ratio_filter(knn12)
        good21 = lowe_ratio_filter(knn21)

        # 3) Symmetric mutual check (keeps only mutual matches)
        mutual = symmetric_check(good12, good21)

        # 4) RANSAC homography filtering to remove geometric outliers
        matches, H, _ = filter_with_ransac(kps1, kps2, mutual)

        print(f"Frame {idx:04d} → {len(matches)} matches")

        if len(matches) >= MIN_MATCH_COUNT:
            draw_matches(
                cv2.cvtColor(logo, cv2.COLOR_GRAY2BGR),
                kps1,
                cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                kps2,
                matches,
                f"Frame {idx} – {len(matches)} matches"
            )
