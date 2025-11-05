#!/usr/bin/env python3
from cv2.typing import MatLike
import cv2, numpy as np, matplotlib.pyplot as plt
from typing import List, Sequence

import cv2
from cv2.typing import MatLike
import numpy as np
from typing import List, Sequence
from matplotlib import pyplot

# ------------------- MATCHING CORE -------------------
RATIO = 0.7 
MIN_MATCH_COUNT = 5

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

def video_generator(path):
    cap = cv2.VideoCapture(path)
    idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret: break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        yield idx, gray, bgr
        idx += 1
    cap.release()

if __name__ == "__main__":
    logo = cv2.imread("logo2.png", cv2.IMREAD_GRAYSCALE)
    kps_l, desc_l = detect_and_compute(logo)

    for idx, gray, bgr in video_generator("source.mp4"):
        kps_f, desc_f = detect_and_compute(gray)

        good12 = lowe_ratio_filter(flann_knn_match(desc_l, desc_f))
        good21 = lowe_ratio_filter(flann_knn_match(desc_f, desc_l))
        mutual = symmetric_check(good12, good21)
        final, H, _ = filter_with_ransac(kps_l, kps_f, mutual)

        print(f"Frame {idx:04d} → {len(final)} inliers")
        if len(final) >= MIN_MATCH_COUNT:
            draw_matches(cv2.cvtColor(logo,cv2.COLOR_GRAY2BGR), kps_l, bgr, kps_f, final,
                         f"Frame {idx} – {len(final)} matches")