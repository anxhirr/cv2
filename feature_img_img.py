"""
Precise feature matching example:
 - SIFT feature detection + descriptors
 - FLANN KNN matching
 - Lowe's ratio test
 - Symmetric (mutual) check
 - RANSAC homography to remove outliers

Usage:
    python feature_img_img.py
"""

import cv2
from cv2.typing import MatLike
import numpy as np
from typing import List, Sequence
from matplotlib import pyplot

RATIO = 0.7
MIN_MATCH_COUNT = 10

img1 = cv2.imread("source.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("logo2.png", cv2.IMREAD_GRAYSCALE)

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

def draw_matches(kps1, kps2, matches):
    out = cv2.drawMatches(img1, kps1, img2, kps2, matches, None)
    pyplot.imshow(out)
    pyplot.show()


def match_images():
    kps1, desc1 = detect_and_compute(img1)
    kps2, desc2 = detect_and_compute(img2)

    # 1) KNN matches 1->2 and 2->1
    knn12 = flann_knn_match(desc1, desc2)
    knn21 = flann_knn_match(desc2, desc1)

    # 2) Lowe ratio test in both directions
    good12 = lowe_ratio_filter(knn12)
    good21 = lowe_ratio_filter(knn21)

    # 3) Symmetric mutual check (keeps only mutual matches)
    mutual = symmetric_check(good12, good21)

    # 4) RANSAC homography filtering to remove geometric outliers
    matches, H, mask = filter_with_ransac(kps1, kps2, mutual)

    return matches, H, mask, kps1, kps2

def main():
    matches, H, mask, kps1, kps2 = match_images()
    print(f"Final matches: {len(matches)}")

    if len(matches) > 0:
        draw_matches(kps1, kps2, matches)

if __name__ == "__main__":
    main()
