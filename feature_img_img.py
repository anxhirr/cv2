"""
Precise feature matching example:
 - SIFT feature detection + descriptors
 - FLANN KNN matching
 - Lowe's ratio test
 - Symmetric (mutual) check
 - RANSAC homography to remove outliers
 - Optional subpixel refinement of keypoints

Usage:
    python precise_feature_matching.py left.jpg right.jpg
"""

import cv2
import numpy as np
import sys
from typing import List, Tuple

def detect_and_compute(img: np.ndarray, nfeatures=0):
    # Convert to gray if needed
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use SIFT (requires opencv-contrib)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kps, desc = sift.detectAndCompute(gray, None)
    return kps, desc

def flann_knn_match(desc1: np.ndarray, desc2: np.ndarray, k=2):
    # FLANN params for SIFT (floating-point descriptors)
    index_params = dict(algorithm=1, trees=5)  # KDTree
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=k)
    return matches

def lowe_ratio_filter(knn_matches: List[List[cv2.DMatch]], ratio=0.75):
    good = []
    for m_n in knn_matches:
        if len(m_n) < 2:
            continue
        m, n = m_n[0], m_n[1]
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def symmetric_check(matches12: List[cv2.DMatch], matches21: List[cv2.DMatch]):
    # Build dict from query->train for both directions
    map12 = { (m.queryIdx, m.trainIdx): m for m in matches12 }
    map21 = { (m.trainIdx, m.queryIdx): m for m in matches21 }  # reversed
    mutual = []
    for (q, t), m in map12.items():
        if (q, t) in map21:
            mutual.append(m)  # accept match from 12 direction
    return mutual

def filter_with_ransac(kps1, kps2, matches, ransac_thresh=5.0, min_inliers=8):
    if len(matches) < 4:
        return [], None, None  # cannot compute homography
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    if mask is None:
        return [], H, None
    mask = mask.ravel().tolist()
    inlier_matches = [m for m, inlier in zip(matches, mask) if inlier]
    if len(inlier_matches) < min_inliers:
        return [], H, mask
    return inlier_matches, H, mask

def draw_matches(img1, img2, kps1, kps2, matches, title="Matches", max_draw=100):
    out = cv2.drawMatches(img1, kps1, img2, kps2, matches[:max_draw], None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow(title, out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def match_images_precise(img1: np.ndarray, img2: np.ndarray,
                         ratio=0.75, ransac_thresh=5.0):
    kps1, desc1 = detect_and_compute(img1)
    kps2, desc2 = detect_and_compute(img2)

    if desc1 is None or desc2 is None:
        return [], None, None, kps1, kps2

    # 1) KNN matches 1->2 and 2->1
    knn12 = flann_knn_match(desc1, desc2, k=2)
    knn21 = flann_knn_match(desc2, desc1, k=2)

    # 2) Lowe ratio test in both directions
    good12 = lowe_ratio_filter(knn12, ratio=ratio)
    good21 = lowe_ratio_filter(knn21, ratio=ratio)

    # 3) Symmetric mutual check (keeps only mutual matches)
    mutual = symmetric_check(good12, good21)

    # 4) RANSAC homography filtering to remove geometric outliers
    inliers, H, mask = filter_with_ransac(kps1, kps2, mutual, ransac_thresh=ransac_thresh)

    return inliers, H, mask, kps1, kps2

def main(argv):
    if len(argv) < 3:
        print("Usage: python precise_feature_matching.py <img1> <img2>")
        return
    img1 = cv2.imread(argv[1])
    img2 = cv2.imread(argv[2])

    inliers, H, mask, kps1, kps2 = match_images_precise(img1, img2, ratio=0.7, ransac_thresh=4.0)
    print(f"Kept matches (inliers after RANSAC): {len(inliers)}")

    if len(inliers) > 0:
        draw_matches(img1, img2, kps1, kps2, inliers, title="Precise Matches (inliers)")

if __name__ == "__main__":
    main(sys.argv)
