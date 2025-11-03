import cv2
import numpy as np

# Load images
source_img = cv2.imread('source.png')
logo_img = cv2.imread('logo.png')

# Convert to grayscale
source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
logo_gray = cv2.cvtColor(logo_img, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors
kp1, des1 = sift.detectAndCompute(logo_gray, None)
kp2, des2 = sift.detectAndCompute(source_gray, None)

# Use FLANN-based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # Adjust ratio for stricter filtering
        good_matches.append(m)

print(f"Number of good matches: {len(good_matches)}")

MIN_MATCH_COUNT = 5  # Minimum number of good matches to consider detection
if len(good_matches) > MIN_MATCH_COUNT:
    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    h, w = logo_gray.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # Draw detected region
    source_detect = source_img.copy()
    cv2.polylines(source_detect, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('Detected Logo', source_detect)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Not enough matches are found - {len(good_matches)}/{MIN_MATCH_COUNT}")
