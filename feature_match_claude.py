import cv2
import numpy as np

def detect_logo(logo_path, source_path, min_match_count=10, ratio_threshold=0.7, homography_reproj_threshold=5.0, template_threshold=0.6):
    """
    Detect a logo in a source image using SIFT feature detection.
    
    Args:
        logo_path: Path to the logo image
        source_path: Path to the source image to search in
        min_match_count: Minimum number of good matches required (default: 10)
        ratio_threshold: Lowe's ratio test threshold (default: 0.7, lower = stricter)
        homography_reproj_threshold: RANSAC reprojection threshold (default: 5.0, lower = stricter)
        template_threshold: Template matching correlation threshold (default: 0.6, higher = stricter)
    
    Returns:
        tuple: (found, coordinates, visualization)
               found: Boolean indicating if logo was detected
               coordinates: Corner points of detected logo (or None)
               visualization: Image with detection visualized
    """
    
    # Load images
    logo = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
    source = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
    source_color = cv2.imread(source_path)
    
    if logo is None or source is None:
        raise ValueError("Could not load images. Check file paths.")
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    kp_logo, des_logo = sift.detectAndCompute(logo, None)
    kp_source, des_source = sift.detectAndCompute(source, None)
    
    if des_logo is None or des_source is None:
        print("No features detected in one or both images")
        return False, None, source_color
    
    # Use FLANN matcher for efficient matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Find matches using KNN (k=2 for ratio test)
    matches = flann.knnMatch(des_logo, des_source, k=2)
    
    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    print(f"Found {len(good_matches)} good matches (minimum required: {min_match_count})")
    
    # Check if we have enough matches
    if len(good_matches) < min_match_count:
        print("Not enough good matches found. Logo not detected.")
        return False, None, source_color
    
    # Extract location of good matches
    src_pts = np.float32([kp_logo[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_source[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography using RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, homography_reproj_threshold)
    
    if M is None:
        print("Could not compute homography. Logo not detected.")
        return False, None, source_color
    
    # Count inliers (matches that fit the homography)
    inliers = np.sum(mask)
    inlier_ratio = inliers / len(good_matches)
    
    print(f"Inliers: {inliers}/{len(good_matches)} ({inlier_ratio*100:.1f}%)")
    
    # Require high inlier ratio to avoid false positives
    if inlier_ratio < 0.7:  # At least 70% of matches should be inliers
        print("Inlier ratio too low. Likely a false match.")
        return False, None, source_color
    
    # Get corners of logo in source image
    h, w = logo.shape
    logo_corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    source_corners = cv2.perspectiveTransform(logo_corners, M)
    
    # Additional validation using template matching
    if not validate_detection(logo, source, source_corners, M):
        print("Template matching validation failed. Logo not detected.")
        return False, None, source_color
    
    # Draw detection on source image
    result_img = source_color.copy()
    cv2.polylines(result_img, [np.int32(source_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
    
    # Draw matches for visualization
    matches_img = cv2.drawMatches(
        logo, kp_logo, source, kp_source, 
        [m for i, m in enumerate(good_matches) if mask[i]], 
        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    print("Logo detected successfully!")
    return True, source_corners, result_img

def validate_detection(logo, source, corners, homography, threshold=0.6):
    """
    Validate detection using template matching on the warped region.
    
    Args:
        logo: Logo image (grayscale)
        source: Source image (grayscale)
        corners: Detected corner points in source image
        homography: Homography matrix from feature matching
        threshold: Minimum correlation coefficient (0-1, default: 0.6)
    
    Returns:
        bool: True if template matching confirms the detection
    """
    try:
        # Extract the detected region by warping it back to logo's perspective
        h, w = logo.shape
        
        # Warp the detected region to match logo orientation/scale
        warped = cv2.warpPerspective(source, np.linalg.inv(homography), (w, h))
        
        # Normalize both images for better comparison
        logo_norm = cv2.normalize(logo, None, 0, 255, cv2.NORM_MINMAX)
        warped_norm = cv2.normalize(warped, None, 0, 255, cv2.NORM_MINMAX)
        
        # Use normalized cross-correlation for template matching
        result = cv2.matchTemplate(warped_norm, logo_norm, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        print(f"Template matching score: {max_val:.3f} (threshold: {threshold})")
        
        # High correlation indicates good match
        if max_val >= threshold:
            return True
        else:
            print(f"Template matching score too low: {max_val:.3f} < {threshold}")
            return False
            
    except Exception as e:
        print(f"Template matching validation error: {e}")
        return False


if __name__ == "__main__":
    # Usage example
    logo_path = "logo.png"
    source_path = "source.png"
    
    try:
        found, coordinates, result_image = detect_logo(
            logo_path, 
            source_path,
            min_match_count=4,      # Increase for stricter matching
            ratio_threshold=0.65,     # Lower = stricter (0.6-0.8 recommended)
            homography_reproj_threshold=3.0  # Lower = stricter
        )
        
        if found:
            print("\n✓ Logo detected!")
            print(f"Coordinates: {coordinates.reshape(-1, 2)}")
            
            # Save result
            cv2.imwrite("detection_result.png", result_image)
            print("Result saved to 'detection_result.png'")
        else:
            print("\n✗ Logo not found in source image")
            
    except Exception as e:
        print(f"Error: {e}")