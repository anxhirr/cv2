import cv2
import numpy as np
from pathlib import Path

def detect_logo_in_frame(logo, logo_descriptors, kp_logo, frame, 
                          min_match_count=10, ratio_threshold=0.7, 
                          homography_reproj_threshold=5.0):
    """
    Detect logo in a single video frame.
    
    Args:
        logo: Logo image (grayscale)
        logo_descriptors: Pre-computed SIFT descriptors for logo
        kp_logo: Pre-computed keypoints for logo
        frame: Video frame to search in
        min_match_count: Minimum number of good matches required
        ratio_threshold: Lowe's ratio test threshold
        homography_reproj_threshold: RANSAC reprojection threshold
    
    Returns:
        tuple: (found, coordinates)
    """
    # Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors for frame
    kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)
    
    if des_frame is None:
        return False, None
    
    # Use FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Find matches
    matches = flann.knnMatch(logo_descriptors, des_frame, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    # Check if we have enough matches
    if len(good_matches) < min_match_count:
        return False, None
    
    # Extract location of good matches
    src_pts = np.float32([kp_logo[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography using RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, homography_reproj_threshold)
    
    if M is None:
        return False, None
    
    # Count inliers
    inliers = np.sum(mask)
    inlier_ratio = inliers / len(good_matches)
    
    # Require high inlier ratio
    if inlier_ratio < 0.7:
        return False, None
    
    # Get corners of logo in frame
    h, w = logo.shape
    logo_corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    frame_corners = cv2.perspectiveTransform(logo_corners, M)
    
    return True, frame_corners


def process_video(logo_path, video_path, output_path=None, 
                  min_match_count=10, ratio_threshold=0.7,
                  homography_reproj_threshold=5.0,
                  skip_frames=1, display_live=False):
    """
    Process video and detect logo in frames.
    
    Args:
        logo_path: Path to the logo image
        video_path: Path to the input video
        output_path: Path to save output video (optional)
        min_match_count: Minimum number of good matches required
        ratio_threshold: Lowe's ratio test threshold
        homography_reproj_threshold: RANSAC reprojection threshold
        skip_frames: Process every Nth frame (1 = process all frames)
        display_live: Show live preview during processing
    
    Returns:
        list: Frame numbers where logo was detected
    """
    # Load logo
    logo = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
    if logo is None:
        raise ValueError(f"Could not load logo from {logo_path}")
    
    # Pre-compute logo features (only once)
    print("Computing logo features...")
    sift = cv2.SIFT_create()
    kp_logo, des_logo = sift.detectAndCompute(logo, None)
    
    if des_logo is None:
        raise ValueError("Could not extract features from logo")
    
    print(f"Logo features: {len(kp_logo)} keypoints")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    print(f"  Processing every {skip_frames} frame(s)\n")
    
    # Setup video writer if output path specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video
    detected_frames = []
    frame_num = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_num % skip_frames == 0:
                # Detect logo
                found, corners = detect_logo_in_frame(
                    logo, des_logo, kp_logo, frame,
                    min_match_count=min_match_count,
                    ratio_threshold=ratio_threshold,
                    homography_reproj_threshold=homography_reproj_threshold
                )
                
                if found:
                    detected_frames.append(frame_num)
                    # Draw bounding box
                    cv2.polylines(frame, [np.int32(corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
                    
                    # Add text
                    cv2.putText(frame, "LOGO DETECTED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    print(f"Frame {frame_num}: Logo detected")
            
            # Write frame to output video
            if writer:
                writer.write(frame)
            
            # Display live preview
            if display_live:
                cv2.imshow('Logo Detection', cv2.resize(frame, (960, 540)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break
            
            # Progress update
            if frame_num % (fps * 5) == 0:  # Every 5 seconds
                progress = (frame_num / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_num}/{total_frames} frames)")
            
            frame_num += 1
            
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if display_live:
            cv2.destroyAllWindows()
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Processed {frame_num} frames")
    print(f"Logo detected in {len(detected_frames)} frames")
    
    if detected_frames:
        print(f"\nDetection times:")
        for fn in detected_frames[:10]:  # Show first 10
            timestamp = fn / fps
            print(f"  Frame {fn}: {timestamp:.2f}s")
        if len(detected_frames) > 10:
            print(f"  ... and {len(detected_frames) - 10} more")
    
    if output_path:
        print(f"\nOutput saved to: {output_path}")
    
    return detected_frames


def extract_frames_with_logo(logo_path, video_path, output_dir,
                              min_match_count=10, ratio_threshold=0.7):
    """
    Extract and save individual frames where logo is detected.
    
    Args:
        logo_path: Path to the logo image
        video_path: Path to the input video
        output_dir: Directory to save extracted frames
        min_match_count: Minimum number of good matches required
        ratio_threshold: Lowe's ratio test threshold
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load logo and compute features
    logo = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    kp_logo, des_logo = sift.detectAndCompute(logo, None)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frame_num = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        found, corners = detect_logo_in_frame(
            logo, des_logo, kp_logo, frame,
            min_match_count=min_match_count,
            ratio_threshold=ratio_threshold
        )
        
        if found:
            # Draw detection
            cv2.polylines(frame, [np.int32(corners)], True, (0, 255, 0), 3)
            
            # Save frame
            timestamp = frame_num / fps
            filename = f"frame_{frame_num:06d}_time_{timestamp:.2f}s.jpg"
            cv2.imwrite(str(output_dir / filename), frame)
            saved_count += 1
            print(f"Saved: {filename}")
        
        frame_num += 1
    
    cap.release()
    print(f"\nExtracted {saved_count} frames to {output_dir}")


if __name__ == "__main__":
    # Configuration
    logo_path = "logo.png"
    video_path = "source.mp4"
    output_video_path = "result.mp4"
    
    # Option 1: Process entire video and save output
    detected_frames = process_video(
        logo_path=logo_path,
        video_path=video_path,
        output_path=output_video_path,
        min_match_count=8,
        ratio_threshold=0.65,
        homography_reproj_threshold=3.0,
        skip_frames=2,  # Process every 2nd frame for speed
        display_live=False  # Set True to see live preview
    )
    
    # Option 2: Extract frames where logo appears
    # extract_frames_with_logo(
    #     logo_path=logo_path,
    #     video_path=video_path,
    #     output_dir="detected_frames",
    #     min_match_count=8,
    #     ratio_threshold=0.65
    # )