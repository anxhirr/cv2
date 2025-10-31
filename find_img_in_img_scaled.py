import cv2
import numpy as np

source = cv2.imread("source.png")
source_mod = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
source_mod = cv2.GaussianBlur(source_mod, (3, 3), 0)

logo = cv2.imread("Capture.png", 0)
logo_mod = cv2.GaussianBlur(logo, (3, 3), 0)

# Multi-scale template matching - search different scales
threshold = 0.8
best_match = None
best_confidence = 0

scales = np.linspace(0.1, 2.0, 50)

for scale in scales:
    scaled_logo = cv2.resize(logo_mod, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # Skip if template is larger than source
    if scaled_logo.shape[0] > source_mod.shape[0] or scaled_logo.shape[1] > source_mod.shape[1]:
        continue
    
    result = cv2.matchTemplate(source_mod, scaled_logo, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if max_val > best_confidence and max_val >= threshold:
        best_confidence = max_val
        w, h = scaled_logo.shape[::-1]
        best_match = (max_loc, (max_loc[0] + w, max_loc[1] + h))

# Draw rectangle if match found
if best_match:
    cv2.rectangle(source, best_match[0], best_match[1], (0, 0, 255), 2)
    print(f"Match found! Confidence: {best_confidence:.4f}")
else:
    print("No match found above threshold")

# Save the result image
output_filename = "result.png"
cv2.imwrite(output_filename, source)
print(f"Result saved to: {output_filename}")

# Show the result
cv2.imshow("Detected Logo", source)
cv2.waitKey(0)
cv2.destroyAllWindows()
