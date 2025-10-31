import cv2
import numpy as np

source = cv2.imread("source.png")
source_mod = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
source_mod = cv2.GaussianBlur(source_mod, (3, 3), 0)

logo = cv2.imread("logo.png", 0)
logo_mod = cv2.GaussianBlur(logo, (3, 3), 0)

w, h = logo_mod.shape[::-1]

# Apply template matching
result = cv2.matchTemplate(source_mod, logo_mod, cv2.TM_CCOEFF_NORMED)

threshold = 0.8
locations = np.where(result >= threshold)

print(f"Number of matches found: {len(locations[0])}")
print(f"Max confidence value: {np.max(result)}")

# Draw rectangles around detected matches
for pt in zip(*locations[::-1]):
    cv2.rectangle(source, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

# Save the result image
output_filename = "result.png"
cv2.imwrite(output_filename, source)
print(f"Result saved to: {output_filename}")

# Show the result
cv2.imshow("Detected Logo", source)
cv2.waitKey(0)
cv2.destroyAllWindows()
