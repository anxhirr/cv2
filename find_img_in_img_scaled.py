import cv2
import numpy as np

# Load images
source = cv2.imread("source.png")
source_mod = cv2.GaussianBlur(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), (3, 3), 0)

logo = cv2.imread("logo5.png", 0)
logo_mod = cv2.GaussianBlur(logo, (3, 3), 0)

# Define scale range and threshold
scales = np.linspace(0.5, 1.3, 25)  # test 25 scales between 50% and 150%
threshold = 0.8

best_val = 0
best_loc = None
best_w, best_h = None, None
best_scale = None

for scale in scales:
    resized = cv2.resize(logo_mod, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = resized.shape

    # Skip if template larger than source
    if h > source_mod.shape[0] or w > source_mod.shape[1]:
        continue

    result = cv2.matchTemplate(source_mod, resized, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    print(f"Scale {scale:.2f}: Max confidence = {max_val:.4f}")

    if max_val > best_val:
        best_val = max_val
        best_loc = max_loc
        best_scale = scale
        best_w, best_h = w, h

# Draw best match
if best_val >= threshold:
    print(f"\n✅ Best match found: {best_val:.3f} at scale {best_scale:.2f}")
    top_left = best_loc
    bottom_right = (top_left[0] + best_w, top_left[1] + best_h)
    cv2.rectangle(source, top_left, bottom_right, (0, 0, 255), 2)
else:
    print(f"\n❌ No match above threshold ({threshold}). Best = {best_val:.3f}")

# Save and show result
cv2.imwrite("result.png", source)
cv2.imshow("Detected Logo", source)
cv2.waitKey(0)
cv2.destroyAllWindows()
