import cv2
import numpy as np

source = cv2.imread("source.png")
source_mod = cv2.GaussianBlur(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), (3, 3), 0)
logo_mod = cv2.GaussianBlur(cv2.imread("logo2.png", 0), (3, 3), 0)

best_match = None
best_conf = 0

for scale in np.linspace(0.1, 2.0, 50):
    scaled = cv2.resize(logo_mod, None, fx=scale, fy=scale)
    if scaled.shape[0] > source_mod.shape[0] or scaled.shape[1] > source_mod.shape[1]:
        continue
    
    _, max_val, _, max_loc = cv2.minMaxLoc(cv2.matchTemplate(source_mod, scaled, cv2.TM_CCOEFF_NORMED))
    
    if max_val > best_conf and max_val >= 0.8:
        best_conf = max_val
        h, w = scaled.shape
        best_match = (max_loc, (max_loc[0] + w, max_loc[1] + h))

if best_match:
    cv2.rectangle(source, best_match[0], best_match[1], (0, 0, 255), 2)
    print(f"Match found! Confidence: {best_conf:.4f}")

cv2.imwrite("result.png", source)
cv2.imshow("Result", source)
cv2.waitKey(0)
cv2.destroyAllWindows()
