import sys
sys.path.insert(0, '/content/phase1')

from pipeline import Phase1Pipeline, run_baseline
import cv2

pipeline = Phase1Pipeline(
    yolo_model_path = "/content/best.pt",
    class_names     = ["cracks in midportion", "cracks under rail seat", "damage"],
    device          = '0',   # GPU
)

img  = cv2.imread("/content/test_frame.jpg")
dets = pipeline.detect(img)
print(pipeline.compute_fp_metrics(dets))

cv2.imwrite("/content/result.jpg", pipeline.visualize(img, dets))
