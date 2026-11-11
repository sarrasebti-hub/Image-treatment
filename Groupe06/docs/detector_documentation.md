# Object Detector Documentation

- **Detector Name:** YOLOv8m (You Only Look Once, version 8, medium)
- **Library Version:** Ultralytics 8.4.207
- **Confidence Threshold:** 0.25
- **Reason for choice:** YOLOv8m offers an excellent balance between speed and accuracy. It processed the 11,100 images in less than 10 minutes on a Tesla T4 GPU. The threshold of 0.25 was chosen because this is an AI-generated detection (unlike COCO captions which are human-made and would use a threshold of 1.0). A 0.25 threshold is the standard baseline for YOLO to balance precision and recall.
- **Empty detections:** 88 images had no detected objects. This is expected for complex, blurry, or abstract scenes and does not require manual fixing. They are simply treated as images with empty metadata.
- **Clean Images:** Resizing and normalization are not saved as separate files; they are handled dynamically during the encoding phase (by Member 3) using CLIP's preprocess pipeline to save disk space.
