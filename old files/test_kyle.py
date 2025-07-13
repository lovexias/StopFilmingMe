import cv2
from utilities import detect_multiple_skeletons_yolov8

# Path to your test video
video_path = "D:\\testing.mp4"

# Predefined list of unique colors (RGB format)
COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Purple
    (0, 128, 255),  # Orange
]

# Load video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Failed to open video: {video_path}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    detections = detect_multiple_skeletons_yolov8(frame)

    detections = detect_multiple_skeletons_yolov8(frame, conf_threshold=0.5)

    for idx, (keypoints, bbox) in enumerate(detections):
        color = COLORS[idx % len(COLORS)]

        # ─── Draw keypoints ────────────────────────
        for x, y in keypoints:
            cv2.circle(frame, (x, y), 4, color, -1)

        # ─── Draw skeleton lines (COCO-style) ──────
        skeleton_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (1, 5), (5, 6), (6, 7),
            (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16)
        ]
        for i, j in skeleton_edges:
            if i < len(keypoints) and j < len(keypoints):
                cv2.line(frame, keypoints[i], keypoints[j], color, 2)

        # ─── Draw bounding box ─────────────────────
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # OPTIONAL: draw label with person index
            label = f"Person {idx+1}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    cv2.imshow("YOLOv8 Pose - Skeleton + BBox", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()