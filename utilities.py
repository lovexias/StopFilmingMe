import cv2
import subprocess
import json
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Initialize YOLOv8 model globally (adjust path as needed)
yolo_model = YOLO('yolov8m.pt')  # Use regular detection model for person detection

# ──────────────────────────────────────────────────────────────
# ADDITION: Initialize MediaPipe solutions globally for reuse
mp_face_global = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
mp_pose_global = mp.solutions.pose.Pose()

# ──────────────────────────────────────────────────────────────

def detect_multiple_people_yolov8(frame, conf_threshold=0.15):
    """
    Detects multiple people in a frame using YOLOv8 detection model.
    Returns a list of detected person bounding boxes in (x1, y1, x2, y2) format.
    """
    results = yolo_model(frame)[0]
    detections = []
    
    if hasattr(results, "boxes") and results.boxes is not None:
        boxes = results.boxes
        
        for i, box in enumerate(boxes):
            # Check if detection is a person (class 0 in COCO dataset)
            if int(box.cls) == 0 and float(box.conf) >= conf_threshold:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append((int(x1), int(y1), int(x2), int(y2)))

    return detections

# Reads the rotation angle metadata from a video file using ffprobe.
# Returns 0 if no rotation metadata is found.
def get_video_rotation(path):
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream_tags=rotate",
            "-of", "json", path
        ]
        output = subprocess.check_output(cmd).decode("utf-8")
        ffprobe_data = json.loads(output)
        tags = ffprobe_data.get("streams", [{}])[0].get("tags", {})
        return int(tags.get("rotate", 0))
    except:
        return 0

# Generates equally spaced thumbnails from the video for preview UI.
def generate_thumbnails(video_path, total_frames, rotation_angle, num_thumbs=10, thumb_size=(80, 45)):
    thumbs = []
    if total_frames <= 0:
        return thumbs

    step = max(1, total_frames // num_thumbs)
    cap = cv2.VideoCapture(video_path)

    for i in range(num_thumbs):
        frame_idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        if rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(frame_rgb, thumb_size)
        thumbs.append((frame_idx, small.copy()))

    cap.release()
    return thumbs

# Detects wave gestures based only on horizontal movement of hand landmarks (no pose).
class WaveDetector:
    def __init__(self, video_path, fps, detection_confidence=0.8):
        self.video_path = video_path
        self.fps = fps or 30.0
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=detection_confidence
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.drawer = mp.solutions.drawing_utils

    def detect_wave_timestamps(self, show_ui=True, frame_skip=3):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        detected = []
        last_x = None
        movement_history = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            # Remove forced resizing to preserve aspect ratio
            # frame = cv2.resize(frame, (640, 360))  # REMOVED - was causing distortion
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(frame_rgb)
            pose_results = self.pose.process(frame_rgb)

            # Removed unnecessary YOLO call - we already have person crop
            # people = detect_multiple_people_yolov8(frame)

            if hand_results.multi_hand_landmarks and pose_results.pose_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    if show_ui:
                        self.drawer.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                    xs = [lm.x for lm in hand_landmarks.landmark]
                    media_x = sum(xs) / len(xs)
                    direction = None
                    if last_x is not None:
                        threshold = 0.005
                        if media_x < last_x - threshold:
                            direction = "left"
                        elif media_x > last_x + threshold:
                            direction = "right"
                        if direction and (not movement_history or movement_history[-1][0] != direction):
                            movement_history.append((direction, frame_count, pose_results.pose_landmarks.landmark))

                    movement_history = [(d, f, l) for d, f, l in movement_history if frame_count - f <= self.fps]
                    if len(movement_history) >= 4:
                        print(f"Wave detected at frame {frame_count}")
                        detected.append((frame_count, pose_results.pose_landmarks.landmark))
                        movement_history.clear()
                    last_x = media_x

                if show_ui:
                    self.drawer.draw_landmarks(frame, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            if show_ui:
                cv2.imshow("Wave Detection", frame)
                if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                    break

            frame_count += 1

        cap.release()
        self.hands.close()
        self.pose.close()
        if show_ui:
            cv2.destroyAllWindows()

        return detected

# Detects if a hand is near the face by checking hand and nose landmark proximity.
class HandOverFaceDetector:
    def __init__(self, video_path, fps, detection_confidence=0.5):
        self.video_path = video_path
        self.fps = fps or 30.0
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=detection_confidence)
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=detection_confidence
        )
        self.drawer = mp.solutions.drawing_utils

    def detect_hand_over_face_frames(self, show_ui=True, frame_skip=3):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        hand_over_face_frames = []  # Initialize the list
        nose_x, nose_y = None, None  # Initialize nose coordinates

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = self.pose.process(frame_rgb)
            hands_result = self.hands.process(frame_rgb)

            # Only proceed if we have both pose and hand landmarks
            if pose_result.pose_landmarks and hands_result.multi_hand_landmarks:
                # Get nose position first
                nose = pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
                nose_x = int(nose.x * frame.shape[1])
                nose_y = int(nose.y * frame.shape[0])

                # Check each hand's proximity to face
                for hand_landmarks in hands_result.multi_hand_landmarks:
                    # Draw landmarks if show_ui is enabled
                    if show_ui:
                        self.drawer.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    
                    # Check each hand landmark's distance to nose
                    for lm in hand_landmarks.landmark:
                        hand_x = int(lm.x * frame.shape[1])
                        hand_y = int(lm.y * frame.shape[0])
                        
                        # Calculate distance between hand landmark and nose
                        dist = np.hypot(nose_x - hand_x, nose_y - hand_y)
                        
                        # If hand is close to face
                        if dist < 40:  # Distance threshold in pixels
                            hand_over_face_frames.append((frame_count, pose_result.pose_landmarks.landmark))
                            print(f"Hand over face detected at frame {frame_count}")
                            break
                    else:
                        continue
                    break

            # Show UI if enabled
            if show_ui:
                if pose_result.pose_landmarks:
                    self.drawer.draw_landmarks(frame, pose_result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                cv2.imshow("Hand Over Face Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1

        # Cleanup
        cap.release()
        if show_ui:
            cv2.destroyAllWindows()
        self.pose.close()
        self.hands.close()

        return hand_over_face_frames

# Blurs only the face of a person whose pose matches the target skeleton landmarks.
# Used after detecting gesture to blur only that specific individual.
last_face_box = None  # Global cache for the last known face box

def blur_faces_of_person(frame, bbox):
    """
    Blur faces within the specified bounding box region.
    """
    if bbox is None:
        return frame
        
    try:
        x1, y1, x2, y2 = bbox
    except (TypeError, ValueError):
        print(f"Invalid bbox format: {bbox}")
        return frame
        
    h, w, _ = frame.shape
    
    # Ensure coordinates are within frame bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return frame
    
    # Extract person region
    person_crop = frame[y1:y2, x1:x2]
    if person_crop.size == 0:
        return frame

    # Detect face in the person crop
    mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
    face_result = mp_face.process(person_rgb)

    if face_result.detections:
        for detection in face_result.detections:
            box = detection.location_data.relative_bounding_box
            fx = int(box.xmin * (x2 - x1)) + x1
            fy = int(box.ymin * (y2 - y1)) + y1
            fw = int(box.width * (x2 - x1))
            fh = int(box.height * (y2 - y1))

            # Ensure coordinates are within frame bounds
            fx, fy = max(0, fx), max(0, fy)
            fw = min(fw, w - fx)
            fh = min(fh, h - fy)

            if fw > 0 and fh > 0:
                face_roi = frame[fy:fy+fh, fx:fx+fw]
                blurred_face = cv2.GaussianBlur(face_roi, (55, 55), 0)
                frame[fy:fy+fh, fx:fx+fw] = blurred_face

    mp_face.close()
    return frame

def match_person_id(existing_people, new_landmarks, tolerance=0.7):
    """
    Match new_landmarks to existing people. Returns person_id if matched, else new ID.
    existing_people: dict of person_id -> landmarks
    """
    for pid, landmarks in existing_people.items():
        total_diff = sum(np.hypot(landmarks[i].x - new_landmarks[i].x,
                                  landmarks[i].y - new_landmarks[i].y)
                         for i in range(len(landmarks)))
        avg_diff = total_diff / len(landmarks)
        if avg_diff < tolerance:
            return pid
    # New person
    new_id = len(existing_people) + 1
    existing_people[new_id] = new_landmarks
    return new_id

# ──────────────────────────────────────────────────────────────
# detect_and_blur_multiple_people function using global solutions

def detect_and_blur_multiple_people(frame, target_landmarks_list=None, conf_threshold=0.5, frame_count=0):
    """
    Detects multiple people using YOLOv8 detection and blurs their faces using MediaPipe.
    Returns blurred frame.
    Uses globally initialized mp_face_global for efficiency.
    """
    h, w, _ = frame.shape
    people = detect_multiple_people_yolov8(frame, conf_threshold)

    for x1, y1, x2, y2 in people:
        # Extract person region
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        face_result = mp_face_global.process(person_rgb)

        # Apply blurring to the face in every frame
        if face_result.detections:
            for detection in face_result.detections:
                box = detection.location_data.relative_bounding_box
                fx = int(box.xmin * (x2 - x1)) + x1
                fy = int(box.ymin * (y2 - y1)) + y1
                fw = int(box.width * (x2 - x1))
                fh = int(box.height * (y2 - y1))

                fx, fy = max(0, fx), max(0, fy)
                fw, fh = min(fw, w - fx), min(fh, h - fy)

                # Apply blur to the face area in real-time (every frame)
                if fw > 0 and fh > 0:
                    face_roi = frame[fy:fy+fh, fx:fx+fw]
                    blurred_face = cv2.GaussianBlur(face_roi, (55, 55), 0)
                    frame[fy:fy+fh, fx:fx+fw] = blurred_face

    return frame

# ──────────────────────────────────────────────────────────────
# close_global_mediapipe function

def close_global_mediapipe():
    """
    Closes globally initialized MediaPipe solutions to release resources.
    Call this once when application exits.
    """
    mp_face_global.close()
    mp_pose_global.close()

def match_person_to_blur_list(current_bbox, blur_list, tolerance=150):
    """
    Match current_bbox to the closest person in blur_list.
    Returns the matching bbox if found, None otherwise.
    """
    if not blur_list:
        return None
        
    current_center = ((current_bbox[0] + current_bbox[2]) // 2, 
                     (current_bbox[1] + current_bbox[3]) // 2)
    
    closest_match = None
    min_distance = float('inf')
    
    for person_bbox in blur_list:
        person_center = ((person_bbox[0] + person_bbox[2]) // 2,
                        (person_bbox[1] + person_bbox[3]) // 2)
        
        distance = ((current_center[0] - person_center[0])**2 + 
                   (current_center[1] - person_center[1])**2)**0.5
        
        if distance < tolerance and distance < min_distance:
            min_distance = distance
            closest_match = person_bbox
    
    return closest_match  # Returns None if no match found

def adjust_bounding_box_aspect_ratio(x1, y1, x2, y2, frame_shape, target_aspect_ratio=0.6):
    """Adjust bounding box to have a more reasonable aspect ratio for person detection."""
    frame_height, frame_width = frame_shape[:2]
    
    # Get original dimensions
    width = x2 - x1
    height = y2 - y1
    current_aspect_ratio = width / height if height > 0 else 1.0
    
    # If aspect ratio is already reasonable, return as-is
    if 0.4 <= current_aspect_ratio <= 1.0:
        return x1, y1, x2, y2
    
    # Center of the bounding box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Adjust to target aspect ratio
    if current_aspect_ratio > 1.0:  # Too wide
        new_width = int(height * target_aspect_ratio)
        new_x1 = max(0, center_x - new_width // 2)
        new_x2 = min(frame_width, center_x + new_width // 2)
        new_y1, new_y2 = y1, y2
    else:  # Too narrow
        new_height = int(width / target_aspect_ratio)
        new_y1 = max(0, center_y - new_height // 2)
        new_y2 = min(frame_height, center_y + new_height // 2)
        new_x1, new_x2 = x1, x2
    
    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)

def detect_gesture_in_person_box(person_box, frame_source, gesture_type="wave", fps=30, duration_seconds=2):
    """
    Detect gestures within a person's bounding box by analyzing the next N seconds of video.
    Returns True if gesture is detected, False otherwise.
    """
    x1, y1, x2, y2 = person_box
    frames_to_collect = int(fps * duration_seconds)
    
    # Collect frames for the specified duration
    person_frames = []
    current_pos = frame_source.get(cv2.CAP_PROP_POS_FRAMES)
    
    for _ in range(frames_to_collect):
        ret, frame = frame_source.read()
        if not ret:
            break
            
        # Adjust bounding box aspect ratio
        adj_x1, adj_y1, adj_x2, adj_y2 = adjust_bounding_box_aspect_ratio(x1, y1, x2, y2, frame.shape)
        
        # Extract person crop
        person_crop = frame[adj_y1:adj_y2, adj_x1:adj_x2]
        if person_crop.size == 0:
            continue
            
        # Scale up small crops for better MediaPipe processing
        if person_crop.shape[0] < 300 or person_crop.shape[1] < 200:
            scale_factor = max(300 / person_crop.shape[0], 200 / person_crop.shape[1])
            new_height = int(person_crop.shape[0] * scale_factor)
            new_width = int(person_crop.shape[1] * scale_factor)
            person_crop = cv2.resize(person_crop, (new_width, new_height))
            
        person_frames.append(person_crop.copy())
    
    # Reset video position
    frame_source.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    
    if len(person_frames) < 10:
        return False
        
    # Create temporary video for gesture detection
    temp_video_path = "temp_person_crop.avi"
    height, width = person_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    for frame in person_frames:
        out.write(frame)
    out.release()
    
    # Run gesture detection
    gesture_frame_idx = None
    try:
        if gesture_type == "wave":
            detector = WaveDetector(temp_video_path, fps, detection_confidence=0.4)
            detected_frames = detector.detect_wave_timestamps(show_ui=False, frame_skip=3)
        elif gesture_type == "hand_over_face":
            detector = HandOverFaceDetector(temp_video_path, fps, detection_confidence=0.3)
            detected_frames = detector.detect_hand_over_face_frames(show_ui=False, frame_skip=3)
        else:
            detected_frames = []

        if detected_frames:
            gesture_frame_idx = detected_frames[0][0]  # First frame index of detection

    except Exception as e:
        print(f"Error in gesture detection: {e}")
        gesture_frame_idx = None
    
    # Clean up temporary file
    try:
        import os
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
    except:
        pass
    
    return gesture_frame_idx

def blur_faces_of_person(frame, bbox):
    """
    Blur faces within the specified bounding box region.
    """
    if bbox is None:
        return frame
        
    x1, y1, x2, y2 = bbox
    h, w, _ = frame.shape
    
    # Extract person region
    person_crop = frame[y1:y2, x1:x2]
    if person_crop.size == 0:
        return frame

    # Detect face in the person crop
    mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
    face_result = mp_face.process(person_rgb)

    if face_result.detections:
        for detection in face_result.detections:
            box = detection.location_data.relative_bounding_box
            fx = int(box.xmin * (x2 - x1)) + x1
            fy = int(box.ymin * (y2 - y1)) + y1
            fw = int(box.width * (x2 - x1))
            fh = int(box.height * (y2 - y1))

            # Ensure coordinates are within frame bounds
            fx, fy = max(0, fx), max(0, fy)
            fw = min(fw, w - fx)
            fh = min(fh, h - fy)

            if fw > 0 and fh > 0:
                face_roi = frame[fy:fy+fh, fx:fx+fw]
                blurred_face = cv2.GaussianBlur(face_roi, (55, 55), 0)
                frame[fy:fy+fh, fx:fx+fw] = blurred_face

    mp_face.close()
    return frame