# ========================
# utilities.py (clean + synced with main.py)
# ========================

import cv2
import subprocess
import json
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
import os

# ===============================
# GLOBAL MODELS / OBJECTS
# ===============================

# YOLOv8 segmentation model for person detection
yolo_model = YOLO('yolov8m-seg.pt')  # Segmentation model for pixel-level detection

# Global MediaPipe solutions
mp_face_global = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
mp_pose_global = mp.solutions.pose.Pose()

# Global ORB + matcher for tracking
orb = cv2.ORB_create(nfeatures=500)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

DEBUG_TRACKING = True


# ===============================
# ADVANCED PERSON TRACKER (ORB)
# ===============================

class PersonTracker:
    def __init__(self, max_disappeared=15, feature_threshold=0.3, motion_threshold=150):
        self.next_id = 0
        self.tracked_people = {}
        self.max_disappeared = max_disappeared
        self.feature_threshold = feature_threshold
        self.motion_threshold = motion_threshold

        # Reuse global ORB + matcher
        self.orb = orb
        self.matcher = bf_matcher

    def extract_person_features(self, frame, bbox, mask=None):
        """Extract ORB features from person region, using mask if available."""
        x1, y1, x2, y2 = bbox

        # Clamp bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None, None

        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None, None

        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY) if len(person_crop.shape) == 3 else person_crop

        mask_crop = None
        if mask is not None:
            mask_crop = mask[y1:y2, x1:x2]
            if mask_crop.dtype != np.uint8:
                mask_crop = (mask_crop * 255).astype(np.uint8)

        keypoints, descriptors = self.orb.detectAndCompute(gray, mask_crop)
        if descriptors is None or len(keypoints) < 10:
            return None, None

        return keypoints, descriptors

    def calculate_feature_similarity(self, desc1, desc2):
        """ORB descriptor similarity (0..1)."""
        if desc1 is None or desc2 is None:
            return 0.0
        try:
            matches = self.matcher.match(desc1, desc2)
            if len(matches) < 5:
                return 0.0
            matches = sorted(matches, key=lambda x: x.distance)[:20]
            avg_distance = sum(m.distance for m in matches) / len(matches)
            return max(0, 1.0 - (avg_distance / 100.0))
        except Exception:
            return 0.0

    def calculate_motion_consistency(self, bbox1, bbox2):
        """How consistent the movement is between frames (0..1)."""
        c1x, c1y = (bbox1[0] + bbox1[2]) * 0.5, (bbox1[1] + bbox1[3]) * 0.5
        c2x, c2y = (bbox2[0] + bbox2[2]) * 0.5, (bbox2[1] + bbox2[3]) * 0.5
        displacement = ((c2x - c1x) ** 2 + (c2y - c1y) ** 2) ** 0.5
        return max(0, 1.0 - (displacement / self.motion_threshold))

    def update(self, frame, detections_with_masks, frame_number):
        """
        detections_with_masks: list of (bbox, mask)
          - bbox: (x1, y1, x2, y2)
          - mask: np.ndarray or None
        """
        current_frame_people = {}

        # No detections: age out old people
        if not detections_with_masks:
            for person_id in self.tracked_people:
                self.tracked_people[person_id]['disappeared'] += 1
            to_remove = [
                pid for pid, data in self.tracked_people.items()
                if data['disappeared'] > self.max_disappeared
            ]
            for pid in to_remove:
                del self.tracked_people[pid]
            return current_frame_people

        # Extract features for each detection
        detection_features = []
        for bbox, mask in detections_with_masks:
            kp, desc = self.extract_person_features(frame, bbox, mask)
            detection_features.append((bbox, mask, kp, desc))

        matched_pairs = []
        used_detections = set()
        used_people = set()

        # Match most recently seen people first
        sorted_people = sorted(
            self.tracked_people.items(),
            key=lambda kv: kv[1].get('last_frame', 0),
            reverse=True
        )

        for person_id, person_data in sorted_people:
            if person_id in used_people:
                continue

            best_match_idx = None
            best_score = 0.0

            for idx, (det_bbox, det_mask, det_kp, det_desc) in enumerate(detection_features):
                if idx in used_detections:
                    continue

                motion_score = self.calculate_motion_consistency(
                    person_data.get('bbox'), det_bbox
                )
                if motion_score < 0.1:
                    continue

                feature_score = self.calculate_feature_similarity(
                    person_data.get('descriptors'), det_desc
                )
                combined_score = (feature_score * 0.6) + (motion_score * 0.4)

                if combined_score > self.feature_threshold and combined_score > best_score:
                    best_score = combined_score
                    best_match_idx = idx

            if best_match_idx is not None:
                matched_pairs.append((person_id, best_match_idx))
                used_people.add(person_id)
                used_detections.add(best_match_idx)

        # Update matched people
        for person_id, det_idx in matched_pairs:
            det_bbox, det_mask, det_kp, det_desc = detection_features[det_idx]
            person_data = self.tracked_people[person_id]
            person_data.update({
                'bbox': det_bbox,
                'mask': det_mask,
                'keypoints': det_kp,
                'descriptors': det_desc,
                'last_frame': frame_number,
                'disappeared': 0,
            })
            current_frame_people[person_id] = person_data

        # Create new tracks for unmatched detections
        for idx, (det_bbox, det_mask, det_kp, det_desc) in enumerate(detection_features):
            if idx in used_detections:
                continue
            self.tracked_people[self.next_id] = {
                'bbox': det_bbox,
                'mask': det_mask,
                'keypoints': det_kp,
                'descriptors': det_desc,
                'last_frame': frame_number,
                'disappeared': 0,
                'scanned': False,
                'gesture_detected': False,
                'first_seen_frame': frame_number
            }
            current_frame_people[self.next_id] = self.tracked_people[self.next_id]
            self.next_id += 1

        # Age everyone not seen this frame
        for person_id in self.tracked_people:
            if person_id not in current_frame_people:
                self.tracked_people[person_id]['disappeared'] += 1

        to_remove = [
            pid for pid, data in self.tracked_people.items()
            if data['disappeared'] > self.max_disappeared
        ]
        for pid in to_remove:
            del self.tracked_people[pid]

        return current_frame_people

    def mark_person_scanned(self, person_id):
        if person_id in self.tracked_people:
            self.tracked_people[person_id]['scanned'] = True

    def mark_gesture_detected(self, person_id):
        if person_id in self.tracked_people:
            self.tracked_people[person_id]['gesture_detected'] = True

    def get_unscanned_people(self):
        return {
            pid: data for pid, data in self.tracked_people.items()
            if not data.get('scanned', False) and data['disappeared'] == 0
        }

    def get_people_with_gestures(self):
        return {
            pid: data for pid, data in self.tracked_people.items()
            if data.get('gesture_detected', False)
        }

    def get_person_ids_to_blur(self):
        return {
            pid for pid, data in self.tracked_people.items()
            if data.get('gesture_detected', False)
        }


# ===============================
# YOLO PERSON DETECTION
# ===============================

def detect_multiple_people_yolov8(frame, conf_threshold=0.15):
    """
    Detect people using YOLOv8 segmentation.
    Returns list of (bbox, mask) where bbox=(x1,y1,x2,y2).
    """
    orig_h, orig_w = frame.shape[:2]
    if orig_w > 1280:  # resize big frames
        scale = 1280 / orig_w
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        frame_small = cv2.resize(frame, (new_w, new_h))
    else:
        frame_small = frame
        scale = 1.0

    results = yolo_model(frame_small)[0]
    detections = []

    if hasattr(results, "boxes") and results.boxes is not None:
        boxes = results.boxes
        masks = getattr(results, 'masks', None)

        for i, box in enumerate(boxes):
            # class 0 = person
            if int(box.cls) == 0 and float(box.conf) >= conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                if scale != 1.0:
                    x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale

                bbox = (int(x1), int(y1), int(x2), int(y2))

                mask = None
                if masks is not None and i < len(masks.data):
                    mask = masks.data[i].cpu().numpy()
                    # Bring mask back to original size
                    if scale != 1.0:
                        mask = cv2.resize(mask, (orig_w, orig_h))
                    elif mask.shape != (orig_h, orig_w):
                        mask = cv2.resize(mask, (orig_w, orig_h))

                detections.append((bbox, mask))

    return detections


def detect_multiple_people_yolov8_optimized(frame, conf_threshold=0.15):
    """
    Bbox-only version: returns [(x1,y1,x2,y2)].
    Used by older batch helpers.
    """
    orig_h, orig_w = frame.shape[:2]
    if orig_w > 1280:
        scale = 1280 / orig_w
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        frame_small = cv2.resize(frame, (new_w, new_h))
    else:
        frame_small = frame
        scale = 1.0

    results = yolo_model(frame_small)[0]
    detections = []

    if hasattr(results, "boxes") and results.boxes is not None:
        for box in results.boxes:
            if int(box.cls) == 0 and float(box.conf) >= conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                if scale != 1.0:
                    x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
                detections.append((int(x1), int(y1), int(x2), int(y2)))

    return detections


# ===============================
# ROTATION & THUMBNAILS
# ===============================

def get_video_rotation(path):
    """Get rotation metadata via ffprobe, for non-MOV fallback."""
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
    except Exception:
        return 0


def generate_thumbnails(video_path, total_frames, rotation_angle, num_thumbs=10, thumb_size=(80, 45)):
    """Generate thumbnails for the scrub bar."""
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


# ===============================
# GESTURE DETECTORS (WAVE / HOF)
# ===============================

class WaveDetector:
    def __init__(self, video_path, fps, detection_confidence=0.8):
        self.video_path = video_path
        self.fps = fps or 30.0
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.3
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
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

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(frame_rgb)
            pose_results = self.pose.process(frame_rgb)

            if hand_results.multi_hand_landmarks and pose_results.pose_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    if show_ui:
                        self.drawer.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                    xs = [lm.x for lm in hand_landmarks.landmark]
                    media_x = sum(xs) / len(xs)
                    direction = None
                    if last_x is not None:
                        threshold = 0.01
                        if media_x < last_x - threshold:
                            direction = "left"
                        elif media_x > last_x + threshold:
                            direction = "right"
                        if direction and (not movement_history or movement_history[-1][0] != direction):
                            movement_history.append((direction, frame_count, pose_results.pose_landmarks.landmark))

                    movement_history = [(d, f, l) for d, f, l in movement_history if frame_count - f <= self.fps]
                    if len(movement_history) >= 4:
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
        hand_over_face_frames = []

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

            if pose_result.pose_landmarks and hands_result.multi_hand_landmarks:
                nose = pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
                nose_x = int(nose.x * frame.shape[1])
                nose_y = int(nose.y * frame.shape[0])

                for hand_landmarks in hands_result.multi_hand_landmarks:
                    if show_ui:
                        self.drawer.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                    for lm in hand_landmarks.landmark:
                        hand_x = int(lm.x * frame.shape[1])
                        hand_y = int(lm.y * frame.shape[0])

                        dist = np.hypot(nose_x - hand_x, nose_y - hand_y)
                        if dist < 40:
                            hand_over_face_frames.append((frame_count, pose_result.pose_landmarks.landmark))
                            break
                    else:
                        continue
                    break

            if show_ui:
                if pose_result.pose_landmarks:
                    self.drawer.draw_landmarks(frame, pose_result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                cv2.imshow("Hand Over Face Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1

        cap.release()
        if show_ui:
            cv2.destroyAllWindows()
        self.pose.close()
        self.hands.close()

        return hand_over_face_frames


# ===============================
# FACE BBOX HELPER FOR MAIN.PY
# ===============================

def face_bbox_in_person(frame, person_bbox):
    """
    Given a full frame + person bbox, return a tighter face bbox using MediaPipe.
    Used by main.py when you choose "face-only" style blurring.
    """
    if person_bbox is None:
        return None

    try:
        x1, y1, x2, y2 = map(int, person_bbox)
    except Exception:
        return None

    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = mp_face_global.process(rgb)
    if not result.detections:
        return None

    box = result.detections[0].location_data.relative_bounding_box
    fx = int(box.xmin * (x2 - x1)) + x1
    fy = int(box.ymin * (y2 - y1)) + y1
    fw = int(box.width * (x2 - x1))
    fh = int(box.height * (y2 - y1))

    fx, fy = max(0, fx), max(0, fy)
    fw = min(fw, w - fx)
    fh = min(fh, h - fy)
    if fw <= 0 or fh <= 0:
        return None

    return (fx, fy, fx + fw, fy + fh)


# ===============================
# BLUR HELPERS (MATCH MAIN.PY)
# ===============================

def _apply_blur_mode(roi, blur_type="gaussian", strength=50):
    """Internal: apply selected blur mode to a small ROI."""
    strength = int(max(0, min(100, strength)))

    if blur_type == "pixelate":
        # pixel size: 4..40
        min_pix, max_pix = 4, 40
        k = int(min_pix + (max_pix - min_pix) * (strength / 100.0))
        k = max(2, k)
        h, w = roi.shape[:2]
        w_small = max(1, w // k)
        h_small = max(1, h // k)
        small = cv2.resize(roi, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    if blur_type == "solid":
        color = roi.mean(axis=(0, 1)).astype(np.uint8).tolist()
        return np.full_like(roi, color)

    # Default gaussian: kernel size ~11..101
    k = int(11 + (101 - 11) * (strength / 100.0))
    if k % 2 == 0:
        k += 1
    k = max(5, min(k, 151))
    return cv2.GaussianBlur(roi, (k, k), 0)


def blur_faces_of_person(frame, bbox, blur_type="gaussian", strength=50):
    """
    Blur faces within the specified person bbox.
    Signature matches main.py:
       blur_faces_of_person(frame, target_bb, blur_type, strength)
    """
    if bbox is None:
        return frame

    try:
        x1, y1, x2, y2 = map(int, bbox)
    except Exception:
        return frame

    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame

    person_crop = frame[y1:y2, x1:x2]
    if person_crop.size == 0:
        return frame

    person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
    face_result = mp_face_global.process(person_rgb)

    if face_result.detections:
        for detection in face_result.detections:
            box = detection.location_data.relative_bounding_box
            fx = int(box.xmin * (x2 - x1)) + x1
            fy = int(box.ymin * (y2 - y1)) + y1
            fw = int(box.width * (x2 - x1))
            fh = int(box.height * (y2 - y1))

            fx, fy = max(0, fx), max(0, fy)
            fw = min(fw, w - fx)
            fh = min(fh, h - fy)

            if fw > 0 and fh > 0:
                face_roi = frame[fy:fy+fh, fx:fx+fw]
                blurred_face = _apply_blur_mode(face_roi, blur_type=blur_type, strength=strength)
                frame[fy:fy+fh, fx:fx+fw] = blurred_face

    return frame


def detect_and_blur_multiple_people(frame, target_landmarks_list=None, conf_threshold=0.5, frame_count=0):
    """
    Legacy helper: detect and blur all faces of all people in a frame (Gaussian).
    Kept for old batch scripts.
    """
    h, w, _ = frame.shape
    people = detect_multiple_people_yolov8_optimized(frame, conf_threshold)

    for x1, y1, x2, y2 in people:
        frame = blur_faces_of_person(frame, (x1, y1, x2, y2), blur_type="gaussian", strength=60)

    return frame


def close_global_mediapipe():
    """Close globally initialized MediaPipe solutions."""
    mp_face_global.close()
    mp_pose_global.close()


# ===============================
# EXTRA HELPERS / LEGACY
# ===============================

def match_person_to_blur_list(current_bbox, blur_list, tolerance=150):
    """Match current bbox to closest bbox in blur_list."""
    if not blur_list:
        return None

    current_center = (
        (current_bbox[0] + current_bbox[2]) // 2,
        (current_bbox[1] + current_bbox[3]) // 2
    )

    closest_match = None
    min_distance = float('inf')

    for person_data in blur_list:
        if isinstance(person_data, dict) and 'bbox' in person_data:
            person_bbox = person_data['bbox']
        else:
            person_bbox = person_data

        person_center = (
            (person_bbox[0] + person_bbox[2]) // 2,
            (person_bbox[1] + person_bbox[3]) // 2
        )

        distance = ((current_center[0] - person_center[0]) ** 2 +
                    (current_center[1] - person_center[1]) ** 2) ** 0.5

        if distance < tolerance and distance < min_distance:
            min_distance = distance
            closest_match = person_data

    return closest_match


def adjust_bounding_box_aspect_ratio(x1, y1, x2, y2, frame_shape, target_aspect_ratio=0.6):
    """Adjust bbox to have reasonable aspect ratio around person."""
    frame_height, frame_width = frame_shape[:2]

    width = x2 - x1
    height = y2 - y1
    current_aspect_ratio = width / height if height > 0 else 1.0

    if 0.4 <= current_aspect_ratio <= 1.0:
        return x1, y1, x2, y2

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    if current_aspect_ratio > 1.0:  # too wide
        new_width = int(height * target_aspect_ratio)
        new_x1 = max(0, center_x - new_width // 2)
        new_x2 = min(frame_width, center_x + new_width // 2)
        new_y1, new_y2 = y1, y2
    else:  # too tall
        new_height = int(width / target_aspect_ratio)
        new_y1 = max(0, center_y - new_height // 2)
        new_y2 = min(frame_height, center_y + new_height // 2)
        new_x1, new_x2 = x1, x2

    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)


def detect_gesture_in_person_box(person_box, frame_source, gesture_type="wave", fps=30, duration_seconds=2):
    """OPTIMIZED: Detect gestures within a person's bounding box."""
    # === this is your original implementation, preserved ===

    # Handle different bbox formats
    if isinstance(person_box, (list, tuple)):
        if len(person_box) == 2:
            # Check if this is (bbox, mask) format from segmentation model
            first_element = person_box[0]
            second_element = person_box[1]
            
            # If first element is a tuple/list of 4 numbers and second is array/large object
            if (isinstance(first_element, (tuple, list)) and len(first_element) == 4 and
                hasattr(second_element, 'shape')):  # second element is likely a numpy array (mask)
                
                try:
                    x1, y1, x2, y2 = first_element
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(f"DEBUG: Extracted bbox from (bbox, mask) format: ({x1}, {y1}, {x2}, {y2})")
                except (ValueError, TypeError) as e:
                    print(f"ERROR: Cannot extract bbox from (bbox, mask) format: {person_box[0]}, error: {e}")
                    return False
            
            # Check if elements are tuples/lists (nested format: [(x1, y1), (x2, y2)])
            elif isinstance(first_element, (tuple, list)) and isinstance(second_element, (tuple, list)):
                try:
                    if len(first_element) >= 2 and len(second_element) >= 2:
                        x1, y1 = int(first_element[0]), int(first_element[1])
                        x2, y2 = int(second_element[0]), int(second_element[1])
                        print(f"DEBUG: Converted nested bbox to ({x1}, {y1}, {x2}, {y2})")
                    else:
                        print(f"ERROR: Invalid nested bbox format: {person_box}")
                        return False
                except (ValueError, TypeError) as e:
                    print(f"ERROR: Cannot process nested bbox: {person_box}, error: {e}")
                    return False
            
            # Simple (center_x, center_y) format
            else:
                try:
                    cx, cy = int(first_element), int(second_element)
                    # Create a default 200x300 bbox around the center
                    x1, y1 = max(0, cx - 100), max(0, cy - 150)
                    x2, y2 = cx + 100, cy + 150
                    print(f"DEBUG: Expanded 2-value bbox {person_box} to ({x1}, {y1}, {x2}, {y2})")
                except (ValueError, TypeError) as e:
                    print(f"ERROR: Cannot process 2-value bbox: {person_box}, error: {e}")
                    return False
                    
        elif len(person_box) == 4:
            # Standard format: (x1, y1, x2, y2)
            try:
                x1, y1, x2, y2 = person_box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(f"DEBUG: Using standard 4-value bbox: ({x1}, {y1}, {x2}, {y2})")
            except (ValueError, TypeError) as e:
                print(f"ERROR: Cannot convert 4-value bbox to integers: {person_box}, error: {e}")
                return False
        else:
            print(f"ERROR: Invalid bbox length: {len(person_box)}, bbox: {person_box}")
            return False
    else:
        print(f"ERROR: Invalid bbox type: {type(person_box)}, value: {person_box}")
        return False

    frames_to_collect = int(fps * duration_seconds)
    
    # Get frame dimensions to ensure bbox is within bounds
    test_ret, test_frame = frame_source.read()
    if not test_ret:
        return False
    
    frame_h, frame_w = test_frame.shape[:2]
    frame_source.set(cv2.CAP_PROP_POS_FRAMES, frame_source.get(cv2.CAP_PROP_POS_FRAMES) - 1)  # Go back one frame
    
    # Ensure bbox is within frame bounds
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(x1 + 1, min(x2, frame_w))
    y2 = max(y1 + 1, min(y2, frame_h))
    
    person_frames = []
    current_pos = frame_source.get(cv2.CAP_PROP_POS_FRAMES)
    
    for _ in range(frames_to_collect):
        ret, frame = frame_source.read()
        if not ret:
            break
            
        # --- Expand ROI to make gesture detection less aggressive ---
        expand_ratio = 0.15  # expand 15% outward in all directions
        w, h = x2 - x1, y2 - y1
        expand_x = int(w * expand_ratio)
        expand_y = int(h * expand_ratio)

        ex1 = max(0, x1 - expand_x)
        ey1 = max(0, y1 - expand_y)
        ex2 = min(frame.shape[1], x2 + expand_x)
        ey2 = min(frame.shape[0], y2 + expand_y)

        adj_x1, adj_y1, adj_x2, adj_y2 = adjust_bounding_box_aspect_ratio(ex1, ey1, ex2, ey2, frame.shape)
        person_crop = frame[adj_y1:adj_y2, adj_x1:adj_x2]

        
        if person_crop.size == 0:
            continue
            
        # OPTIMIZATION: Scale up small crops for better processing
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
    
    # Run gesture detection with higher confidence
    gesture_detected = False
    try:
        if gesture_type == "wave":
            detector = WaveDetector(temp_video_path, fps, detection_confidence=0.7)
            detected_frames = detector.detect_wave_timestamps(show_ui=False, frame_skip=3)
        elif gesture_type == "hand_over_face":
            detector = HandOverFaceDetector(temp_video_path, fps, detection_confidence=0.6)
            detected_frames = detector.detect_hand_over_face_frames(show_ui=False, frame_skip=3)
        else:
            detected_frames = []

        gesture_detected = len(detected_frames) > 0

    except Exception as e:
        gesture_detected = False
    
    # Clean up
    try:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
    except:
        pass
    
    return gesture_detected


# Legacy support functions (kept in case other scripts still call them)

def match_person_id(existing_people, new_landmarks, tolerance=0.7):
    """Legacy function for backward compatibility."""
    for pid, landmarks in existing_people.items():
        total_diff = sum(np.hypot(landmarks[i].x - new_landmarks[i].x,
                                  landmarks[i].y - new_landmarks[i].y)
                         for i in range(len(landmarks)))
        avg_diff = total_diff / len(landmarks)
        if avg_diff < tolerance:
            return pid
    new_id = len(existing_people) + 1
    existing_people[new_id] = new_landmarks
    return new_id


def compute_orb_features(frame, bbox):
    """Legacy ORB feature extraction."""
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, descriptors = orb.detectAndCompute(gray, None)
    return descriptors


def match_orb_features(desc1, desc2, match_threshold=15):
    """Legacy ORB matching."""
    if desc1 is None or desc2 is None:
        return False
    matches = bf_matcher.match(desc1, desc2)
    good_matches = [m for m in matches if m.distance < 60]
    return len(good_matches) >= match_threshold
