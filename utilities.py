# ========================
# MERGED utilities.py
# ========================

# --------- FILE 1 CONTENT (Optimized) ---------

import cv2
import subprocess
import json
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
import os

# OPTIMIZATION: Use segmentation model for better person boundaries (Kyle's improvement)
# but with caching to avoid reloading
yolo_model = YOLO('yolov8m-seg.pt')  # Segmentation model for pixel-level detection

# OPTIMIZATION: Initialize MediaPipe solutions globally for reuse
mp_face_global = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
mp_pose_global = mp.solutions.pose.Pose()

# OPTIMIZATION: Global feature extractors to avoid recreation
orb = cv2.ORB_create(nfeatures=500)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Debug and tracking globals
DEBUG_TRACKING = True
next_person_id = 1

# ──────────────────────────────────────────────────────────────
# ADVANCED PERSON TRACKING SYSTEM (From Kyle) - OPTIMIZED ok
class PersonTracker:
    def __init__(self, max_disappeared=15, feature_threshold=0.3, motion_threshold=150):
        self.next_id = 0
        self.tracked_people = {}
        self.max_disappeared = max_disappeared
        self.feature_threshold = feature_threshold
        self.motion_threshold = motion_threshold
        
        # OPTIMIZATION: Reuse global ORB and matcher
        self.orb = orb
        self.matcher = bf_matcher
        
    def extract_person_features(self, frame, bbox, mask=None):
        """OPTIMIZED: Extract ORB features from person region, using mask if available."""
        x1, y1, x2, y2 = bbox
        
        # SPEED: Early bounds checking
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None, None
            
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None, None
            
        # OPTIMIZATION: Convert to grayscale only once
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY) if len(person_crop.shape) == 3 else person_crop
        
        # Use mask if available (from segmentation)
        mask_crop = None
        if mask is not None:
            mask_crop = mask[y1:y2, x1:x2]
            if mask_crop.dtype != np.uint8:
                mask_crop = (mask_crop * 255).astype(np.uint8)
        
        # OPTIMIZATION: Use global ORB extractor
        keypoints, descriptors = self.orb.detectAndCompute(gray, mask_crop)
        
        # SPEED: Quick validation
        if descriptors is None or len(keypoints) < 10:
            return None, None
            
        return keypoints, descriptors
    
    def calculate_feature_similarity(self, desc1, desc2):
        """OPTIMIZED: Calculate similarity between descriptors with early exits."""
        if desc1 is None or desc2 is None:
            return 0.0
            
        try:
            # OPTIMIZATION: Use global matcher
            matches = self.matcher.match(desc1, desc2)
            if len(matches) < 5:  # SPEED: Early exit
                return 0.0
                
            # OPTIMIZATION: Only sort top matches
            matches = sorted(matches, key=lambda x: x.distance)[:20]
            avg_distance = sum(m.distance for m in matches) / len(matches)
            
            # SPEED: Fast similarity calculation
            return max(0, 1.0 - (avg_distance / 100.0))
            
        except:
            return 0.0
    
    def calculate_motion_consistency(self, bbox1, bbox2):
        """SPEED: Fast motion consistency check."""
        # OPTIMIZATION: Calculate centers without intermediate variables
        c1x, c1y = (bbox1[0] + bbox1[2]) * 0.5, (bbox1[1] + bbox1[3]) * 0.5
        c2x, c2y = (bbox2[0] + bbox2[2]) * 0.5, (bbox2[1] + bbox2[3]) * 0.5
        
        # SPEED: Fast distance calculation
        displacement = ((c2x - c1x)**2 + (c2y - c1y)**2)**0.5
        return max(0, 1.0 - (displacement / self.motion_threshold))
    
    def update(self, frame, detections_with_masks, frame_number):
        """OPTIMIZED: Update tracker with feature matching - prioritizing speed."""
        current_frame_people = {}
        
        # SPEED: Early exit if no detections
        if not detections_with_masks:
            # Update disappeared counts
            for person_id in self.tracked_people:
                self.tracked_people[person_id]['disappeared'] += 1
            
            # Remove old people
            to_remove = [pid for pid, data in self.tracked_people.items() 
                        if data['disappeared'] > self.max_disappeared]
            for pid in to_remove:
                del self.tracked_people[pid]
            
            return current_frame_people
        
        # OPTIMIZATION: Extract features for all detections in batch
        detection_features = []
        for bbox, mask in detections_with_masks:
            keypoints, descriptors = self.extract_person_features(frame, bbox, mask)
            detection_features.append((bbox, mask, keypoints, descriptors))
        
        # SPEED: Greedy matching algorithm (faster than Hungarian)
        matched_pairs = []
        used_detections = set()
        used_people = set()
        
        # Sort existing people by last seen (prioritize recent ones)
        sorted_people = sorted(self.tracked_people.items(), 
                              key=lambda x: x[1].get('last_frame', 0), reverse=True)
        
        for person_id, person_data in sorted_people:
            if person_id in used_people:
                continue
                
            best_match_idx = None
            best_score = 0
            
            for idx, (det_bbox, det_mask, det_kp, det_desc) in enumerate(detection_features):
                if idx in used_detections:
                    continue
                
                # SPEED: Quick distance check first
                motion_score = self.calculate_motion_consistency(
                    person_data.get('bbox'), det_bbox
                )
                
                if motion_score < 0.1:  # SPEED: Skip if motion is inconsistent
                    continue
                
                # Then feature similarity
                feature_score = self.calculate_feature_similarity(
                    person_data.get('descriptors'), det_desc
                )
                
                # OPTIMIZATION: Weighted scoring favoring motion (faster)
                combined_score = (feature_score * 0.6) + (motion_score * 0.4)
                
                if combined_score > self.feature_threshold and combined_score > best_score:
                    best_score = combined_score
                    best_match_idx = idx
            
            if best_match_idx is not None:
                matched_pairs.append((person_id, best_match_idx))
                used_people.add(person_id)
                used_detections.add(best_match_idx)
        
        # SPEED: Batch update matched people
        for person_id, det_idx in matched_pairs:
            det_bbox, det_mask, det_kp, det_desc = detection_features[det_idx]
            
            # OPTIMIZATION: Update in place
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
        
        # Create new people for unmatched detections
        for idx, (det_bbox, det_mask, det_kp, det_desc) in enumerate(detection_features):
            if idx not in used_detections:
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
        
        # SPEED: Batch update disappeared counts
        for person_id in self.tracked_people:
            if person_id not in current_frame_people:
                self.tracked_people[person_id]['disappeared'] += 1
        
        # SPEED: Batch remove old people
        to_remove = [pid for pid, data in self.tracked_people.items() 
                    if data['disappeared'] > self.max_disappeared]
        for pid in to_remove:
            del self.tracked_people[pid]
        
        return current_frame_people
    
    def mark_person_scanned(self, person_id):
        """Mark a person as scanned to avoid rescanning."""
        if person_id in self.tracked_people:
            self.tracked_people[person_id]['scanned'] = True
    
    def mark_gesture_detected(self, person_id):
        """Mark that a gesture was detected for this person."""
        if person_id in self.tracked_people:
            self.tracked_people[person_id]['gesture_detected'] = True
    
    def get_unscanned_people(self):
        """Get list of people who haven't been scanned yet."""
        return {pid: data for pid, data in self.tracked_people.items() 
                if not data.get('scanned', False) and data['disappeared'] == 0}
    
    def get_people_with_gestures(self):
        """Get list of people who have detected gestures."""
        return {pid: data for pid, data in self.tracked_people.items() 
                if data.get('gesture_detected', False)}
    
    def get_person_ids_to_blur(self):
        """Get set of person IDs that should be blurred."""
        return {pid for pid, data in self.tracked_people.items() 
                if data.get('gesture_detected', False)}

# ──────────────────────────────────────────────────────────────

def detect_multiple_people_yolov8(frame, conf_threshold=0.15):
    """
    OPTIMIZED: Detects multiple people using YOLOv8 segmentation model with speed optimizations.
    Returns a list of (bbox, mask) tuples for better person tracking.
    """
    # OPTIMIZATION: Resize frame for faster detection (from original utilities.py)
    orig_h, orig_w = frame.shape[:2]
    if orig_w > 1280:  # Only resize if larger than 720p
        scale = 1280 / orig_w
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        frame_small = cv2.resize(frame, (new_w, new_h))
    else:
        frame_small = frame
        scale = 1.0
    
    # SPEED: Single model inference
    results = yolo_model(frame_small)[0]
    detections = []
    
    if hasattr(results, "boxes") and results.boxes is not None:
        boxes = results.boxes
        masks = getattr(results, 'masks', None)
        
        for i, box in enumerate(boxes):
            # Check if detection is a person (class 0 in COCO dataset)
            if int(box.cls) == 0 and float(box.conf) >= conf_threshold:
                # Get bounding box coordinates and scale back
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                if scale != 1.0:
                    x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                
                bbox = (int(x1), int(y1), int(x2), int(y2))
                
                # Get corresponding mask if available
                mask = None
                if masks is not None and i < len(masks.data):
                    mask = masks.data[i].cpu().numpy()
                    # OPTIMIZATION: Only resize mask if frame was resized
                    if scale != 1.0:
                        mask = cv2.resize(mask, (orig_w, orig_h))
                    elif mask.shape != (orig_h, orig_w):
                        mask = cv2.resize(mask, (orig_w, orig_h))
                
                detections.append((bbox, mask))

    return detections

# OPTIMIZATION: Keep the original optimized function as fallback
def detect_multiple_people_yolov8_optimized(frame, conf_threshold=0.15):
    """SPEED OPTIMIZED: Fastest person detection with frame preprocessing."""
    
    # Resize frame for faster detection (maintain aspect ratio)
    orig_h, orig_w = frame.shape[:2]
    if orig_w > 1280:  # Only resize if larger than 720p
        scale = 1280 / orig_w
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        frame_small = cv2.resize(frame, (new_w, new_h))
    else:
        frame_small = frame
        scale = 1.0
    
    # Run YOLO on smaller frame
    results = yolo_model(frame_small)[0]
    detections = []
    
    if hasattr(results, "boxes") and results.boxes is not None:
        boxes = results.boxes
        
        for box in boxes:
            if int(box.cls) == 0 and float(box.conf) >= conf_threshold:
                # Scale coordinates back to original size
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                if scale != 1.0:
                    x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                
                detections.append((int(x1), int(y1), int(x2), int(y2)))
    
    return detections

def get_video_rotation(path):
    """Get video rotation metadata."""
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

def generate_thumbnails(video_path, total_frames, rotation_angle, num_thumbs=10, thumb_size=(80, 45)):
    """Generate video thumbnails for preview."""
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

# Wave gesture detector
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
                        threshold = 0.005
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

# Hand over face detector
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

# OPTIMIZED: Face blurring function
def blur_faces_of_person(frame, bbox):
    """OPTIMIZED: Blur faces within the specified bounding box region."""
    if bbox is None:
        return frame
        
    try:
        x1, y1, x2, y2 = bbox
    except (TypeError, ValueError):
        return frame
        
    h, w, _ = frame.shape
    
    # SPEED: Ensure coordinates are within frame bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return frame
    
    # Extract person region
    person_crop = frame[y1:y2, x1:x2]
    if person_crop.size == 0:
        return frame

    # OPTIMIZATION: Use global MediaPipe face detector
    person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
    face_result = mp_face_global.process(person_rgb)

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
                # SPEED: Use smaller blur kernel for faster processing
                blurred_face = cv2.GaussianBlur(face_roi, (51, 51), 0)
                frame[fy:fy+fh, fx:fx+fw] = blurred_face

    return frame

def detect_and_blur_multiple_people(frame, target_landmarks_list=None, conf_threshold=0.5, frame_count=0):
    """OPTIMIZED: Detect and blur multiple people using global solutions."""
    h, w, _ = frame.shape
    
    # SPEED: Use optimized detection
    people = detect_multiple_people_yolov8_optimized(frame, conf_threshold)

    for x1, y1, x2, y2 in people:
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

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
                fw, fh = min(fw, w - fx), min(fh, h - fy)

                if fw > 0 and fh > 0:
                    face_roi = frame[fy:fy+fh, fx:fx+fw]
                    blurred_face = cv2.GaussianBlur(face_roi, (51, 51), 0)
                    frame[fy:fy+fh, fx:fx+fw] = blurred_face

    return frame

def close_global_mediapipe():
    """Close globally initialized MediaPipe solutions."""
    mp_face_global.close()
    mp_pose_global.close()

def match_person_to_blur_list(current_bbox, blur_list, tolerance=150):
    """Match current_bbox to the closest person in blur_list."""
    if not blur_list:
        return None
        
    current_center = ((current_bbox[0] + current_bbox[2]) // 2, 
                     (current_bbox[1] + current_bbox[3]) // 2)
    
    closest_match = None
    min_distance = float('inf')
    
    for person_data in blur_list:
        if isinstance(person_data, dict) and 'bbox' in person_data:
            person_bbox = person_data['bbox']
        else:
            person_bbox = person_data
            
        person_center = ((person_bbox[0] + person_bbox[2]) // 2,
                        (person_bbox[1] + person_bbox[3]) // 2)
        
        distance = ((current_center[0] - person_center[0])**2 + 
                   (current_center[1] - person_center[1])**2)**0.5
        
        if distance < tolerance and distance < min_distance:
            min_distance = distance
            closest_match = person_data
    
    return closest_match

def adjust_bounding_box_aspect_ratio(x1, y1, x2, y2, frame_shape, target_aspect_ratio=0.6):
    """Adjust bounding box to have reasonable aspect ratio."""
    frame_height, frame_width = frame_shape[:2]
    
    width = x2 - x1
    height = y2 - y1
    current_aspect_ratio = width / height if height > 0 else 1.0
    
    if 0.4 <= current_aspect_ratio <= 1.0:
        return x1, y1, x2, y2
    
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
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

# ADD: Global gesture cache to avoid re-detecting same person
_gesture_cache = {}  # (person_id, gesture_type) -> (result, frame_count)
_gesture_cache_ttl = 150  # frames before expiry

def detect_gesture_in_person_box(person_box, frame_source, gesture_type="wave", fps=30, duration_seconds=2):
    """OPTIMIZED: Fast gesture detection with minimal temp video overhead."""
    
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
    current_frame_pos = frame_source.get(cv2.CAP_PROP_POS_FRAMES) - 1
    frame_source.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
    
    # Cache check
    cache_key = (id(frame_source), gesture_type)
    if cache_key in _gesture_cache:
        cached_result, cached_frame = _gesture_cache[cache_key]
        frames_since = abs(current_frame_pos - cached_frame)
        if frames_since < _gesture_cache_ttl:
            return cached_result
    
    # Bounds checking
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(x1 + 1, min(x2, frame_w))
    y2 = max(y1 + 1, min(y2, frame_h))
    
    # Pre-filter: Sample first frame's clarity
    temp_ret, temp_frame = frame_source.read()
    if temp_ret:
        roi = temp_frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = gray.mean()
        
        if sharpness < 10 or brightness < 30:
            _gesture_cache[cache_key] = (False, current_frame_pos)
            return False
    
    frame_source.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
    
    person_frames = []
    
    for _ in range(frames_to_collect):
        ret, frame = frame_source.read()
        if not ret:
            break
        
        adj_x1, adj_y1, adj_x2, adj_y2 = adjust_bounding_box_aspect_ratio(x1, y1, x2, y2, frame.shape)
        person_crop = frame[adj_y1:adj_y2, adj_x1:adj_x2]
        
        if person_crop.size == 0:
            continue
        
        # Scale up small crops
        if person_crop.shape[0] < 300 or person_crop.shape[1] < 200:
            scale_factor = max(300 / person_crop.shape[0], 200 / person_crop.shape[1])
            new_height = int(person_crop.shape[0] * scale_factor)
            new_width = int(person_crop.shape[1] * scale_factor)
            person_crop = cv2.resize(person_crop, (new_width, new_height))
        
        person_frames.append(person_crop.copy())
    
    # Reset video position
    frame_source.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
    
    if len(person_frames) < 10:
        _gesture_cache[cache_key] = (False, current_frame_pos)
        return False
    
    # ═══════════════════════════════════════════════════════════
    # OPTIMIZATION: Minimal temp video with fresh detectors per person
    # ═══════════════════════════════════════════════════════════
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
        temp_video_path = tmp.name
    
    try:
        # Write cropped frames to temp video (FAST: no compression, low quality)
        height, width = person_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        for frame in person_frames:
            out.write(frame)
        out.release()
        
        # Create FRESH detectors for THIS person only
        if gesture_type == "wave":
            hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
            pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7)
            detector = WaveDetector(temp_video_path, fps, detection_confidence=0.7)
        elif gesture_type == "hand_over_face":
            hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
            pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.6)
            detector = HandOverFaceDetector(temp_video_path, fps, detection_confidence=0.6)
        else:
            return False
        
        # Run detection
        gesture_detected = False
        try:
            if gesture_type == "wave":
                detected_frames = detector.detect_wave_timestamps(show_ui=False, frame_skip=3)
            elif gesture_type == "hand_over_face":
                detected_frames = detector.detect_hand_over_face_frames(show_ui=False, frame_skip=3)
            else:
                detected_frames = []
            
            gesture_detected = len(detected_frames) > 0
            
        except Exception as e:
            gesture_detected = False
        finally:
            hands.close()
            pose.close()
        
        _gesture_cache[cache_key] = (gesture_detected, current_frame_pos)
        return gesture_detected
        
    finally:
        # Clean up temp file
        try:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        except:
            pass

def _analyze_wave_gesture(hand_detections, total_frames):
    """Analyze hand movement pattern for wave gesture."""
    if len(hand_detections) < 4:
        return False
    
    # Check for consistent left-right movement
    x_positions = []
    for frame_idx, landmarks in hand_detections:
        xs = [lm.x for hand in landmarks for lm in hand.landmark]
        if xs:
            x_positions.append(sum(xs) / len(xs))
    
    if len(x_positions) < 4:
        return False
    
    # Look for oscillation pattern (left → right → left or vice versa)
    direction_changes = 0
    for i in range(1, len(x_positions)):
        if (x_positions[i] - x_positions[i-1]) * (x_positions[i-1] - (x_positions[i-2] if i > 1 else 0)) < 0:
            direction_changes += 1
    
    # Wave typically has 2+ direction changes in a short sequence
    return direction_changes >= 2


def _analyze_hand_over_face(hand_detections, pose_landmarks_list, person_frames):
    """Analyze if hand is consistently near face."""
    if not hand_detections or not pose_landmarks_list:
        return False
    
    hand_near_face_count = 0
    
    for h_idx, (h_frame, hand_landmarks) in enumerate(hand_detections):
        # Find corresponding pose frame
        pose_frame = next((p_frame for p_frame, _ in pose_landmarks_list if p_frame == h_frame), None)
        if pose_frame is None:
            continue
        
        pose_lm = next(lm for f, lm in pose_landmarks_list if f == pose_frame)
        
        # Get nose position
        nose = pose_lm.landmark[mp.solutions.pose.PoseLandmark.NOSE]
        frame_h, frame_w = person_frames[h_frame].shape[:2]
        nose_x = int(nose.x * frame_w)
        nose_y = int(nose.y * frame_h)
        
        # Check hand proximity to nose
        for hand in hand_landmarks:
            for lm in hand.landmark:
                hand_x = int(lm.x * frame_w)
                hand_y = int(lm.y * frame_h)
                dist = np.hypot(nose_x - hand_x, nose_y - hand_y)
                
                if dist < 60:  # Within 60 pixels of nose
                    hand_near_face_count += 1
                    break
    
    # Hand over face detected if present in >30% of frames
    return hand_near_face_count > len(hand_detections) * 0.3

# Legacy support functions
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

# --------- FILE 2 CONTENT (Original) ---------

import cv2
import subprocess
import json
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
import os

# Initialize YOLOv8 segmentation model globally for pixel-level person detection
yolo_model = YOLO('yolov8m-seg.pt')  # Use segmentation model for better person boundaries

# ──────────────────────────────────────────────────────────────
# ADDITION: Initialize MediaPipe solutions globally for reuse
mp_face_global = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
mp_pose_global = mp.solutions.pose.Pose()

# ──────────────────────────────────────────────────────────────
# ADVANCED PERSON TRACKING SYSTEM WITH ORB FEATURES
class PersonTracker:
    def __init__(self, max_disappeared=15, feature_threshold=0.3, motion_threshold=150):
        self.next_id = 0
        self.tracked_people = {}  # id -> PersonTrackingData
        self.max_disappeared = max_disappeared
        self.feature_threshold = feature_threshold
        self.motion_threshold = motion_threshold
        
        # Initialize feature extractors
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def extract_person_features(self, frame, bbox, mask=None):
        """Extract ORB features from person region, using mask if available."""
        x1, y1, x2, y2 = bbox
        # Ensure bbox is within frame bounds
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
        if desc1 is None or desc2 is None:
            return 0.0
        try:
            matches = self.matcher.match(desc1, desc2)
            if len(matches) < 5:
                return 0.0
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:min(20, len(matches))]
            avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
            return max(0, 1.0 - (avg_distance / 100.0))
        except:
            return 0.0
    
    def calculate_motion_consistency(self, bbox1, bbox2):
        c1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        c2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        displacement = np.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)
        return max(0, 1.0 - (displacement / self.motion_threshold))
    
    def update(self, frame, detections_with_masks, frame_number):
        current_frame_people = {}
        detection_features = []
        for bbox, mask in detections_with_masks:
            keypoints, descriptors = self.extract_person_features(frame, bbox, mask)
            detection_features.append((bbox, mask, keypoints, descriptors))
        matched_pairs = []
        used_detections = set()
        used_people = set()
        for person_id, person_data in self.tracked_people.items():
            if person_id in used_people:
                continue
            best_match_idx = None
            best_score = 0
            for idx, (det_bbox, det_mask, det_kp, det_desc) in enumerate(detection_features):
                if idx in used_detections:
                    continue
                feature_score = self.calculate_feature_similarity(person_data.get('descriptors'), det_desc)
                motion_score = self.calculate_motion_consistency(person_data.get('bbox'), det_bbox)
                combined_score = (feature_score * 0.7) + (motion_score * 0.3)
                if combined_score > self.feature_threshold and combined_score > best_score:
                    best_score = combined_score
                    best_match_idx = idx
            if best_match_idx is not None:
                matched_pairs.append((person_id, best_match_idx))
                used_people.add(person_id)
                used_detections.add(best_match_idx)
        for person_id, det_idx in matched_pairs:
            det_bbox, det_mask, det_kp, det_desc = detection_features[det_idx]
            self.tracked_people[person_id].update({
                'bbox': det_bbox,
                'mask': det_mask,
                'keypoints': det_kp,
                'descriptors': det_desc,
                'last_frame': frame_number,
                'disappeared': 0,
            })
            current_frame_people[person_id] = self.tracked_people[person_id]
        for idx, (det_bbox, det_mask, det_kp, det_desc) in enumerate(detection_features):
            if idx not in used_detections:
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
        for person_id in self.tracked_people:
            if person_id not in current_frame_people:
                self.tracked_people[person_id]['disappeared'] += 1
        to_remove = [pid for pid, data in self.tracked_people.items() if data['disappeared'] > self.max_disappeared]
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
        return {pid: data for pid, data in self.tracked_people.items() if not data.get('scanned', False) and data['disappeared'] == 0}
    
    def get_people_with_gestures(self):
        return {pid: data for pid, data in self.tracked_people.items() if data.get('gesture_detected', False)}
    
    def get_person_ids_to_blur(self):
        return {pid for pid, data in self.tracked_people.items() if data.get('gesture_detected', False)}
