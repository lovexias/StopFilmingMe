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
            
        # Convert to grayscale for feature extraction
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY) if len(person_crop.shape) == 3 else person_crop
        
        # Use mask if available (from segmentation)
        mask_crop = None
        if mask is not None:
            mask_crop = mask[y1:y2, x1:x2]
            # Convert mask to uint8 if needed
            if mask_crop.dtype != np.uint8:
                mask_crop = (mask_crop * 255).astype(np.uint8)
        
        # Extract features
        keypoints, descriptors = self.orb.detectAndCompute(gray, mask_crop)
        
        if descriptors is None or len(keypoints) < 10:
            return None, None
            
        return keypoints, descriptors
    
    def calculate_feature_similarity(self, desc1, desc2):
        """Calculate similarity between two sets of descriptors."""
        if desc1 is None or desc2 is None:
            return 0.0
            
        try:
            matches = self.matcher.match(desc1, desc2)
            if len(matches) < 5:  # Need minimum matches
                return 0.0
                
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Use top matches for scoring
            good_matches = matches[:min(20, len(matches))]
            avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
            
            # Convert distance to similarity (lower distance = higher similarity)
            similarity = max(0, 1.0 - (avg_distance / 100.0))
            return similarity
            
        except Exception as e:
            return 0.0
    
    def calculate_motion_consistency(self, bbox1, bbox2):
        """Calculate motion consistency between two bounding boxes."""
        # Calculate centers
        c1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        c2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        # Calculate displacement
        displacement = np.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)
        
        # Normalize by expected max movement
        consistency = max(0, 1.0 - (displacement / self.motion_threshold))
        return consistency
    
    def update(self, frame, detections_with_masks, frame_number):
        """
        Update tracker with new detections using feature matching.
        detections_with_masks: list of (bbox, mask) tuples
        """
        current_frame_people = {}
        
        # Extract features for all new detections
        detection_features = []
        for bbox, mask in detections_with_masks:
            keypoints, descriptors = self.extract_person_features(frame, bbox, mask)
            detection_features.append((bbox, mask, keypoints, descriptors))
        
        # Match detections to existing tracked people
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
                
                # Calculate feature similarity
                feature_score = self.calculate_feature_similarity(
                    person_data.get('descriptors'), det_desc
                )
                
                # Calculate motion consistency
                motion_score = self.calculate_motion_consistency(
                    person_data.get('bbox'), det_bbox
                )
                
                # Combined score (weighted)
                combined_score = (feature_score * 0.7) + (motion_score * 0.3)
                
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
            
            self.tracked_people[person_id].update({
                'bbox': det_bbox,
                'mask': det_mask,
                'keypoints': det_kp,
                'descriptors': det_desc,
                'last_frame': frame_number,
                'disappeared': 0,
            })
            
            current_frame_people[person_id] = self.tracked_people[person_id]
        
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
                print(f"New person detected with ID: {self.next_id}")
                self.next_id += 1
        
        # Update disappeared count for unmatched people
        for person_id in self.tracked_people:
            if person_id not in current_frame_people:
                self.tracked_people[person_id]['disappeared'] += 1
        
        # Remove people who have been gone too long
        to_remove = [pid for pid, data in self.tracked_people.items() 
                    if data['disappeared'] > self.max_disappeared]
        for pid in to_remove:
            print(f"Person ID {pid} lost (disappeared for {self.tracked_people[pid]['disappeared']} frames)")
            del self.tracked_people[pid]
        
        return current_frame_people
    
    def mark_person_scanned(self, person_id):
        """Mark a person as scanned to avoid rescanning."""
        if person_id in self.tracked_people:
            self.tracked_people[person_id]['scanned'] = True
            print(f"Person ID {person_id} marked as scanned")
    
    def mark_gesture_detected(self, person_id):
        """Mark that a gesture was detected for this person."""
        if person_id in self.tracked_people:
            self.tracked_people[person_id]['gesture_detected'] = True
            print(f"Gesture detected for Person ID {person_id} - will be blurred throughout video")
    
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

# Global tracker instance
person_tracker = PersonTracker()

# ──────────────────────────────────────────────────────────────

def detect_multiple_people_yolov8(frame, conf_threshold=0.15):
    """
    Detects multiple people in a frame using YOLOv8 segmentation model.
    Returns a list of (bbox, mask) tuples for better person tracking.
    """
    results = yolo_model(frame)[0]
    detections = []
    
    if hasattr(results, "boxes") and results.boxes is not None:
        boxes = results.boxes
        masks = getattr(results, 'masks', None)
        
        for i, box in enumerate(boxes):
            # Check if detection is a person (class 0 in COCO dataset)
            if int(box.cls) == 0 and float(box.conf) >= conf_threshold:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = (int(x1), int(y1), int(x2), int(y2))
                
                # Get corresponding mask if available
                mask = None
                if masks is not None and i < len(masks.data):
                    mask = masks.data[i].cpu().numpy()
                    # Resize mask to frame size if needed
                    if mask.shape != frame.shape[:2]:
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                
                detections.append((bbox, mask))

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
        print(f"    ⚠️  Not enough frames collected ({len(person_frames)}) for gesture detection")
        return False
        
    # Create temporary video for gesture detection
    temp_video_path = "temp_person_crop.avi"
    height, width = person_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    for frame in person_frames:
        out.write(frame)
    out.release()
    
    # Run gesture detection with higher confidence thresholds to reduce false positives
    gesture_detected = False
    try:
        if gesture_type == "wave":
            # Increase detection confidence to reduce false positives
            detector = WaveDetector(temp_video_path, fps, detection_confidence=0.7)  # Increased from 0.4
            detected_frames = detector.detect_wave_timestamps(show_ui=False, frame_skip=3)
        elif gesture_type == "hand_over_face":
            # Increase detection confidence to reduce false positives
            detector = HandOverFaceDetector(temp_video_path, fps, detection_confidence=0.6)  # Increased from 0.3
            detected_frames = detector.detect_hand_over_face_frames(show_ui=False, frame_skip=3)
        else:
            detected_frames = []

        if detected_frames:
            print(f"    ✅ {gesture_type} detected! Found {len(detected_frames)} gesture frames")
            gesture_detected = True
        else:
            print(f"    ❌ No {gesture_type} detected in this person's region")

    except Exception as e:
        print(f"    ⚠️  Error in gesture detection: {e}")
        gesture_detected = False
    
    # Clean up temporary file
    try:
        import os
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
    except:
        pass
    
    return gesture_detected  # Return boolean instead of frame index

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