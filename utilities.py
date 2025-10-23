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
import os
import torch



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
#DEBUG_TRACKING = True
#next_person_id = 1


def _select_device():
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon (optional)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _select_device()
USE_HALF = (DEVICE == "cuda")  # half precision only on CUDA

# Load once, move to device, and fuse layers
yolo_model = YOLO("yolov8m-seg.pt")
try:
    yolo_model.to(DEVICE)
    yolo_model.fuse()  # small speed bump
    # For Ultralytics >=8.2 this is enough; if you want hard FP16:
    if USE_HALF and hasattr(yolo_model.model, "half"):
        yolo_model.model.half()
except Exception:
    pass

# Optional — helps cuDNN pick optimal algos on variable sizes
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

try:
    cv2.setNumThreads(1)
except Exception:
    pass




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
    orig_h, orig_w = frame.shape[:2]
    if orig_w > 1280:
        scale = 1280 / orig_w
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        frame_small = cv2.resize(frame, (new_w, new_h))
    else:
        frame_small = frame
        scale = 1.0

    res = yolo_model.predict(
        source=frame_small,
        device=DEVICE,
        half=USE_HALF,
        verbose=False
    )[0]

    dets = []

    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls   = res.boxes.cls.cpu().numpy().astype(int)
        conf  = res.boxes.conf.cpu().numpy()
        masks = getattr(res, "masks", None)

        for i in range(xyxy.shape[0]):
            if cls[i] != 0 or conf[i] < conf_threshold:
                continue

            x1, y1, x2, y2 = xyxy[i]
            if scale != 1.0:
                x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
            bbox = (int(x1), int(y1), int(x2), int(y2))

            mask = None
            if masks is not None and masks.data is not None:
                m = masks.data[i].cpu().numpy()        # HxW float mask
                # keep it binary-ish and resize correctly
                m = (m > 0.5).astype(np.uint8)
                if scale != 1.0 or m.shape[:2] != (orig_h, orig_w):
                    mask = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                else:
                    mask = m

            dets.append((bbox, mask))

    return dets


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
    results = yolo_model.predict(
        source=frame_small,
        device=DEVICE,
        half=USE_HALF,
        verbose=False
    )[0]

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


def blur_faces_of_person(img, bbox, blur_type="gaussian", strength=50, solid_color=(64, 64, 64)):
    # bbox = (x1,y1,x2,y2)
    x1, y1, x2, y2 = map(int, bbox)
    h, w = img.shape[:2]
    x1 = np.clip(x1, 0, w-1); x2 = np.clip(x2, 1, w); 
    y1 = np.clip(y1, 0, h-1); y2 = np.clip(y2, 1, h)
    if x2 <= x1 or y2 <= y1:
        return img

    roi = img[y1:y2, x1:x2]
    out = img

    s = int(np.clip(strength, 0, 100))
    if s <= 0:
        return img

    if blur_type.lower() in ("solid", "rectangle", "box"):
        # Map strength 0..100 -> alpha 0..1 (0=transparent, 1=opaque)
        alpha = s / 100.0
        # Optional: give more finesse at low values
        # alpha = (s/100.0) ** 1.2     # comment the line above and use this if you want even lighter lows
        overlay = np.full_like(roi, solid_color, dtype=roi.dtype)
        blended = cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0)
        out[y1:y2, x1:x2] = blended

    elif blur_type.lower() in ("gaussian", "gauss"):
        # Kernel size 3..71, odd numbers; sigma auto
        k = int(np.interp(s, [0, 100], [3, 71]))
        k = k + (1 - k % 2)  # make odd
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        out[y1:y2, x1:x2] = blurred

    elif blur_type.lower() in ("pixelate", "pixel", "mosaic"):
        # Downscale factor 2..50 (higher => chunkier pixels)
        factor = max(2, int(np.interp(s, [0, 100], [2, 50])))
        sh, sw = roi.shape[:2]
        down = cv2.resize(roi, (max(1, sw // factor), max(1, sh // factor)), interpolation=cv2.INTER_LINEAR)
        up   = cv2.resize(down, (sw, sh), interpolation=cv2.INTER_NEAREST)
        out[y1:y2, x1:x2] = up

    else:
        # default to gaussian if unknown
        k = 9
        out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)

    return out

def face_bbox_in_person(frame, person_bbox):
    """Return a face bbox inside a person bbox; fallback to 'head' region."""
    x1, y1, x2, y2 = map(int, person_bbox)
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w-1)); x2 = max(x1+1, min(x2, w))
    y1 = max(0, min(y1, h-1)); y2 = max(y1+1, min(y2, h))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return person_bbox

    # MediaPipe expects RGB
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    res = mp_face_global.process(crop_rgb)

    if res and res.detections:
        b = res.detections[0].location_data.relative_bounding_box
        fx1 = x1 + int(b.xmin * (x2 - x1))
        fy1 = y1 + int(b.ymin * (y2 - y1))
        fw  = int(b.width  * (x2 - x1))
        fh  = int(b.height * (y2 - y1))
        fx2 = fx1 + fw
        fy2 = fy1 + fh
        # clamp
        fx1 = max(0, min(fx1, w-1)); fx2 = max(fx1+1, min(fx2, w))
        fy1 = max(0, min(fy1, h-1)); fy2 = max(fy1+1, min(fy2, h))
        return (fx1, fy1, fx2, fy2)

    # Fallback: top ~45% of the person box (a “head” heuristic)
    head_h = max(8, int(0.45 * (y2 - y1)))
    return (x1, y1, x2, min(y2, y1 + head_h))



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

def detect_gesture_in_person_box(person_box, frame_source, gesture_type="wave", fps=30, duration_seconds=2):
    """OPTIMIZED: Detect gestures within a person's bounding box."""
    
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
            
        adj_x1, adj_y1, adj_x2, adj_y2 = adjust_bounding_box_aspect_ratio(x1, y1, x2, y2, frame.shape)
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

def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = (ax2-ax1)*(ay2-ay1); area_b = (bx2-bx1)*(by2-by1)
    return inter / float(area_a + area_b - inter + 1e-6)

def expand_bbox(bb, w, h, pad=0.08):
    x1,y1,x2,y2 = bb
    bw, bh = x2-x1, y2-y1
    px, py = int(bw*pad), int(bh*pad)
    nx1 = max(0, x1 - px); ny1 = max(0, y1 - py)
    nx2 = min(w, x2 + px); ny2 = min(h, y2 + py)
    return (nx1, ny1, nx2, ny2)

