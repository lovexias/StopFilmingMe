# video_utils.py

import cv2
import subprocess
import json
import numpy as np
import mediapipe as mp

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
        
        # ─── Initialize MediaPipe once ────────────────────────────────
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.85
        )

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.85
        )

        self.drawer = mp.solutions.drawing_utils

    def __del__(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
        if hasattr(self, 'pose') and self.pose:
            self.pose.close()

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

            # Resize for faster processing
            frame = cv2.resize(frame, (640, 360))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process both hands and pose
            hand_results = self.hands.process(frame_rgb)
            pose_results = self.pose.process(frame_rgb)

            if hand_results.multi_hand_landmarks and pose_results.pose_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    if show_ui:
                        self.drawer.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                    # Calculate median x position for stability
                    xs = [lm.x for lm in hand_landmarks.landmark]
                    media_x = np.median(xs)  # More stable than mean
                    
                    direction = None
                    if last_x is not None:
                        threshold = 0.005
                        if media_x < last_x - threshold:
                            direction = "left"
                        elif media_x > last_x + threshold:
                            direction = "right"
                        
                        if direction and (not movement_history or movement_history[-1][0] != direction):
                            movement_history.append((direction, frame_count, pose_results.pose_landmarks.landmark))

                    # Keep only recent movements
                    movement_history = [(d, f, l) for d, f, l in movement_history if frame_count - f <= self.fps]
                    
                    # Detect wave pattern
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
        if show_ui:
            cv2.destroyAllWindows()

        return detected

# Detects if a hand is near the face by checking hand and nose landmark proximity.
class HandOverFaceDetector:
    def __init__(self, video_path, fps, detection_confidence=0.5):
        self.video_path = video_path
        self.fps = fps or 30.0
        
        # ─── Initialize MediaPipe once ────────────────────────────────
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.85
        )
        
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.85
        )
        
        self.drawer = mp.solutions.drawing_utils

    def __del__(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'pose') and self.pose:
            self.pose.close()
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()

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

            # Resize for faster processing
            frame = cv2.resize(frame, (640, 360))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_result = self.pose.process(frame_rgb)
            hands_result = self.hands.process(frame_rgb)

            if pose_result.pose_landmarks:
                nose = pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
                nose_x, nose_y = int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0])

                if hands_result.multi_hand_landmarks:
                    for hand_landmarks in hands_result.multi_hand_landmarks:
                        # Check multiple key points for better detection
                        key_points = [0, 4, 8, 12, 16, 20]  # Wrist, thumb, index, middle, ring, pinky tips
                        min_dist = float('inf')
                        
                        for lm_idx in key_points:
                            if lm_idx < len(hand_landmarks.landmark):
                                lm = hand_landmarks.landmark[lm_idx]
                                hand_x = int(lm.x * frame.shape[1])
                                hand_y = int(lm.y * frame.shape[0])
                                dist = np.hypot(nose_x - hand_x, nose_y - hand_y)
                                min_dist = min(min_dist, dist)

                        if min_dist < 40:
                            hand_over_face_frames.append((frame_count, pose_result.pose_landmarks.landmark))
                            print(f"Hand over face at frame {frame_count}")
                            break

            if show_ui:
                if pose_result.pose_landmarks:
                    self.drawer.draw_landmarks(frame, pose_result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                if hands_result.multi_hand_landmarks:
                    for hand_landmarks in hands_result.multi_hand_landmarks:
                        self.drawer.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                cv2.imshow("Hand Over Face Detection", frame)
                if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                    break

            frame_count += 1

        cap.release()
        if show_ui:
            cv2.destroyAllWindows()

        return hand_over_face_frames

# New PersonTracker class for continuous person tracking
class PersonTracker:
    def __init__(self, video_path, fps, detection_confidence=0.7):
        self.video_path = video_path
        self.fps = fps or 30.0
        
        # Initialize MediaPipe pose detection
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=0.5
        )
        
        # Tracking state
        self.tracked_people = {}  # person_id -> latest_landmarks
        
    def __del__(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'pose') and self.pose:
            self.pose.close()
    
    def track_person_continuous(self, target_landmarks, person_id, tolerance=0.4, progress_callback=None):
        """
        Track a person continuously throughout the video.
        Returns list of (start_frame, end_frame) tuples where person is present.
        """
        if not target_landmarks:
            return []
        
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Convert target landmarks to numpy array for faster comparison
        target_array = np.array([(lm.x, lm.y) for lm in target_landmarks])
        
        segments = []
        current_segment_start = None
        frame_count = 0
        
        # Track person appearance throughout video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 3rd frame for performance
            if frame_count % 3 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_result = self.pose.process(frame_rgb)
                
                person_found = False
                if pose_result.pose_landmarks:
                    # Compare with target landmarks
                    current_array = np.array([(lm.x, lm.y) for lm in pose_result.pose_landmarks.landmark])
                    diff = np.linalg.norm(current_array - target_array, axis=1)
                    avg_diff = np.mean(diff)
                    
                    if avg_diff < tolerance:
                        person_found = True
                        # Update tracking
                        self.tracked_people[person_id] = pose_result.pose_landmarks.landmark
                
                # Handle segment tracking
                if person_found and current_segment_start is None:
                    # Person entered frame
                    current_segment_start = frame_count
                elif not person_found and current_segment_start is not None:
                    # Person left frame
                    segments.append((current_segment_start, frame_count - 1))
                    current_segment_start = None
            
            frame_count += 1
            
            # Progress callback
            if progress_callback and frame_count % 30 == 0:
                progress_callback(frame_count)
        
        # Handle case where person is still in frame at video end
        if current_segment_start is not None:
            segments.append((current_segment_start, total_frames - 1))
        
        cap.release()
        
        # Merge close segments (within 1 second)
        merged_segments = self._merge_close_segments(segments, self.fps)
        
        print(f"Person {person_id} tracked in {len(merged_segments)} segments: {merged_segments}")
        return merged_segments
    
    def _merge_close_segments(self, segments, fps, gap_threshold=1.0):
        """
        Merge segments that are close together (within gap_threshold seconds)
        """
        if not segments:
            return []
        
        # Sort segments by start frame
        segments.sort(key=lambda x: x[0])
        
        merged = [segments[0]]
        gap_frames = int(gap_threshold * fps)
        
        for current_start, current_end in segments[1:]:
            last_start, last_end = merged[-1]
            
            # If current segment is close to previous, merge them
            if current_start - last_end <= gap_frames:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
        
        return merged
    
    def find_person_entry_exit(self, target_landmarks, tolerance=0.4):
        """
        Find the first and last frame where a person appears in the video.
        Returns (entry_frame, exit_frame) or (None, None) if person not found.
        """
        if not target_landmarks:
            return None, None
        
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Convert target landmarks to numpy array
        target_array = np.array([(lm.x, lm.y) for lm in target_landmarks])
        
        entry_frame = None
        exit_frame = None
        
        # Find first appearance
        for frame_count in range(0, total_frames, 3):  # Check every 3rd frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = self.pose.process(frame_rgb)
            
            if pose_result.pose_landmarks:
                current_array = np.array([(lm.x, lm.y) for lm in pose_result.pose_landmarks.landmark])
                diff = np.linalg.norm(current_array - target_array, axis=1)
                avg_diff = np.mean(diff)
                
                if avg_diff < tolerance:
                    entry_frame = frame_count
                    break
        
        # Find last appearance (search backwards)
        if entry_frame is not None:
            for frame_count in range(total_frames - 1, entry_frame - 1, -3):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_result = self.pose.process(frame_rgb)
                
                if pose_result.pose_landmarks:
                    current_array = np.array([(lm.x, lm.y) for lm in pose_result.pose_landmarks.landmark])
                    diff = np.linalg.norm(current_array - target_array, axis=1)
                    avg_diff = np.mean(diff)
                    
                    if avg_diff < tolerance:
                        exit_frame = frame_count
                        break
        
        cap.release()
        return entry_frame, exit_frame

# Optimized blur function - kept for compatibility but should use EditorCore's version
def blur_faces_of_person(frame, target_landmarks=None, tolerance=0.3):
    """
    Legacy function - creates new MediaPipe instances (not recommended)
    Use EditorCore._blur_faces_of_person_optimized instead
    """
    if target_landmarks is None:
        # Just blur all detected faces
        mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_result = mp_face.process(frame_rgb)
        h, w, _ = frame.shape
        
        if face_result.detections:
            mask = np.zeros((h, w, 3), dtype=np.uint8)
            for detection in face_result.detections:
                box = detection.location_data.relative_bounding_box
                x = max(0, int(box.xmin * w))
                y = max(0, int(box.ymin * h))
                w_box = min(int(box.width * w), w - x)
                h_box = min(int(box.height * h), h - y)
                mask[y:y + h_box, x:x + w_box] = 255
            
            blurred = cv2.GaussianBlur(frame, (55, 55), 0)
            frame = np.where(mask == 255, blurred, frame)
        
        mp_face.close()
        return frame
    
    # Original implementation with new instances (causes crashes)
    mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
    mp_pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.7)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_result = mp_pose.process(frame_rgb)
    face_result = mp_face.process(frame_rgb)
    h, w, _ = frame.shape

    if not pose_result.pose_landmarks:
        mp_face.close()
        mp_pose.close()
        return frame

    # Fast numpy comparison
    curr_array = np.array([(lm.x, lm.y) for lm in pose_result.pose_landmarks.landmark])
    target_array = np.array([(lm.x, lm.y) for lm in target_landmarks])
    diff = np.linalg.norm(curr_array - target_array, axis=1)
    avg_diff = np.mean(diff)

    if avg_diff > tolerance:
        mp_face.close()
        mp_pose.close()
        return frame

    mask = np.zeros((h, w, 3), dtype=np.uint8)

    if face_result.detections:
        detection = face_result.detections[0]
        box = detection.location_data.relative_bounding_box
        x = max(0, int(box.xmin * w))
        y = max(0, int(box.ymin * h))
        w_box = min(int(box.width * w), w - x)
        h_box = min(int(box.height * h), h - y)
        mask[y:y + h_box, x:x + w_box] = 255

    if np.any(mask):
        blurred = cv2.GaussianBlur(frame, (55, 55), 0)
        frame = np.where(mask == 255, blurred, frame)

    mp_face.close()
    mp_pose.close()
    return frame

def match_person_id(existing_people, new_landmarks, tolerance=0.7):
    """
    Match new_landmarks to existing people. Returns person_id if matched, else new ID.
    existing_people: dict of person_id -> landmarks
    """
    if not new_landmarks:
        return len(existing_people) + 1
    
    # Convert to numpy for faster comparison
    new_array = np.array([(lm.x, lm.y) for lm in new_landmarks])
    
    for pid, landmarks in existing_people.items():
        if landmarks:
            existing_array = np.array([(lm.x, lm.y) for lm in landmarks])
            diff = np.linalg.norm(existing_array - new_array, axis=1)
            avg_diff = np.mean(diff)
            
            if avg_diff < tolerance:
                return pid
    
    # New person
    new_id = len(existing_people) + 1
    existing_people[new_id] = new_landmarks
    return new_id