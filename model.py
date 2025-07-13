import cv2
import mediapipe as mp
from utilities import (
    WaveDetector,
    get_video_rotation,
    generate_thumbnails,
    blur_faces_of_person,
    HandOverFaceDetector,
    match_person_id,
    detect_multiple_people_yolov8,
    detect_gesture_in_person_box,
    match_person_to_blur_list
)

class EditorCore:
    """
    All video‐loading, thumbnail generation, face/hands detection+blurring,
    and export logic.  No Qt or UI code here.
    """

    def __init__(self):
        self.cap = None
        self.video_path = None
        self.rotation_angle = 0
        self.total_frames = 0
        self.fps = 30.0

        # Which frames to blur:
        self.blurred_frames = set()         # set of frame indices
        self.blurred_cache = dict()         # frame_idx -> blurred BGR numpy array

    def load_video(self, video_path: str) -> dict:
        """
        Open the video and read basic properties.  Clears any previous blur‐state.
        Returns a dict with {rotation_angle, total_frames, fps}.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.rotation_angle = get_video_rotation(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Clear any old blur state
        self.blurred_frames.clear()
        self.blurred_cache.clear()

        return {
            "rotation_angle": self.rotation_angle,
            "total_frames": self.total_frames,
            "fps": self.fps
        }

    def generate_thumbnails(self, num_thumbs: int = 16):
        """
        Returns a list of (frame_idx, RGB‐thumb_numpy) for num_thumbs equally spaced frames.
        """
        if not self.video_path:
            return []

        thumbs = generate_thumbnails(
            self.video_path,
            self.total_frames,
            self.rotation_angle,
            num_thumbs=num_thumbs
        )
        # That helper already returns (frame_idx, RGB numpy)
        return thumbs

    # def manually_blur_frame(self, frame_idx: int):
    #     """
    #     Blur exactly frame_idx (faces only), cache it in blurred_cache & blurred_frames.
    #     Returns the blurred BGR numpy array.
    #     """
    #     if self.cap is None:
    #         return None

    #     self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    #     ret, frame = self.cap.read()
    #     if not ret:
    #         return None

    #     blurred = blur_faces_of_person(frame)
    #     self.blurred_frames.add(frame_idx)
    #     self.blurred_cache[frame_idx] = blurred
    #     return blurred

    def detect_and_blur_hand_segments(self, progress_callback=None):
        """
        First pass: Only detect gestures and identify people.
        Returns list of (person_id, gesture_type, frame_idx, bbox).
        """
        if not self.video_path:
            return []

        print("PASS 1: Analyzing video for gesture detection...")
        
        gesture_timestamps = []
        frame_count = 0
        global_frame_skip = 60  # Process every 60 frames during detection
        detected_person_gestures = {}  # key: person_id, value: set of detected gestures
        
        cap = cv2.VideoCapture(self.video_path)
        
        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret:
                break

            # Apply rotation if needed
            if self.rotation_angle == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation_angle == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotation_angle == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Only process every 60 frames for gesture detection
            if frame_count % global_frame_skip != 0:
                frame_count += 1
                continue
            
            # Detect people
            people_detected = detect_multiple_people_yolov8(frame, conf_threshold=0.5)
            
            if not people_detected:
                frame_count += global_frame_skip
                continue

            print(f"Frame {frame_count}: Analyzing {len(people_detected)} people")
            
            # Check each person for gestures
            for person_id, bbox in enumerate(people_detected, 1):
                detected_gestures = detected_person_gestures.get(person_id, set())

                if 'wave' not in detected_gestures:
                    wave_gesture_frame = detect_gesture_in_person_box(
                        bbox, cap, gesture_type="wave", fps=self.fps, duration_seconds=2
                    )
                    if wave_gesture_frame is not None:
                        actual_frame = frame_count + wave_gesture_frame
                        gesture_timestamps.append((person_id, 'wave', actual_frame, bbox))

                if 'cover_face' not in detected_gestures:
                    cover_gesture_frame = detect_gesture_in_person_box(
                        bbox, cap, gesture_type="hand_over_face", fps=self.fps, duration_seconds=2
                    )
                    if cover_gesture_frame:
                        actual_frame = frame_count + cover_gesture_frame
                        gesture_timestamps.append((person_id, 'cover face', actual_frame, bbox))

                detected_person_gestures[person_id] = detected_gestures

            if progress_callback:
                progress_callback(frame_count)

            frame_count += global_frame_skip

        cap.release()
        return gesture_timestamps

    def get_frame(self, frame_idx: int):
        """
        Returns a BGR numpy for frame_idx, or a cached blurred version if available.
        """
        if self.cap is None:
            return None

        if frame_idx in self.blurred_cache:
            return self.blurred_cache[frame_idx]

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def export_video(self, output_path: str) -> bool:
        """
        Write a new MP4 where every frame in blurred_frames is replaced by the cached blurred version.
        Returns True on success, False otherwise.
        """
        if not self.video_path:
            return False

        in_cap = cv2.VideoCapture(self.video_path)
        w = int(in_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))

        total = int(in_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(total):
            ret, frame = in_cap.read()
            if not ret:
                break

            if i in self.blurred_frames and i in self.blurred_cache:
                out.write(self.blurred_cache[i])
            elif i in self.blurred_frames:
                out.write(blur_faces_of_person(frame))
            else:
                out.write(frame)

        in_cap.release()
        out.release()
        return True
    
    def blur_person_in_video(self, person_bbox, start_frame, progress_callback=None):
        """
        Blur the target person's face from their first gesture-detected frame (start_frame) onward.
        Tracks the person continuously even if not detected every frame.
        """
        if self.cap is None:
            return []

        blurred_frames = []
        frame_idx = start_frame
        yolo_skip_frames = 30
        last_detected_people = []
        last_matched_bbox = person_bbox  # Start tracking from this bbox

        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Apply rotation if needed
            if self.rotation_angle == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation_angle == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotation_angle == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Refresh detection every N frames
            if frame_idx % yolo_skip_frames == 0:
                last_detected_people = detect_multiple_people_yolov8(frame, conf_threshold=0.5)

            # Attempt to match current bbox
            matched_bbox = match_person_to_blur_list(last_matched_bbox, last_detected_people)
            if matched_bbox:
                last_matched_bbox = matched_bbox  # Update tracking
            else:
                matched_bbox = last_matched_bbox  # Fallback to last known bbox

            if matched_bbox:
                frame = blur_faces_of_person(frame, matched_bbox)
                self.blurred_cache[frame_idx] = frame
                blurred_frames.append(frame_idx)
                self.blurred_frames.add(frame_idx)

            if progress_callback:
                progress_callback(frame_idx)

            frame_idx += 1

        cap.release()
        return blurred_frames