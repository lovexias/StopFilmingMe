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
        First pass: Detect gestures using 60-frame skip interval.
        Returns list of (person_id, gesture_type, frame_idx, bbox).
        """
        if not self.video_path:
            return []

        print("PASS 1: Analyzing video for gesture detection...")
        
        gesture_timestamps = []
        frame_count = 0
        global_frame_skip = 60  # Process every 60 frames during detection phase
        detected_person_gestures = {}  # Track detected gestures per person
        
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            # Skip frames
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

            # Detect people
            people_detected = detect_multiple_people_yolov8(frame, conf_threshold=0.5)
            
            if people_detected:
                print(f"Frame {frame_count}: Analyzing {len(people_detected)} people")
                
                # Check each person for gestures
                for person_id, bbox in enumerate(people_detected, 1):
                    detected_gestures = detected_person_gestures.get(person_id, set())

                    # Only check for gestures we haven't detected for this person
                    if 'wave' not in detected_gestures:
                        wave_detected = detect_gesture_in_person_box(
                            bbox, cap, gesture_type="wave", 
                            fps=self.fps, duration_seconds=2
                        )
                        if wave_detected:
                            gesture_timestamps.append((person_id, 'wave', frame_count, bbox))
                            detected_gestures.add('wave')
                            print(f"  ✓ Wave detected for Person {person_id}!")

                    if 'cover_face' not in detected_gestures:
                        cover_detected = detect_gesture_in_person_box(
                            bbox, cap, gesture_type="hand_over_face",
                            fps=self.fps, duration_seconds=2
                        )
                        if cover_detected:
                            gesture_timestamps.append((person_id, 'cover_face', frame_count, bbox))
                            detected_gestures.add('cover_face')
                            print(f"  ✓ Cover face detected for Person {person_id}!")

                    detected_person_gestures[person_id] = detected_gestures

            if progress_callback:
                progress_callback(frame_count)

            # Skip to next frame block
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
    
    def blur_person_in_video(self, bbox, start_frame, progress_callback=None):
        """
        Second pass: Create clean blurred video for specific person.
        Uses 30-frame skip for YOLO detection but processes every frame.
        """
        if self.cap is None:
            return []

        blurred_frames = []
        frame_idx = 0
        yolo_skip_frames = 30  # Run YOLO detection every 30 frames
        last_detected_people = []
        last_matched_bbox = bbox  # Keep track of last matched position

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

            # Run YOLO detection every 30 frames
            if frame_idx % yolo_skip_frames == 0:
                current_people = detect_multiple_people_yolov8(frame, conf_threshold=0.5)
                # Try to find matching person
                for detected_bbox in current_people:
                    if match_person_to_blur_list(last_matched_bbox, [detected_bbox]):
                        last_matched_bbox = detected_bbox
                        break
                last_detected_people = current_people

            # Apply blurring if we have a match
            if last_matched_bbox is not None:
                frame = blur_faces_of_person(frame, last_matched_bbox)
                self.blurred_cache[frame_idx] = frame
                blurred_frames.append(frame_idx)

            if progress_callback:
                progress_callback(frame_idx)

            frame_idx += 1  # Process every frame for smooth blurring

        cap.release()
        return blurred_frames
