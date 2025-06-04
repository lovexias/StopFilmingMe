# editor_core.py

import cv2
import mediapipe as mp
from video_utils import (
    get_video_rotation,
    generate_thumbnails,
    blur_faces_in_frame
)


class EditorCore:
    """
    This class knows nothing about Qt.  It only knows how to:
      - load a video from disk
      - generate thumbnails
      - keep track of which frames have been blurred
      - run MediaPipe to detect “hand → start blurring until face disappears”
      - manually blur a specific frame
      - export a final MP4 with all blurred frames baked in

    The UI (EditorPage) will instantiate one of these and then simply
    call its methods.  Core never touches QWidget, QPixmap, etc.
    """

    def __init__(self):
        # Video capture / state
        self.cap = None
        self.video_path = None
        self.rotation_angle = 0
        self.total_frames = 0
        self.fps = 30.0

        # Which frames should be blurred (indices)
        self.blurred_frames = set()
        # Cache: frame_idx → blurred BGR numpy array
        self.blurred_cache = {}

    def load_video(self, video_path):
        """
        Open the video and read its basic properties.  Clears any
        previous blur‐state, and returns metadata needed by the UI.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.rotation_angle = get_video_rotation(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Clear any previous blur state
        self.blurred_frames.clear()
        self.blurred_cache.clear()

        # Let the UI know: (rotation, total_frames, fps)
        return {
            "rotation_angle": self.rotation_angle,
            "total_frames": self.total_frames,
            "fps": self.fps
        }

    def generate_thumbnails(self, num_thumbs=16):
        """
        Returns a list of (frame_idx, RGB‐thumbnail‐numpy) for `num_thumbs` equally spaced frames.
        The UI will convert each numpy into a QImage/QPixmap.
        """
        if not self.video_path:
            return []

        thumbs = generate_thumbnails(
            self.video_path,
            self.total_frames,
            self.rotation_angle,
            num_thumbs=num_thumbs
        )
        # That `generate_thumbnails` helper already returns (idx, BGR→RGB) numpy arrays.
        return thumbs

    def manually_blur_frame(self, frame_idx):
        """
        Blur exactly `frame_idx`, store it in blurred_cache & blurred_frames.
        Returns the blurred BGR image (so UI can immediately display it).
        """
        if self.cap is None:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None

        blurred = blur_faces_in_frame(frame)
        self.blurred_frames.add(frame_idx)
        self.blurred_cache[frame_idx] = blurred
        return blurred

    def blur_face_at(self, frame_idx, click_x, click_y):
            """
            Run Mediapipe Face Detection on frame `frame_idx`. If the click at (click_x, click_y)
            falls inside one of the detected face bounding boxes, blur only that ROI, cache it,
            and return the blurred‐ROI frame (BGR). Otherwise, return None.
            """
            if not self.video_path:
                return None

            # 1) Seek to the requested frame:
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None

            h_frame, w_frame, _ = frame.shape

            # 2) Run Mediapipe Face Detection on the RGB copy:
            mp_face = mp.solutions.face_detection
            face_detector = mp_face.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb)

            face_detector.close()
            cap.release()

            if not results.detections:
                # No faces found → nothing to blur
                return None

            # 3) Loop through detections, convert each to pixel bbox and check if click is inside.
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x_min = int(bbox.xmin * w_frame)
                y_min = int(bbox.ymin * h_frame)
                box_w = int(bbox.width * w_frame)
                box_h = int(bbox.height * h_frame)

                # Clamp to valid pixel range
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                if x_min + box_w > w_frame:
                    box_w = w_frame - x_min
                if y_min + box_h > h_frame:
                    box_h = h_frame - y_min

                # Check if the click coordinates fall inside this face bbox:
                if (click_x >= x_min and click_x <= x_min + box_w
                and click_y >= y_min and click_y <= y_min + box_h):
                    # 4) Blur *only* this ROI:
                    roi = frame[y_min : y_min + box_h, x_min : x_min + box_w]
                    # You can choose any blur kernel size; 51x51 is fairly heavy
                    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                    frame[y_min : y_min + box_h, x_min : x_min + box_w] = blurred_roi

                    # 5) Cache the result for future calls:
                    self.blurred_frames.add(frame_idx)
                    self.blurred_cache[frame_idx] = frame.copy()
                    return frame

            # If we get here, the click was not on any face region:
            return None

    def detect_and_blur_hand_segments(self):
        """
        Runs MediaPipe Hands + Face Detection over every frame.  Once any hand is
        detected AND a face is visible, we begin a “blur segment” that continues
        until face detection fails.  Every frame in that segment is blurred and cached.

        Returns:
          A list of all (segment_start_frame_idx) values, so the UI can
          display timestamps (one entry per new segment).
        """
        if not self.video_path:
            return []

        mp_hands = mp.solutions.hands
        mp_face = mp.solutions.face_detection

        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        face_detector = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        in_blur_segment = False
        segment_start = None
        segment_starts = []  # list of frame_idx where a new blur‐segment began

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_res = hands.process(rgb)
            face_res = face_detector.process(rgb)

            hand_present = (hand_res.multi_hand_landmarks is not None)
            face_present = (face_res.detections is not None and len(face_res.detections) > 0)

            # If we see a hand + a face, and we’re not already in a blur segment,
            # start one now.
            if hand_present and face_present and not in_blur_segment:
                in_blur_segment = True
                segment_start = frame_idx
                segment_starts.append(segment_start)

            if in_blur_segment:
                # As long as face_present is true, continue to blur every frame:
                if face_present:
                    blurred = blur_faces_in_frame(frame)
                    self.blurred_frames.add(frame_idx)
                    self.blurred_cache[frame_idx] = blurred
                else:
                    # Once face disappears, end the blur segment
                    in_blur_segment = False

            frame_idx += 1

        hands.close()
        face_detector.close()
        cap.release()
        return segment_starts  # UI can convert each to a timestamp


    def get_frame(self, frame_idx):
        """
        Returns a **BGR** numpy array for frame_idx—but if frame_idx is in
        self.blurred_cache, it returns that blurred version instead of the raw.
        If it’s not cached as blurred, this will read directly from the capture.
        """
        if self.cap is None:
            return None

        # If we already blurred it, return the cached blur:
        if frame_idx in self.blurred_cache:
            return self.blurred_cache[frame_idx]

        # Otherwise, read from CV2 capture
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def export_video(self, output_path):
        """
        Writes out a new MP4 where every frame in self.blurred_frames is replaced by
        the cached blurred version.  All other frames remain unmodified.
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
                # In case we forgot to cache it, blur on the fly:
                out.write(blur_faces_in_frame(frame))
            else:
                out.write(frame)

        in_cap.release()
        out.release()
        return True
