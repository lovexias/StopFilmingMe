# editor_core.py

import cv2
import mediapipe as mp
from video_utils import (
    WaveDetector,
    get_video_rotation,
    generate_thumbnails,
    blur_faces_in_frame
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

    def manually_blur_frame(self, frame_idx: int):
        """
        Blur exactly frame_idx (faces only), cache it in blurred_cache & blurred_frames.
        Returns the blurred BGR numpy array.
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

    def detect_and_blur_hand_segments(self, progress_callback=None):
        """
        Uses WaveDetector to find wave gestures, then blurs faces starting from
        each detected wave. Returns list of wave start frame indices.
        """
        if not self.video_path:
            return []

        wave_detector = WaveDetector(self.video_path, self.fps)
        wave_timestamps = wave_detector.detect_wave_timestamps(show_ui=False, frame_skip=3)
        wave_frame_indices = [int(ts * self.fps) for ts in wave_timestamps]

        blur_duration_frames = int(self.fps * 1.5)  # 1.5 seconds of blur per wave
        segment_starts = []

        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for wave_start in wave_frame_indices:
            segment_starts.append(wave_start)
            for i in range(wave_start, min(wave_start + blur_duration_frames, total)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    continue

                blurred = blur_faces_in_frame(frame)
                self.blurred_frames.add(i)
                self.blurred_cache[i] = blurred

                if progress_callback:
                    progress_callback(i + 1)

        cap.release()
        return segment_starts

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
                out.write(blur_faces_in_frame(frame))
            else:
                out.write(frame)

        in_cap.release()
        out.release()
        return True