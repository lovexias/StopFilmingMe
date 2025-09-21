import os
import cv2
import shutil
import tempfile
import subprocess
import numpy as np

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
    Core logic: video loading, frame access, gesture detection, blurring, export.
    Keeps UI (Qt) completely separate.
    """

    def __init__(self):
        self.cap = None
        self.video_path = None
        self.rotation_angle = 0
        self.total_frames = 0
        self.fps = 30.0
        self.src_w = 0
        self.src_h = 0

        # Detection knobs (editable from UI)
        self.confidence_threshold = 0.80  # 80%
        self.frame_skip = 2               # frames to skip between checks

        # Export knobs (editable from UI Export dialog)
        self.export_container = "mp4"      # mp4 | mov | avi | mkv
        self.export_codec = "mp4v"         # h264 | hevc | mjpeg | mp4v | prores (ffmpeg path)
        self.export_bitrate_mbps = 12      # only honored on ffmpeg path
        self.export_override_res = None    # (w, h) or None
        self.export_override_fps = None    # int or None

        # Which frames to blur:
        self.blurred_frames = set()        # set of frame indices
        self.blurred_cache = dict()        # frame_idx -> blurred BGR numpy array

    def _validate_fps(self, fps_value):
        """Validate and normalize FPS value to handle problematic video metadata"""
        try:
            fps = float(fps_value)
            
            # Handle common problematic FPS values
            if fps <= 0 or fps > 120:
                print(f"Warning: Unusual FPS value {fps}, defaulting to 30")
                return 30.0
            elif fps < 1:
                print(f"Warning: Very low FPS value {fps}, defaulting to 30") 
                return 30.0
            elif 23 <= fps <= 25:  # 23.976, 24, 25 FPS content
                return round(fps, 3)
            elif 29 <= fps <= 31:  # 29.97, 30 FPS content
                return round(fps, 3)
            elif 59 <= fps <= 61:  # 59.94, 60 FPS content
                return round(fps, 3)
            else:
                # Other valid FPS values
                return round(fps, 3)
                
        except (ValueError, TypeError):
            print(f"Warning: Invalid FPS value {fps_value}, defaulting to 30")
            return 30.0

    # ────────────────────────────────────────────────────────────────
    # Editable settings from UI
    # ────────────────────────────────────────────────────────────────
    def set_detection_params(self, confidence: float, frame_skip: int):
        """Called by the controller to apply 'Detection Settings' from the panel."""
        try:
            self.confidence_threshold = float(confidence)
        except Exception:
            self.confidence_threshold = 0.80
        try:
            self.frame_skip = max(1, int(frame_skip))
        except Exception:
            self.frame_skip = 2

    def set_export_format(self, container: str):
        c = (container or "mp4").lower()
        if c not in {"mp4", "mov", "avi"}:
            c = "mp4"
        self.export_container = c


    def set_export_codec(self, codec: str):
        self.export_codec = (codec or "h264").lower()

    def set_export_bitrate_mbps(self, mbps: int):
        try:
            self.export_bitrate_mbps = max(1, int(mbps))
        except Exception:
            self.export_bitrate_mbps = 12

    def set_export_overrides(self, resolution="Original", fps="Original"):
        res_map = {
            "Original": None,
            "1080p": (1920, 1080),
            "720p": (1280, 720),
            "480p": (854, 480),
        }
        self.export_override_res = res_map.get(resolution, None)
        self.export_override_fps = None if (fps in (None, "", "Original")) else int(fps)

    # ────────────────────────────────────────────────────────────────
    # Video loading
    # ────────────────────────────────────────────────────────────────
    def load_video(self, video_path: str) -> dict:
        if not os.path.exists(video_path):
            raise IOError(f"Video not found: {video_path}")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            raise IOError("Could not open video. File may be corrupt or unsupported.")

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass

        try:
            self.rotation_angle = get_video_rotation(video_path)
        except Exception:
            self.rotation_angle = 0

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        # --- corrected FPS handling ---
        fps_raw = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = self._validate_fps(fps_raw)
        print(f"Video FPS: raw={fps_raw}, validated={self.fps}")
        # --------------------------------
        

        self.src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        self.src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

        if self.total_frames <= 0:
            self.cap.release()
            self.cap = None
            raise IOError("Video has no readable frames (possibly partial or corrupt).")

        # Reset blur state
        self.blurred_frames.clear()
        self.blurred_cache.clear()

        return {
            "rotation_angle": self.rotation_angle,
            "total_frames": self.total_frames,
            "fps": self.fps
        }


    # ────────────────────────────────────────────────────────────────
    # Thumbnails
    # ────────────────────────────────────────────────────────────────
    def generate_thumbnails(self, num_thumbs: int = 16):
        if not self.video_path:
            return []
        return generate_thumbnails(
            self.video_path,
            self.total_frames,
            self.rotation_angle,
            num_thumbs=num_thumbs
        )

    # ────────────────────────────────────────────────────────────────
    # Gesture detection (first pass)
    # ────────────────────────────────────────────────────────────────
    def _apply_rotation(self, frame_bgr):
        if self.rotation_angle == 90:
            return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_angle == 180:
            return cv2.rotate(frame_bgr, cv2.ROTATE_180)
        elif self.rotation_angle == 270:
            return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame_bgr

    def detect_and_blur_hand_segments(self, progress_callback=None):
        if not self.video_path:
            return []

        print("PASS 1: Analyzing video for gesture detection...")

        gesture_timestamps = []
        frame_count = 0
        skip = max(1, int(self.frame_skip))  # respect UI frame-skip
        detected_person_gestures = {}

        cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        while cap.isOpened() and frame_count < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame = self._apply_rotation(frame)

            # use UI confidence threshold
            people = detect_multiple_people_yolov8(frame, conf_threshold=float(self.confidence_threshold))
            if people:
                # print(f"Frame {frame_count}: {len(people)} person(s)")
                for person_id, bbox in enumerate(people, 1):
                    detected_gestures = detected_person_gestures.get(person_id, set())

                    if "wave" not in detected_gestures:
                        if detect_gesture_in_person_box(bbox, cap, "wave", self.fps, 2):
                            gesture_timestamps.append((person_id, "wave", frame_count, bbox))
                            detected_gestures.add("wave")

                    if "cover_face" not in detected_gestures:
                        if detect_gesture_in_person_box(bbox, cap, "hand_over_face", self.fps, 2):
                            gesture_timestamps.append((person_id, "cover_face", frame_count, bbox))
                            detected_gestures.add("cover_face")

                    detected_person_gestures[person_id] = detected_gestures

            if progress_callback:
                try:
                    pct = int(min(100, (frame_count + 1) * 100 / max(1, total_frames)))
                    progress_callback(pct)
                except Exception:
                    pass

            frame_count += skip

        if progress_callback:
            try:
                progress_callback(100)
            except Exception:
                pass

        cap.release()
        return gesture_timestamps

    # ────────────────────────────────────────────────────────────────
    # Frame access
    # ────────────────────────────────────────────────────────────────
    def get_frame(self, frame_idx: int):
        if self.cap is None:
            return None

        if frame_idx in self.blurred_cache:
            return self.blurred_cache[frame_idx]

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return self._apply_rotation(frame)

    # ────────────────────────────────────────────────────────────────
    # Export helpers
    # ────────────────────────────────────────────────────────────────
    def _fourcc_for(self, codec: str):
        c = (codec or "").lower()
        if c in ("h264", "avc1", "x264"): return cv2.VideoWriter_fourcc(*"avc1")
        if c in ("hevc", "h265", "x265"): return cv2.VideoWriter_fourcc(*"HEVC")
        if c in ("mjpeg", "motionjpeg"):  return cv2.VideoWriter_fourcc(*"MJPG")
        if c in ("mp4v", "mpeg4"):        return cv2.VideoWriter_fourcc(*"mp4v")
        # fallback
        return cv2.VideoWriter_fourcc(*"mp4v")

    def _ffmpeg_codec(self, codec: str):
        c = (codec or "").lower()
        return {
            "h264":  "libx264",
            "hevc":  "libx265",
            "mjpeg": "mjpeg",
            "mp4v":  "mpeg4",
            "prores":"prores_ks",
        }.get(c, "libx264")

    def _write_intermediate_with_opencv(self, tmp_path, w, h, fps, progress_cb=None):
        """
        Writes an intermediate file (no scaling/fps change yet) using OpenCV.
        """
        in_cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        try:
            in_cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass
        if not in_cap.isOpened():
            return False

        total = int(in_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fourcc = self._fourcc_for("mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            in_cap.release()
            return False

        for i in range(total):
            ret, frame = in_cap.read()
            if not ret or frame is None:
                break
            frame = self._apply_rotation(frame)
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

            # apply blur if scheduled
            if i in self.blurred_frames and i in self.blurred_cache:
                writer.write(self.blurred_cache[i])
            elif i in self.blurred_frames:
                writer.write(blur_faces_of_person(frame))
            else:
                writer.write(frame)

            if progress_cb and (i % 10 == 0 or i == total - 1):
                try:
                    progress_cb(int(50 * (i + 1) / total))  # 0..50% for the intermediate stage
                except Exception:
                    pass

        writer.release()
        in_cap.release()
        return True

    # ────────────────────────────────────────────────────────────────
    # Export (FFmpeg when available; fallback to OpenCV)
    # ────────────────────────────────────────────────────────────────
    def export_video(self, output_path: str, progress_cb=None) -> bool:
        if not self.video_path:
            return False

        # Source props
        src_w = self.src_w or int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        src_h = self.src_h or int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        src_fps = self.fps if self.fps else float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)

        # Apply overrides
        out_w, out_h = self.export_override_res or (src_w, src_h)
        out_fps = self.export_override_fps or src_fps

        # Ensure extension matches container
        root, ext = os.path.splitext(output_path)
        want_ext = "." + (self.export_container or "mp4")
        if ext.lower() != want_ext:
            output_path = root + want_ext

        # If ffmpeg exists, use it to honor bitrate/codec/res/fps exactly
        ffmpeg = shutil.which("ffmpeg")

        if ffmpeg:
            # 1) write an intermediate (scaled to resolution so ffmpeg just handles codec/bitrate/fps cleanly)
            tmp_mp4 = os.path.join(tempfile.gettempdir(), "sfm_intermediate.mp4")
            if os.path.exists(tmp_mp4):
                try: os.remove(tmp_mp4)
                except Exception: pass

            ok = self._write_intermediate_with_opencv(tmp_mp4, out_w, out_h, out_fps, progress_cb=progress_cb)
            if not ok:
                return False

            # 2) transcode to final with requested codec/bitrate/container
            vcodec = self._ffmpeg_codec(self.export_codec)
            mbps = max(1, int(self.export_bitrate_mbps))

            cmd = [
                ffmpeg, "-y",
                "-i", tmp_mp4,
                "-c:v", vcodec,
                "-b:v", f"{mbps}M",
                "-r", str(out_fps),
                "-pix_fmt", "yuv420p",   # better compatibility
                output_path
            ]

            try:
                # We won't parse progress; split progress roughly 50..100%
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                if progress_cb:
                    try: progress_cb(100)
                    except Exception: pass
                return True
            except Exception:
                # fallback to plain OpenCV path if ffmpeg fails
                pass

        # ── OpenCV-only fallback ────────────────────────────────────
        in_cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        try:
            in_cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass
        if not in_cap.isOpened():
            return False

        fourcc = self._fourcc_for(self.export_codec)
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))
        if not writer.isOpened():
            in_cap.release()
            return False

        total = int(in_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        for i in range(total):
            ret, frame = in_cap.read()
            if not ret or frame is None:
                break

            frame = self._apply_rotation(frame)
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            if i in self.blurred_frames and i in self.blurred_cache:
                writer.write(self.blurred_cache[i])
            elif i in self.blurred_frames:
                writer.write(blur_faces_of_person(frame))
            else:
                writer.write(frame)

            if progress_cb and (i % 10 == 0 or i == total - 1):
                try:
                    progress_cb(int(100 * (i + 1) / total))
                except Exception:
                    pass

        in_cap.release()
        writer.release()
        return True

    # ────────────────────────────────────────────────────────────────
    # Blurring pass (track person across frames)
    # ────────────────────────────────────────────────────────────────
    def blur_person_in_video(self, bbox, start_frame=0, progress_callback=None):
        if self.cap is None:
            return []

        blurred_frames = []
        frame_idx = 0
        # reuse detection skip here? keep dense blur; otherwise it can skip faces
        refresh_every = 30
        last_matched_bbox = bbox

        cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame = self._apply_rotation(frame)

            # Refresh YOLO match periodically
            if frame_idx % refresh_every == 0:
                current_people = detect_multiple_people_yolov8(frame, conf_threshold=float(self.confidence_threshold))
                for detected_bbox in current_people:
                    if match_person_to_blur_list(last_matched_bbox, [detected_bbox]):
                        last_matched_bbox = detected_bbox
                        break

            if last_matched_bbox is not None:
                frame_b = blur_faces_of_person(frame, last_matched_bbox)
                self.blurred_cache[frame_idx] = frame_b
                self.blurred_frames.add(frame_idx)
                blurred_frames.append(frame_idx)

            if progress_callback and (frame_idx % 5 == 0 or frame_idx == total_frames - 1):
                try:
                    pct = int((frame_idx + 1) * 100 / max(1, total_frames))
                    progress_callback(pct)
                except Exception:
                    pass

            frame_idx += 1

        if progress_callback:
            try:
                progress_callback(100)
            except Exception:
                pass

        cap.release()
        return blurred_frames
