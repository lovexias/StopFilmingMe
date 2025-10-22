import os
import cv2
import shutil
import tempfile
import subprocess
import numpy as np
import shutil
import tempfile

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
        self._frame_cache = {}  # Simple frame cache
        self._max_cached_frames = 50

        # NEW FOR HIGHLIGHT PERSON
        self.detected_people = []  # To store gesture detection results
        self.highlighted_person_ids = set() # Stores person_ids to highlight


    def close_video(self):
        # stop using the current capture and clear caches
        if getattr(self, "cap", None) is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self._frame_cache.clear()
        self.blurred_cache.clear()
        self.blurred_frames.clear()
        self.video_path = None
        self.total_frames = 0

    def load_video(self, video_path: str) -> dict:
        if not os.path.exists(video_path):
            raise IOError(f"Video not found: {video_path}")

        # >>> IMPORTANT: release the previous file handle <<<
        self.close_video()

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        ...

    def get_frame(self, frame_idx: int):
        # Check cache first
        if frame_idx in self._frame_cache:
            return self._frame_cache[frame_idx].copy()
        
        # Load from video
        if self.cap is None:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        
        frame = self._apply_rotation(frame)
        
        # Cache the frame (LRU eviction)
        if len(self._frame_cache) >= self._max_cached_frames:
            # Remove oldest entry
            oldest_key = next(iter(self._frame_cache))
            del self._frame_cache[oldest_key]
        
        self._frame_cache[frame_idx] = frame.copy()
        return frame

    def _validate_fps(self, fps_value):
        """Validate and normalize FPS value to handle problematic video metadata"""
        try:
            fps = float(fps_value)
            
            # Handle common problematic FPS values - ALLOW HIGHER FPS
            if fps <= 0:
                print(f"Warning: Invalid FPS value {fps}, defaulting to 30")
                return 30.0
            elif fps > 240:  # Increased from 120 to 240 for high-speed cameras
                print(f"Warning: Extremely high FPS value {fps}, capping at 240")
                return 240.0
            elif fps < 1:
                print(f"Warning: Very low FPS value {fps}, defaulting to 30") 
                return 30.0
            elif 23 <= fps <= 25:  # 23.976, 24, 25 FPS content
                return round(fps, 3)
            elif 29 <= fps <= 31:  # 29.97, 30 FPS content
                return round(fps, 3)
            elif 59 <= fps <= 61:  # 59.94, 60 FPS content
                return round(fps, 3)
            elif 119 <= fps <= 121:  # 120 FPS content
                return round(fps, 3)
            else:
                # Other valid FPS values - PRESERVE ORIGINAL
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

        self.close_video()
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


        # DEBUG: Check what FPS values we're getting
        fps_raw = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps_validated = self._validate_fps(fps_raw)
        
        print(f"=== FPS DEBUG ===")
        print(f"Raw FPS from video: {fps_raw}")
        print(f"Validated FPS: {fps_validated}")
        print(f"Total frames: {self.total_frames}")
        print(f"Calculated duration: {self.total_frames / fps_validated:.2f}s")
        print("================")
        
        self.fps = fps_validated
        

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
        """Multi-threaded gesture detection - won't freeze UI"""
        if not self.video_path:
            return []

        print("PASS 1: Multi-threaded gesture analysis...")
        
        # Split video into 4 chunks for parallel processing
        total_frames = self.total_frames
        chunk_size = total_frames // 4
        chunks = []
        
        for i in range(4):
            start_frame = i * chunk_size
            end_frame = (i + 1) * chunk_size if i < 3 else total_frames
            chunks.append((start_frame, end_frame))
        
        # Process chunks in parallel
        all_detections = []
        completed_chunks = 0
        
        def process_chunk(start_frame, end_frame):
            """Process a chunk of frames"""
            chunk_detections = []
            cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
            
            for frame_idx in range(start_frame, end_frame, self.frame_skip):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame = self._apply_rotation(frame)
                people = detect_multiple_people_yolov8(frame, conf_threshold=self.confidence_threshold)
                
                for person_id, bbox in enumerate(people, 1):
                    # Quick gesture detection
                    if detect_gesture_in_person_box(bbox, cap, "wave", self.fps, 1):  # Reduced duration
                        chunk_detections.append((person_id, "wave", frame_idx, bbox))
            
            cap.release()
            return chunk_detections
        
        # Use ThreadPoolExecutor for parallel processing
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_chunk, start, end) for start, end in chunks]
            
            for future in concurrent.futures.as_completed(futures):
                chunk_result = future.result()
                all_detections.extend(chunk_result)
                completed_chunks += 1
                
                if progress_callback:
                    progress_callback(int(100 * completed_chunks / 4))
        
        return sorted(all_detections, key=lambda x: x[2])

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

    def _write_intermediate_video_fast(self, out_path, out_w, out_h, out_fps, progress_cb=None):
        """
        Single fast pass:
        - Read each original frame
        - If a disk-cached blurred frame exists, use it
        else if RAM cache has it, use it
        else use the original
        - Write to a single OpenCV video (video-only)
        """
        cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass
        if not cap.isOpened():
            return False

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        # Use a fast, widely compatible writer; mp4v is fine
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            return False

        rotation = getattr(self, "rotation_angle", 0)
        blur_dir = getattr(self, "blur_cache_dir", None)

        for i in range(total):
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            # Apply rotation to the ORIGINAL frame so it matches the UI/export orientation
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Prefer disk-cached blurred frame (source of truth)
            used = False
            if blur_dir:
                jpg_path = os.path.join(blur_dir, f"{i:06d}.jpg")
                if os.path.exists(jpg_path):
                    b = cv2.imread(jpg_path)
                    if b is not None:
                        frame = b
                        used = True

            # Fallback to RAM cache if present
            if not used and i in self.blurred_cache:
                frame = self.blurred_cache[i]

            # Resize if user requested override
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            writer.write(frame)

            if progress_cb and (i % 25 == 0 or i == total - 1):
                # ~0–90% reserved for this step; remaining 10% for mux
                progress_cb(int(90 * (i + 1) / total))

        writer.release()
        cap.release()
        return True


    # ────────────────────────────────────────────────────────────────
    # Export (FFmpeg when available; fallback to OpenCV)
    # ────────────────────────────────────────────────────────────────
    def export_video(self, output_path, progress_cb=None):
        if not self.video_path:
            return False

        # use user’s chosen container/codec, but our fast path ignores codec during A (OpenCV),
        # then mux audio in B without re-encoding video
        src_w = self.src_w or int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        src_h = self.src_h or int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        src_fps = self.fps if self.fps else float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)

        out_w, out_h = self.export_override_res or (src_w, src_h)
        out_fps = self.export_override_fps or src_fps

        root, ext = os.path.splitext(output_path)
        want_ext = "." + (self.export_container or "mp4")
        if ext.lower() != want_ext:
            output_path = root + want_ext

        # 0) If nothing was blurred, fast remux/copy original -> output
        if not self.blurred_frames:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                # Stream copy both audio & video (no re-encode)
                cmd = [
                    ffmpeg, "-y", "-i", self.video_path,
                    "-c", "copy",
                    output_path
                ]
                subprocess.run(cmd, check=False)
                return os.path.exists(output_path)
            else:
                # As a fallback, just copy the file
                try:
                    shutil.copy2(self.video_path, output_path)
                    return True
                except Exception:
                    return False

        # 1) Step A: create a fast intermediate video (video only) with OpenCV
        tmp_dir = tempfile.mkdtemp(prefix="export_fast_")
        tmp_video = os.path.join(tmp_dir, "video_only.mp4")

        ok = self._write_intermediate_video_fast(
            tmp_video, out_w, out_h, out_fps, progress_cb=progress_cb
        )
        if not ok:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False

        # 2) Step B: Mux original audio onto the intermediate (no video re-encode)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            # We’ll just move the video-only file to output (no audio)
            shutil.move(tmp_video, output_path)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return True

        # Keep audio from original; copy video from intermediate
        cmd = [
            ffmpeg, "-y",
            "-i", tmp_video,    # video-only
            "-i", self.video_path,  # original for audio
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            output_path
        ]
        subprocess.run(cmd, check=False)

        shutil.rmtree(tmp_dir, ignore_errors=True)
        return os.path.exists(output_path)

        
    def _export_with_ffmpeg_and_audio(self, output_path: str, out_w: int, out_h: int, out_fps: float, progress_cb=None):
        """Export video with ffmpeg, preserving audio, using existing blurred caches only (no re-detect)."""
        import subprocess  # os, cv2, tempfile are already imported at module level

        vcodec = self._ffmpeg_codec(self.export_codec)
        mbps = max(1, int(self.export_bitrate_mbps))

        try:
            # Step 1: Create a temporary directory for frame processing
            with tempfile.TemporaryDirectory() as temp_dir:
                if progress_cb:
                    progress_cb(5)

                # Step 2: Process frames and save them as individual images
                frames_dir = os.path.join(temp_dir, "frames")
                os.makedirs(frames_dir, exist_ok=True)

                in_cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
                try:
                    in_cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                except Exception:
                    pass

                if not in_cap.isOpened():
                    return False

                total_frames = int(in_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

                for i in range(total_frames):
                    ret, frame = in_cap.read()
                    if not ret or frame is None:
                        break

                    # Apply rotation
                    frame = self._apply_rotation(frame)

                    # --- cache-first, no re-detect ---
                    if hasattr(self, 'blur_cache_dir') and os.path.exists(self.blur_cache_dir):
                        jpg_path = os.path.join(self.blur_cache_dir, f"{i:06d}.jpg")
                    else:
                        jpg_path = None

                    if jpg_path and os.path.exists(jpg_path):
                        # 1) Prefer disk-cached blurred frame
                        frame_to_write = cv2.imread(jpg_path)
                    elif i in self.blurred_cache:
                        # 2) Fallback to RAM cache
                        frame_to_write = self.blurred_cache[i]
                    else:
                        # 3) No blurred frame available -> write original frame as-is
                        frame_to_write = frame

                    # Resize if needed (honor export overrides)
                    if (frame_to_write.shape[1] != out_w) or (frame_to_write.shape[0] != out_h):
                        frame_to_write = cv2.resize(frame_to_write, (out_w, out_h), interpolation=cv2.INTER_AREA)

                    # Save PNG (lossless-ish) for ffmpeg muxing
                    frame_path = os.path.join(frames_dir, f"frame_%08d.png" % i)
                    cv2.imwrite(frame_path, frame_to_write, [cv2.IMWRITE_PNG_COMPRESSION, 1])

                    if progress_cb and (i % 10 == 0 or i == total_frames - 1):
                        # 5–85% for frame processing
                        progress_cb(int(5 + (i + 1) * 80 / max(1, total_frames)))

                in_cap.release()

                if progress_cb:
                    progress_cb(85)

                # Step 3: Use ffmpeg to combine frames with original audio
                frame_pattern = os.path.join(frames_dir, "frame_%08d.png")

                cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(out_fps),
                    "-i", frame_pattern,           # Input: processed frames
                    "-i", self.video_path,         # Input: original video (for audio)
                    "-map", "0:v",                 # Use video from first input (frames)
                    "-map", "1:a",                 # Use audio from second input (original video)
                    "-c:v", vcodec,
                    "-b:v", f"{mbps}M",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-shortest",
                    "-pix_fmt", "yuv420p",
                    "-r", str(out_fps),
                    output_path
                ]

                if progress_cb:
                    progress_cb(90)

                print(f"Running ffmpeg: {' '.join(cmd[:8])}...")

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes
                )

                if result.returncode != 0:
                    print(f"FFmpeg error (return code {result.returncode}):")
                    print(f"stderr: {result.stderr[-1000:]}")
                    return False

                if progress_cb:
                    progress_cb(100)

                print(f"Export completed successfully: {output_path}")
                return True

        except subprocess.TimeoutExpired:
            print("FFmpeg export timed out")
            return False
        except Exception as e:
            print(f"Export with audio failed: {e}")
            import traceback
            traceback.print_exc()
            return False


    def _export_opencv_fallback(self, output_path: str, out_w: int, out_h: int, out_fps: float, progress_cb=None):
        """Fallback export using OpenCV (no audio).
        Never re-detect on export. For each frame:
        1) Prefer disk-cached blurred JPG (if present)
        2) Else use RAM cache (self.blurred_cache[i])
        3) Else write the original frame
        """

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
        rotation = getattr(self, "rotation_angle", 0)
        blur_dir = getattr(self, "blur_cache_dir", None)

        for i in range(total):
            ret, original_frame = in_cap.read()
            if not ret or original_frame is None:
                break

            # Apply rotation to the original so it matches preview/export orientation
            if rotation == 90:
                original_frame = cv2.rotate(original_frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                original_frame = cv2.rotate(original_frame, cv2.ROTATE_180)
            elif rotation == 270:
                original_frame = cv2.rotate(original_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # ---------- Per-frame decision ----------
            out_frame = None

            # 1) Prefer disk-cached blurred JPG
            jpg_path = os.path.join(blur_dir, f"{i:06d}.jpg") if blur_dir else None
            if jpg_path and os.path.exists(jpg_path):
                out_frame = cv2.imread(jpg_path)

            # 2) Fallback to RAM cache
            if out_frame is None and i in self.blurred_cache:
                out_frame = self.blurred_cache[i]

            # 3) Else: no blurred frame available — write original (NO re-detect)
            if out_frame is None:
                out_frame = original_frame

            # Resize if needed to meet output dimensions
            if out_frame.shape[1] != out_w or out_frame.shape[0] != out_h:
                out_frame = cv2.resize(out_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            writer.write(out_frame)

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
                
                base = self.blurred_cache.get(frame_idx, frame)
                frame_b = blur_faces_of_person(base, last_matched_bbox)
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