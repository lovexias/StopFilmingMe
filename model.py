# editor_core.py
from collections import deque
from scipy.optimize import linear_sum_assignment 
from scipy.spatial.distance import cdist 
from dataclasses import dataclass
import numpy as np
import cv2
import mediapipe as mp
from utils import (
    WaveDetector,
    get_video_rotation,
    generate_thumbnails,
    blur_faces_of_person,
    HandOverFaceDetector,
    match_person_id,
    PersonTracker
)
# a tiny struct
@dataclass
class PersonState:
        face_box: tuple  # (x,y,w,h) in pixels
        last_seen: int   # frame index
        blur_active: bool
        misses: int      # consec. frames not seen
        history: deque   # past N centres (for wave detection)
        
def _iou(a, b):
        """Intersection-over-Union for two boxes (x, y, w, h)."""
        ax1, ay1, aw, ah = a; ax2, ay2 = ax1 + aw, ay1 + ah
        bx1, by1, bw, bh = b; bx2, by2 = bx1 + bw, by1 + bh
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        if inter == 0: return 0.0
        union = aw * ah + bw * bh - inter
        return inter / union

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
        
        # Person tracking for continuous blurring
        self.tracked_people = set()         # set of person_ids to blur
        self.person_tracker = None          # PersonTracker instance
        
        # MediaPipe instances - create once and reuse
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.7
        )
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=0,  # short-range
        min_detection_confidence=0.6
        )
        self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=4,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.3
        )
                        
        # Per-person face tracking cache - reset when person exits
        self.person_face_cache = {}  # person_id -> last_face_box

        # ── NEW person-tracking containers ───────────────
        self.person_states = {}      # pid → PersonState
        self.next_person_id = 0

    def __del__(self):
        if getattr(self, "mp_pose", None):
            self.mp_pose.close()
        if getattr(self, "mp_face", None):
            self.mp_face.close()
        if getattr(self, "mp_face_detection", None):
            self.mp_face_detection.close()
        if getattr(self, "mp_hands", None):
            self.mp_hands.close()


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
        self.tracked_people.clear()
        self.person_face_cache.clear()
        
        # Initialize person tracker
        self.person_tracker = PersonTracker(self.video_path, self.fps)

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

    def detect_and_blur_hand_segments(self, progress_callback=None):
        """
        Uses WaveDetector and HandOverFaceDetector to find gestures.
        Only blurs the specific person from when they first appear until they exit the frame.
        Returns list of (person_id, gesture_type, frame_idx).
        """
        if not self.video_path:
            return []

        # Initialize detectors
        wave_detector = WaveDetector(self.video_path, self.fps)
        wave_data = wave_detector.detect_wave_timestamps(show_ui=False, frame_skip=1)

        cover_detector = HandOverFaceDetector(self.video_path, self.fps)
        cover_data = cover_detector.detect_hand_over_face_frames(show_ui=False, frame_skip=1)

        # Existing persons dict: person_id -> landmarks
        existing_people = {}

        # Final data: {person_id: {gesture_type: (frame_idx, landmarks)}}
        person_gestures = {}

        def process_gesture(data, gesture_type):
            for frame_idx, landmarks in data:
                person_id = match_person_id(existing_people, landmarks)
                if person_id not in person_gestures:
                    person_gestures[person_id] = {}
                # append every occurrence to a list
                person_gestures[person_id] \
                    .setdefault(gesture_type, []) \
                    .append((frame_idx, landmarks))

        # Process both gestures
        process_gesture(wave_data, "wave")
        process_gesture(cover_data, "cover_face")

        # Process each person's gestures individually - don't track continuously
        gesture_timestamps = []
        
        for person_id, gestures in person_gestures.items():
            # Add this person to tracking list
            self.tracked_people.add(person_id)
            
            # Get representative landmarks for this person
            representative_landmarks = None
            gesture_frame = None
            for gesture_type, occurrences in gestures.items():
                if occurrences:
                    representative_landmarks = occurrences[0][1]  # Use first occurrence
                    gesture_frame = occurrences[0][0]  # Get the frame where gesture happened
                    break
            
            if representative_landmarks and gesture_frame is not None:
                # Instead of tracking continuously, find the segment where this person appears
                # starting from their first appearance until they exit
                self._blur_person_appearance_segment(
                    gesture_frame,
                    representative_landmarks,
                    person_id,
                    progress_callback=progress_callback
                )
            
            # Record gesture timestamps
            for gesture_type, occurrences in gestures.items():
                for frame_idx, landmarks in occurrences:
                    gesture_timestamps.append(
                        (person_id, gesture_type, frame_idx, landmarks)
                    )

        return gesture_timestamps

    def _blur_person_appearance_segment(
        self,
        gesture_frame: int,
        target_landmarks,
        person_id,
        progress_callback=None,
    ):
        """
        Blur from the first moment this person is detected up to the moment
        we have N consecutive frames without a positive match.

        We tolerate short detection drop-outs (e.g. occlusions, pose-estimator
        hic-cups) so the blur does **not** flicker while the person is still
        physically in the shot.
        """
        MISSES_ALLOWED = 5          # ← how many consecutive misses we tolerate
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ── search BACKWARDS to find true appearance start ───────────────
        misses = 0
        start_frame = gesture_frame
        for fi in range(gesture_frame, -1, -1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frm = cap.read()
            if not ok:
                break

            if self._is_person_present(frm, target_landmarks):
                start_frame = fi
                misses = 0
            else:
                misses += 1
                if misses >= MISSES_ALLOWED:
                    start_frame = fi + MISSES_ALLOWED
                    break

        # ── search FORWARDS to find true disappearance end ───────────────
        misses = 0
        end_frame = gesture_frame
        for fi in range(gesture_frame, total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frm = cap.read()
            if not ok:
                break

            if self._is_person_present(frm, target_landmarks):
                end_frame = fi
                misses = 0
            else:
                misses += 1
                if misses >= MISSES_ALLOWED:
                    end_frame = fi - MISSES_ALLOWED
                    break

        # ── blur every frame in [start_frame … end_frame] ────────────────
        for fi in range(start_frame, end_frame + 1):
            if fi in self.blurred_cache:          # already done
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frm = cap.read()
            if not ok:
                continue

            # Double-check (strict) before blurring
            if self._is_person_present(frm, target_landmarks):
                blurred = self._blur_faces_of_person_optimized(
                    frm, target_landmarks, person_id, tolerance=0.25
                )
                self.blurred_frames.add(fi)
                self.blurred_cache[fi] = blurred

            if progress_callback and fi % 5 == 0:
                progress_callback(fi + 1)

        cap.release()

    def _is_person_present(self, frame, target_landmarks, tolerance=0.25):
        """
        Check if the person with target_landmarks is present in the current frame.
        Uses stricter tolerance to avoid false positives with similar poses.
        """
        if frame is None or frame.size == 0 or not target_landmarks:
            return False
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = self.mp_pose.process(frame_rgb)
        
        if not pose_result.pose_landmarks:
            return False
            
        # Compare current landmarks with target landmarks
        curr_array = np.array([(lm.x, lm.y) for lm in pose_result.pose_landmarks.landmark])
        target_array = np.array([(lm.x, lm.y) for lm in target_landmarks])
        
        # Use more sophisticated matching - check multiple key points
        # Focus on more unique identifying landmarks (shoulders, hips, etc.)
        key_indices = [11, 12, 23, 24, 13, 14, 25, 26]  # shoulders, hips, elbows, knees
        
        if len(curr_array) > max(key_indices) and len(target_array) > max(key_indices):
            key_curr = curr_array[key_indices]
            key_target = target_array[key_indices]
            diff = np.linalg.norm(key_curr - key_target, axis=1)
            avg_diff = np.mean(diff)
            
            # Also check overall pose similarity with stricter tolerance
            all_diff = np.linalg.norm(curr_array - target_array, axis=1)
            all_avg_diff = np.mean(all_diff)
            
            # Person is present if both key points and overall pose match
            return avg_diff <= tolerance and all_avg_diff <= tolerance * 1.4

        else:
            # Fallback to all landmarks if key indices not available
            diff = np.linalg.norm(curr_array - target_array, axis=1)
            avg_diff = np.mean(diff)
            return avg_diff <= tolerance

    def _blur_segment(self, gesture_frame, target_landmarks,
                      tolerance=0.3, progress_callback=None):
        """
        Finds the first frame they appear → last frame they're visible,
        then blurs every frame in that window.
        """
        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pre-compute target landmarks for faster comparison
        target_array = np.array([(lm.x, lm.y) for lm in target_landmarks])

        # helper to compute avg landmark diff on a single frame
        def _avg_diff(frame):
            # Validate frame
            if frame is None or frame.size == 0:
                return float("inf")
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.mp_pose.process(rgb)
            
            if not res.pose_landmarks:
                return float("inf")
            
            curr_array = np.array([(lm.x, lm.y) for lm in res.pose_landmarks.landmark])
            diff = np.linalg.norm(curr_array - target_array, axis=1)
            return np.mean(diff)

        # ─── find window start ────────────────────────────────────────
        start = gesture_frame
        for fi in range(gesture_frame, max(0, gesture_frame - 30), -1):  # Limit search range
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frm = cap.read()
            if not ok or _avg_diff(frm) > tolerance:
                break
            start = fi

        # ─── find window end ──────────────────────────────────────────
        end = gesture_frame
        for fi in range(gesture_frame, min(total, gesture_frame + 30)):  # Limit search range
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frm = cap.read()
            if not ok or _avg_diff(frm) > tolerance:
                break
            end = fi

        # ─── now blur only [start…end] ────────────────────────────────
        for fi in range(start, end + 1):
            # Skip if already processed
            if fi in self.blurred_cache:
                continue
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frm = cap.read()
            if not ok:
                break
            
            # Use default person_id for legacy blur method
            b = self._blur_faces_of_person_optimized(frm, target_landmarks, "default", tolerance=tolerance)
            self.blurred_frames.add(fi)
            self.blurred_cache[fi] = b
            
            if progress_callback and fi % 5 == 0:
                progress_callback(fi + 1)

        cap.release()

    def _blur_faces_of_person_optimized(self, frame, target_landmarks, person_id, tolerance=0.25):
        """
        Optimized version of blur_faces_of_person that reuses MediaPipe instances
        and uses per-person face caching to prevent blurring when person is absent.
        Uses stricter tolerance to avoid false positives.
        """
        # Validate input
        if frame is None or frame.size == 0:
            return frame
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = self.mp_pose.process(frame_rgb)
        face_result = self.mp_face.process(frame_rgb)
        h, w, _ = frame.shape

        # If we have target landmarks, check if this person is in the frame
        person_present = True
        if target_landmarks and pose_result.pose_landmarks:
            # Use the same strict matching as _is_person_present
            curr_array = np.array([(lm.x, lm.y) for lm in pose_result.pose_landmarks.landmark])
            target_array = np.array([(lm.x, lm.y) for lm in target_landmarks])
            
            # Focus on key identifying landmarks
            key_indices = [11, 12, 23, 24, 13, 14, 25, 26]  # shoulders, hips, elbows, knees
            
            if len(curr_array) > max(key_indices) and len(target_array) > max(key_indices):
                key_curr = curr_array[key_indices]
                key_target = target_array[key_indices]
                diff = np.linalg.norm(key_curr - key_target, axis=1)
                avg_diff = np.mean(diff)
                
                # Also check overall pose similarity
                all_diff = np.linalg.norm(curr_array - target_array, axis=1)
                all_avg_diff = np.mean(all_diff)
                
                if avg_diff > tolerance or all_avg_diff > tolerance * 1.2:
                    person_present = False
            else:
                # Fallback to all landmarks
                diff = np.linalg.norm(curr_array - target_array, axis=1)
                avg_diff = np.mean(diff)
                if avg_diff > tolerance:
                    person_present = False

        # If person is not present, clear their face cache and return original frame
        if not person_present:
            if person_id in self.person_face_cache:
                self.person_face_cache[person_id] = None
            return frame

        # Handle face detection with per-person caching
        mask = np.zeros((h, w, 3), dtype=np.uint8)

        if face_result.detections:
            detection = face_result.detections[0]
            box = detection.location_data.relative_bounding_box
            x = max(0, int(box.xmin * w))
            y = max(0, int(box.ymin * h))
            w_box = min(int(box.width * w), w - x)
            h_box = min(int(box.height * h), h - y)
            
            # Update this person's face cache
            self.person_face_cache[person_id] = (x, y, w_box, h_box)
            mask[y:y + h_box, x:x + w_box] = 255
            
        elif person_id in self.person_face_cache and self.person_face_cache[person_id]:
            # Use cached face box only if person is still present
            x, y, w_box, h_box = self.person_face_cache[person_id]
            # Validate cached box
            if x >= 0 and y >= 0 and x + w_box <= w and y + h_box <= h:
                mask[y:y + h_box, x:x + w_box] = 255

        if np.any(mask):
            blurred = cv2.GaussianBlur(frame, (55, 55), 0)
            return np.where(mask == 255, blurred, frame)
        else:
            return frame

    def get_frame(self, frame_idx: int):
        """
        Returns a *blurred* BGR frame for previews / thumbnails.
        """
        if self.cap is None:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            return None

        # Update tracker + apply blur (same two lines you use in export_video)
        self._update_people(frame_idx, frame)
        for pid, st in self.person_states.items():
            if st.blur_active:
                frame = self._blur_box(frame, st.face_box)

        return frame

    


    def _blur_box(self, frame, box):
        """
        Gaussian–blur a rectangular sub-region in-place and return the frame.
        `box` is (x, y, w, h) in pixel coords.
        """
        x, y, w, h = box
        sub = frame[y : y + h, x : x + w]
        if sub.size == 0:            # guard against out-of-bounds
            return frame
        sub = cv2.GaussianBlur(sub, (55, 55), 0)
        frame[y : y + h, x : x + w] = sub
        return frame


    def export_video(self, output_path: str, progress_callback=None) -> bool:
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

            # First update the people tracker for this frame
            self._update_people(i, frame)

            # Apply blur to everyone whose blur_active flag is on
            for pid, st in self.person_states.items():
                if st.blur_active:
                    frame = self._blur_box(frame, st.face_box)

            out.write(frame)

            if progress_callback and i % 10 == 0:
                progress_callback(i + 1)


        in_cap.release()
        out.release()
        return True
       



    

       

        # ------------------------------------------------------------------
    def _update_people(self, frame_idx: int, frame_bgr):
        """
        Multi-person tracker + gesture logic (v2):
        • Hungarian face matching, same as before
        • NEW hand → nearest-face assignment
        • NEW wave = wrist swing (>40 px) over last 10 frames
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = self.mp_face_detection.process(rgb).detections or []
        hands_res = self.mp_hands.process(rgb)
        hands = hands_res.multi_hand_landmarks or []

        H, W, _ = frame_bgr.shape

        # ── 1. build current face boxes & centres ─────────────────────
        new_boxes, new_centres = [], []

        for det in faces:
            b = det.location_data.relative_bounding_box
            x, y, w, h = (int(b.xmin * W), int(b.ymin * H),
                          int(b.width * W), int(b.height * H))
            box = (x, y, w, h)

            # ── NEW: skip boxes that overlap any already-blurred face ──
            skip = False
            for st in self.person_states.values():
                if st.blur_active and _iou(box, st.face_box) > 0.4:
                    skip = True
                    break
            if skip:
                continue

            new_boxes.append(box)
            new_centres.append((x + w // 2, y + h // 2))


        # ── 2. Hungarian match faces to existing IDs (unchanged) ──────
        prev_ids  = list(self.person_states.keys())
        prev_cent = [(bx + bw // 2, by + bh // 2)
                     for (bx, by, bw, bh) in [self.person_states[i].face_box
                                              for i in prev_ids]]

        if prev_cent and new_centres:
            D = cdist(prev_cent, new_centres)
            rows, cols = linear_sum_assignment(D)
        else:
            rows, cols = [], []

        matched_new, matched_prev = set(), set()
        for r, c in zip(rows, cols):
            if D[r, c] < 60:                      # 60-px threshold
                pid = prev_ids[r]
                self.person_states[pid].face_box = new_boxes[c]
                self.person_states[pid].last_seen = frame_idx
                self.person_states[pid].misses = 0
                matched_new.add(c)
                matched_prev.add(pid)

        # create IDs for new faces
        for idx, box in enumerate(new_boxes):
            if idx in matched_new:
                continue
            pid = self.next_person_id
            self.next_person_id += 1
            self.person_states[pid] = PersonState(
                face_box=box,
                last_seen=frame_idx,
                blur_active=False,
                misses=0,
                history=deque(maxlen=10)   # wrist x-history
            )

        # ageing / forget
        for pid in list(self.person_states.keys()):
            if pid not in matched_prev:
                st = self.person_states[pid]
                st.misses += 1
                if st.misses >= 5:
                    del self.person_states[pid]

                # ── 3. Hand → face assignment (improved) ─────────────────────
        # Build inflated boxes so wrists must actually overlap person area
        INF = 40  # margin in pixels
        inflated = {}
        for pid, st in self.person_states.items():
            x, y, w, h = st.face_box
            inflated[pid] = (x - INF, y - INF, w + 2*INF, h + 2*INF)

        #  collect wrist coords
        wrists = []
        for h_lm in hands:
            wpt = h_lm.landmark[0]        # WRIST index
            wx, wy = int(wpt.x * W), int(wpt.y * H)
            wrists.append((wx, wy))

        #  helper: is (px,py) inside box?
        def _inside(px, py, box):
            x, y, w, h = box
            return x <= px <= x + w and y <= py <= y + h

        for wx, wy in wrists:
            chosen_pid = None

            # 3-A) primary: first face whose inflated box contains wrist,
            #       and which is *not yet* blur_active
            for pid, box in inflated.items():
                if not self.person_states[pid].blur_active and _inside(wx, wy, box):
                    chosen_pid = pid
                    break

        
            # 3-B) fallback: nearest face centre — but only for faces not yet blur_active
            if chosen_pid is None:
                candidates = {pid: st for pid, st in self.person_states.items()
                            if not st.blur_active}          # ← no **
                if candidates:
                    pid, st_min = min(
                        candidates.items(),
                        key=lambda kv: np.hypot(
                            wx - (kv[1].face_box[0] + kv[1].face_box[2] // 2),
                            wy - (kv[1].face_box[1] + kv[1].face_box[3] // 2),
                        ),
                    )
                    if np.hypot(
                        wx - (st_min.face_box[0] + st_min.face_box[2] // 2),
                        wy - (st_min.face_box[1] + st_min.face_box[3] // 2),
                    ) < 250:
                        chosen_pid = pid


            print(f"Frame {frame_idx:04d}: wrist→PID {chosen_pid}")

