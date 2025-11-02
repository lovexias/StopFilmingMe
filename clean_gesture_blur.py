# clean_gesture_blur.py
import os
import time
import cv2
import numpy as np
import mediapipe as mp
from moviepy import VideoFileClip  # for rotation metadata

# === Shared project utilities (no duplication) ===
from utilities import (
    detect_multiple_people_yolov8,
    detect_gesture_in_person_box,
    PersonTracker,
    blur_faces_of_person,
    close_global_mediapipe,
)

# =========================
# Configuration (kept in sync with main.py)
# =========================
video_path = r"D:\01 KYLE\School\College\4th Year\3rd Term\THS-ST2\HOF-20251102T082520Z-1-001\HOF\indoor_hof\nearface_hof\onehandpalmfront_2ft_nearface_indoor.MOV"
OUTPUT_PATH = "clean_blurred_output.mp4"

SHOW_UI = True
UI_SCALE_FACTOR = 0.5
DISCOVERY_FRAME_SKIP = 70
ANALYSIS_FRAME_SKIP = 35
GESTURE_DURATION = 3
YOLO_DETECTION_INTERVAL = 10
GESTURE_TYPES = ["wave", "hand_over_face"]  # order matters (same as main’s loop intent)

# =========================
# Orientation helpers (keep)
# =========================
def get_video_rotation(video_path: str) -> int:
    """Get rotation using moviepy clip metadata; fall back to 0."""
    try:
        clip = VideoFileClip(video_path)
        rotation = int(getattr(clip, "rotation", 0) or 0)
        clip.close()
        return rotation if rotation in (0, 90, 180, 270) else 0
    except Exception:
        return 0

def orient_frame(frame, rotation: int):
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

# =========================
# UI helpers (keep)
# =========================
def draw_person_box(frame, bbox, pid, status="Detecting", color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"Person {pid}: {status}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.6, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    ty = max(th + 5, y1 - 10)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 10, ty + 4), color, -1)
    cv2.putText(frame, text, (x1 + 4, ty), font, scale, (255, 255, 255), thick)
    return frame

# pose overlay is purely visual
_mp_pose_for_ui = mp.solutions.pose.Pose()
def draw_skeleton_green(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame
    results = _mp_pose_for_ui.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return frame
    h, w = roi.shape[:2]
    pts = []
    for lm in results.pose_landmarks.landmark:
        px = int(lm.x * w) + x1
        py = int(lm.y * h) + y1
        pts.append((px, py))
        cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
    for i, j in mp.solutions.pose.POSE_CONNECTIONS:
        if i < len(pts) and j < len(pts):
            cv2.line(frame, pts[i], pts[j], (0, 255, 0), 2)
    return frame

def scale_frame_for_display(frame, factor):
    if factor == 1.0:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * factor), int(h * factor)))

def safe_destroy_window(name: str):
    try:
        if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(name)
    except cv2.error:
        pass

# =========================
# PASS 1 — Discover & analyze gestures (match main.py)
# =========================
def first_pass_detect_gestures(video_path, fps, rotation, tracker: PersonTracker):
    """
    Mirrors main.py:_on_detect_requested for consistent results:
    - discovery using DISCOVERY_FRAME_SKIP
    - adaptive clarity thresholds based on early sampling
    - analyze per person for GESTURE_TYPES
    - early stop per person if gesture found or face unclear
    """
    tracker.tracked_people.clear()
    tracker.next_id = 0

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("=" * 70)
    print("PASS 1: OPTIMIZED GESTURE DETECTION (Improved Skip Logic)")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Total Frames: {total_frames}, FPS: {fps:.2f}, Rotation: {rotation}°")

    # --- Pre-scan: estimate clarity (exact sampling like main.py) ---
    print("\nEstimating average video clarity...")
    sharp_samples, bright_samples = [], []
    for i in range(0, min(total_frames, 300), max(1, total_frames // 50 or 1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, fr = cap.read()
        if not ret:
            break
        fr = orient_frame(fr, rotation)
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        sharp_samples.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        bright_samples.append(gray.mean())
    cap.release()

    avg_sharp = float(np.mean(sharp_samples)) if sharp_samples else 50.0
    avg_bright = float(np.mean(bright_samples)) if bright_samples else 100.0
    print(f"Avg sharpness: {avg_sharp:.1f}, Avg brightness: {avg_bright:.1f}")

    sharp_thresh = max(10.0, avg_sharp * 0.4)
    bright_thresh = max(25.0, avg_bright * 0.5)
    area_thresh = 5000
    print(f"Using thresholds -> sharpness<{sharp_thresh:.1f}, brightness<{bright_thresh:.1f}, area<{area_thresh}")

    # --- Stage 1: Discover people (sparse scan) ---
    print("\nSTEP 1: Discovering people...")
    cap = cv2.VideoCapture(video_path)
    for frame_idx in range(0, total_frames, DISCOVERY_FRAME_SKIP):
        ret = cap.grab()  # quicker than set+read on some codecs; we’ll still reorient on retrieve
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame = orient_frame(frame, rotation)

        people_detected = detect_multiple_people_yolov8(frame, conf_threshold=0.5)
        if people_detected:
            tracker.update(frame, people_detected, frame_idx)

        if SHOW_UI:
            ui = frame.copy()
            for pid, pdata in tracker.tracked_people.items():
                draw_person_box(ui, pdata["bbox"], pid, "Tracking", (0, 255, 0))
            cv2.imshow("Pass 1 – Discovery", scale_frame_for_display(ui, UI_SCALE_FACTOR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # small delay to mimic GUI pacing
        time.sleep(0.02)

    cap.release()
    safe_destroy_window("Pass 1 – Discovery")
    discovered_people = list(tracker.tracked_people.keys())
    print(f"\nSTEP 1 COMPLETE: {len(discovered_people)} people discovered.")

    # --- Stage 2: Analyze gestures per person ---
    print("\nSTEP 2: Analyzing gestures (adaptive clarity filter)...")
    people_to_blur = []
    gesture_found_for_person = {pid: False for pid in discovered_people}
    unclear_face_for_person = {pid: False for pid in discovered_people}

    # process gesture types in order; stop early per person as in main.py
    for gesture_type in GESTURE_TYPES:
        print(f"\nChecking gesture type: {gesture_type}")

        for pid in discovered_people:
            if gesture_found_for_person[pid]:
                print(f"Skipping Person {pid} (already has gesture)")
                continue
            if unclear_face_for_person[pid]:
                print(f"Skipping Person {pid} (face unclear or not visible)")
                continue

            print(f"  Analyzing Person {pid} for {gesture_type}...")
            cap = cv2.VideoCapture(video_path)

            for frame_num in range(0, total_frames, ANALYSIS_FRAME_SKIP):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    break

                frame = orient_frame(frame, rotation)

                people_detected = detect_multiple_people_yolov8(frame, conf_threshold=0.5)
                if not people_detected:
                    time.sleep(0.01)
                    continue

                current_people = tracker.update(frame, people_detected, frame_num)
                if pid not in current_people:
                    time.sleep(0.01)
                    continue

                bbox = tuple(map(int, current_people[pid]["bbox"]))
                x1, y1, x2, y2 = bbox
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # adaptive clarity check (exactly like main.py)
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                brightness = gray.mean()
                area = (x2 - x1) * (y2 - y1)

                if sharpness < sharp_thresh or brightness < bright_thresh or area < area_thresh:
                    unclear_face_for_person[pid] = True
                    print(
                        f"    Face unclear for Person {pid} "
                        f"(sharp={sharpness:.1f}/{sharp_thresh:.1f}, "
                        f"bright={brightness:.1f}/{bright_thresh:.1f}) – skipping."
                    )
                    break

                if SHOW_UI:
                    vis = frame.copy()
                    draw_skeleton_green(vis, bbox)
                    draw_person_box(vis, bbox, pid, f"Checking {gesture_type.title()}...", (0, 255, 0))
                    cv2.imshow(f"Analyzing Person {pid}", scale_frame_for_display(vis, UI_SCALE_FACTOR))
                    cv2.waitKey(1)

                detected = detect_gesture_in_person_box(
                    bbox, cap, gesture_type, fps, duration_seconds=GESTURE_DURATION
                )

                if detected:
                    gesture_found_for_person[pid] = True
                    pdata = tracker.tracked_people[pid]
                    people_to_blur.append({
                        "person_id": pid,
                        "gesture": gesture_type,
                        "frame": frame_num,
                        "bbox": bbox,
                        "first_seen_frame": pdata.get("first_seen_frame", 0)
                    })
                    print(f"    Detected {gesture_type} for Person {pid}")
                    break

                # small delay to mimic GUI pacing
                time.sleep(0.01)

            cap.release()

            # close the per-person window if shown
            if SHOW_UI:
                safe_destroy_window(f"Analyzing Person {pid}")

        # stop outer gesture loop if all people done
        if all(gesture_found_for_person[pid] or unclear_face_for_person[pid] for pid in discovered_people):
            break

    # Deduplicate by person_id (last wins), consistent with main's finalization
    unique_people = {p["person_id"]: p for p in people_to_blur}
    people_to_blur = list(unique_people.values())

    print("\nDETECTION COMPLETE:")
    print(f"- Total people with gestures: {len(people_to_blur)}")
    print(f"- IDs: {[p['person_id'] for p in people_to_blur]}")
    print("=" * 70)
    print("PASS 1 COMPLETE — Ready for blurring.")
    print("=" * 70)

    return people_to_blur

# =========================
# PASS 2 — Blur & burn-in reason (reuse tracker for ID consistency)
# =========================
def second_pass_create_clean_video(video_path, people_to_blur, fps, rotation, out_w, out_h, tracker: PersonTracker):
    """
    Mirrors main.py:_on_blur_requested logic:
    - reuse same PersonTracker instance across passes
    - periodic YOLO re-detection
    - blur targeted IDs
    - burn-in reason label
    - UI preview window
    """
    reason_by_id = {p["person_id"]: p.get("gesture", "?") for p in people_to_blur}
    person_ids_to_blur = {p["person_id"] for p in people_to_blur}

    # Reset disappeared counts so re-detections don’t get dropped immediately
    for pid in tracker.tracked_people:
        tracker.tracked_people[pid]["disappeared"] = 0

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (out_w, out_h))

    print(f"PASS 2: Blurring people: {sorted(person_ids_to_blur)}")
    last_dets = []

    for fidx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame = orient_frame(frame, rotation)

        # periodic YOLO re-detection
        if fidx % YOLO_DETECTION_INTERVAL == 0:
            last_dets = detect_multiple_people_yolov8(frame, conf_threshold=0.5)

        cur_people = tracker.update(frame, last_dets, fidx)

        # UI overlay (preview)
        if SHOW_UI:
            ui = frame.copy()

        # blur targeted IDs
        for pid, pdata in cur_people.items():
            if pid in person_ids_to_blur:
                frame = blur_faces_of_person(frame, tuple(map(int, pdata["bbox"])))
                if SHOW_UI:
                    reason = reason_by_id.get(pid, "?")
                    draw_person_box(ui, pdata["bbox"], pid, f"BLURRED ({reason})", (0, 255, 0))
            elif SHOW_UI:
                draw_person_box(ui, pdata["bbox"], pid, "Normal", (0, 255, 255))

        # burn-in small reason label on the final output frame
        for pid, pdata in cur_people.items():
            if pid in person_ids_to_blur:
                reason = reason_by_id.get(pid, "?")
                x1, y1, x2, y2 = map(int, pdata["bbox"])
                cv2.rectangle(frame, (x1, max(0, y1 - 24)), (x1 + 160, y1), (0, 0, 0), -1)
                cv2.putText(frame, f"{pid}: {reason}", (x1 + 5, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if SHOW_UI:
            cv2.imshow("Pass 2 – Blurring", scale_frame_for_display(ui, UI_SCALE_FACTOR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        writer.write(frame)

        # tiny pacing for determinism vs GUI
        if fidx % 3 == 0:
            time.sleep(0.005)

    cap.release()
    writer.release()
    safe_destroy_window("Pass 2 – Blurring")
    print("PASS 2 COMPLETE.")


# =========================
# Main (standalone runner)
# =========================
def main():
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    rotation = get_video_rotation(video_path)
    out_w, out_h = (raw_h, raw_w) if rotation in (90, 270) else (raw_w, raw_h)

    print("=" * 70)
    print("CLEAN GESTURE BLURRING — Standalone Harness (matches app logic)")
    print("=" * 70)
    print(f"Input: {video_path}")
    print(f"FPS: {fps:.2f}, Size: {raw_w}x{raw_h}, Rotation: {rotation}")

    # use ONE tracker across both passes to keep IDs consistent
    tracker = PersonTracker(max_disappeared=30, feature_threshold=0.3, motion_threshold=200)

    people_to_blur = first_pass_detect_gestures(video_path, fps, rotation, tracker)
    second_pass_create_clean_video(video_path, people_to_blur, fps, rotation, out_w, out_h, tracker)

    print(f"Done! Output saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback; traceback.print_exc()
    finally:
        try:
            close_global_mediapipe()
        except Exception:
            pass
