# ---- SAFETY SWITCHES MUST BE SET BEFORE ANY OTHER IMPORTS ----
import os, faulthandler, sys
faulthandler.enable()

# Safer TF Lite / OpenMP on Windows
os.environ.setdefault("DISABLE_XNNPACK", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import webbrowser
import time
import cv2
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QFileDialog, QMessageBox, QDialog,
    QListWidgetItem  # Add this import
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QElapsedTimer
from PyQt5.QtGui import QPixmap, QIcon, QPainter, QColor, QPen, QBrush

from view import KeyboardShortcutsDialog
from view import EditorPanel, ProcessingDialog, ExportDialog
from model import EditorCore

from utilities import (
    PersonTracker,
    detect_multiple_people_yolov8,
    detect_gesture_in_person_box
)

# Add these constants
DISCOVERY_FRAME_SKIP = 70
ANALYSIS_FRAME_SKIP = 35
YOLO_DETECTION_INTERVAL = 10
GESTURE_DURATION = 3  # seconds

# Initialize ORB and matcher for feature tracking
orb = cv2.ORB_create(nfeatures=500)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Load YOLO segmentation model
yolo_model = YOLO('yolov8m-seg.pt')

# ---- Function to blur the face of a person ----
def blur_faces_of_person(frame, bbox):
    """
    This function blurs the face within the bounding box.
    """
    # Ensure bbox is a tuple and unpack it properly
    if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
    else:
        print(f"Error: Bounding box is not a tuple or list with 4 values, it's {type(bbox)}: {bbox}")
        return frame

    # Apply Gaussian blur to the face region
    face_roi = frame[y1:y2, x1:x2]  # Extract the face region
    blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)

    # Place the blurred face back into the frame
    frame[y1:y2, x1:x2] = blurred_face

    return frame

# ---------------- Workers ----------------
class BlurPersonWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, core, sel_pid):
        super().__init__()
        self.core = core
        self.sel_pid = sel_pid

    def run(self):
        import cv2
        # Load video properties
        cap = cv2.VideoCapture(self.core.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rotation = getattr(self.core, "rotation_angle", 0)

        self.core.blurred_cache.clear()  # Clear any previous blurred frames
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Apply rotation if needed
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Blur the selected person's face
            for person in self.core.detected_people:
                if person["person_id"] == self.sel_pid:
                    frame = blur_faces_of_person(frame, person["bbox"])

            # Cache the blurred frame
            self.core.blurred_cache[frame_idx] = frame

            # Emit progress
            self.progress.emit(int(frame_idx / total_frames * 100))

        cap.release()
        self.finished.emit()

class GestureDetectWorker(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)

    def __init__(self, core):
        super().__init__()
        self.core = core

    def run(self):
        # Get video properties
        cap = cv2.VideoCapture(self.core.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rotation = self.core.rotation_angle
        
        # First pass: Discover people and detect gestures
        person_tracker = PersonTracker(max_disappeared=30, feature_threshold=0.3, motion_threshold=200)
        discovered_people = []
        people_with_gestures = []
        
        # STEP 1: Discover all unique people - Process fewer frames
        sample_frames = range(0, total_frames, DISCOVERY_FRAME_SKIP)
        for frame_count in sample_frames:
            self.progress.emit(int((frame_count / total_frames) * 50))
            
            frame = self.core.get_frame(frame_count)
            if frame is None:
                continue
                
            # Add confidence threshold to reduce false detections    
            people_detected = detect_multiple_people_yolov8(frame, conf_threshold=0.5)
            if people_detected:
                current_people = person_tracker.update(frame, people_detected, frame_count)
                for person_id in current_people:
                    if person_id not in discovered_people:
                        discovered_people.append(person_id)
        
        # STEP 2: Analyze each person for gestures
        if discovered_people:
            frames_per_person = total_frames // len(discovered_people)
            for i, person_id in enumerate(discovered_people):
                progress = 50 + (i / len(discovered_people) * 50)
                self.progress.emit(int(progress))
                
                # Get person data from tracker
                person_data = person_tracker.tracked_people.get(person_id)
                if not person_data:
                    continue
                
                # Only analyze a portion of frames for each person
                start_frame = max(0, person_data.get('first_seen_frame', 0))
                end_frame = min(total_frames, start_frame + frames_per_person)
                
                has_gesture = self._analyze_person_across_entire_video(
                    person_id, 
                    person_tracker,
                    start_frame,
                    end_frame,
                    fps,
                    rotation
                )
                
                if has_gesture:
                    frame_idx = person_data.get('first_seen_frame', 0)
                    bbox = person_data.get('bbox')
                    people_with_gestures.append({
                        'person_id': person_id,
                        'frame': frame_idx,
                        'bbox': bbox
                    })
        
        cap.release()
        self.finished.emit(people_with_gestures)

    def _analyze_person_across_entire_video(self, person_id, person_tracker, start_frame, end_frame, fps, rotation):
        """Analyze a specific person between start_frame and end_frame to detect gestures."""
        for frame_num in range(start_frame, end_frame, ANALYSIS_FRAME_SKIP):
            frame = self.core.get_frame(frame_num)
            if frame is None:
                continue
            
            people_detected = detect_multiple_people_yolov8(frame)
            if not people_detected:
                continue
                
            current_people = person_tracker.update(frame, people_detected, frame_num)
            if person_id in current_people:
                person_data = current_people[person_id]
                gesture_detected = detect_gesture_in_person_box(
                    person_data['bbox'],
                    self.core.cap,
                    "wave",
                    fps,
                    duration_seconds=GESTURE_DURATION
                )
                if gesture_detected:
                    return True
        
        return False


class ExportWorker(QThread):
    progress = pyqtSignal(int)         # 0–100
    done = pyqtSignal(bool, str)       # ok, out_path

    def __init__(self, core, out_path):
        super().__init__()
        self.core = core
        self.out_path = out_path

    def run(self):
        def cb(pct):
            self.progress.emit(int(pct))
        ok = self.core.export_video(self.out_path, progress_cb=cb)
        self.done.emit(bool(ok), self.out_path if ok else "")


# ---------------- Main Window ----------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(create_eye_icon())
        self.setWindowTitle("StopFilming - Privacy Protection Video Editor")

        # Model
        self.core = EditorCore()
        # Shared tracker across passes for consistent IDs
        self.person_tracker = PersonTracker(max_disappeared=30, feature_threshold=0.3, motion_threshold=200)

        # View
        self.editor_panel = EditorPanel()
        self.setCentralWidget(self.editor_panel)
        self.setWindowTitle("StopFilming")

        # Wire signals
        self.editor_panel.importRequested.connect(self._on_import_requested)
        self.editor_panel.playToggled.connect(self._on_play_toggled)
        self.editor_panel.frameChanged.connect(self._on_frame_changed)
        self.editor_panel.detectRequested.connect(self._on_detect_requested)
        self.editor_panel.blurRequested.connect(self._on_blur_requested)
        self.editor_panel.thumbnailClicked.connect(self._on_thumbnail_clicked)
        self.editor_panel.gestureItemClicked.connect(self._on_gesture_item_clicked)
        self.editor_panel.exportRequested.connect(self._on_save_project)

        # Playback timer
        self.play_timer = QTimer()
        self.play_clock = QElapsedTimer()
        self.play_start_frame = 0
        self.play_timer.timeout.connect(self._on_timer_tick)

        # Add memory monitor
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self._check_memory)
        self.memory_timer.start(5000)

        # Menubar
        menubar = self.menuBar()  # Add this line to define the menubar
        self.menuBar().setStyleSheet(""" 
            QMenuBar { background-color: #2D3748; color: #E2E8F0; spacing: 6px; padding: 2px 10px; }
            QMenuBar::item { background: transparent; padding: 4px 12px; }
            QMenuBar::item:selected { background-color: #4FD1C7; color: #1A202C; border-radius: 4px; }
            QMenu { background-color: #2D3748; color: #E2E8F0; border: 1px solid #4A5568; margin: 2px; }
            QMenu::item { padding: 6px 20px; }
            QMenu::item:selected { background-color: #4A5568; }
        """)

        # File
        file_menu = menubar.addMenu("File")
        open_vid_action = QAction("Open Video...", self)
        open_vid_action.setShortcut("Ctrl+O")
        open_vid_action.triggered.connect(self._on_import_requested)
        file_menu.addAction(open_vid_action)

        save_action = QAction("Save Project...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)

        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit
        edit_menu = menubar.addMenu("Edit")
        undo_action = QAction("Undo", self); undo_action.setShortcut("Ctrl+Z"); undo_action.triggered.connect(lambda: None)
        redo_action = QAction("Redo", self); redo_action.setShortcut("Ctrl+Y"); redo_action.triggered.connect(lambda: None)
        edit_menu.addAction(undo_action); edit_menu.addAction(redo_action)
        edit_menu.addSeparator()
        preferences_action = QAction("Preferences...", self)
        preferences_action.triggered.connect(self._toggle_appearance)
        edit_menu.addAction(preferences_action)

        # View
        view_menu = menubar.addMenu("View")
        toggle_thumbs_action = QAction("Toggle Thumbnails", self, checkable=True)
        toggle_thumbs_action.setChecked(True)
        toggle_thumbs_action.triggered.connect(
            lambda checked: self.editor_panel.thumbnail_scroll.setVisible(checked)
        )
        view_menu.addAction(toggle_thumbs_action)

        toggle_markers_action = QAction("Toggle Markers Panel", self, checkable=True)
        toggle_markers_action.setChecked(True)
        toggle_markers_action.triggered.connect(
            lambda checked: self.editor_panel.gesture_list.parentWidget().setVisible(checked)
        )
        view_menu.addAction(toggle_markers_action)

        fullscreen_action = QAction("Fullscreen", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        # Tools
        tools_menu = menubar.addMenu("Tools")
        detect_action = QAction("Detect Gestures", self)
        detect_action.triggered.connect(self.editor_panel.detectRequested.emit)
        tools_menu.addAction(detect_action)

        blur_action = QAction("Blur Current Frame", self)
        blur_action.triggered.connect(lambda: self.editor_panel.blurRequested.emit(self.editor_panel.current_frame_idx))
        tools_menu.addAction(blur_action)

        export_action = QAction("Export Blurred Video...", self)
        export_action.triggered.connect(self._on_save_project)
        tools_menu.addAction(export_action)

        clear_blurs_action = QAction("Clear All Blurs", self)
        clear_blurs_action.triggered.connect(self._on_clear_blurs)
        tools_menu.addAction(clear_blurs_action)

        # Help
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About StopFilming", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.triggered.connect(self._show_shortcuts_reference)
        help_menu.addAction(shortcuts_action)

    # ---------- Menus and Controller slots ----------
    def _check_memory(self):
        """Monitor memory usage and show warnings"""
        try:
            import psutil
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            if memory_mb > 2048:  # More than 2GB
                print(f"⚠️ High memory usage: {memory_mb:.0f}MB")
                if hasattr(self.core, '_frame_cache'):
                    self.core._frame_cache.clear()
                    print("🧹 Cleared frame cache")
                import gc
                gc.collect()
                if memory_mb > 4096:
                    QMessageBox.warning(self, "Memory Warning", f"High memory usage detected ({memory_mb:.0f}MB).")
        except ImportError:
            pass  # psutil not available

    # ---------- Controller slots ----------
    def _on_import_requested(self):
        vid_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.MP4 *.avi *.AVI *.mov *.MOV);;All Files (*)"
        )
        if not vid_path:
            return

        meta = self.core.load_video(vid_path)
        old_panel = self.editor_panel
        if hasattr(old_panel, 'cleanup_audio_resources'):
            old_panel.cleanup_audio_resources()
        old_panel.deleteLater()

        self.editor_panel = EditorPanel()
        self.setCentralWidget(self.editor_panel)
        self.editor_panel.importRequested.connect(self._on_import_requested)
        self.editor_panel.playToggled.connect(self._on_play_toggled)
        self.editor_panel.frameChanged.connect(self._on_frame_changed)
        self.editor_panel.detectRequested.connect(self._on_detect_requested)
        self.editor_panel.blurRequested.connect(self._on_blur_requested)
        self.editor_panel.thumbnailClicked.connect(self._on_thumbnail_clicked)
        self.editor_panel.gestureItemClicked.connect(self._on_gesture_item_clicked)
        self.editor_panel.exportRequested.connect(self._on_save_project)

        self.editor_panel.set_video_info(
            rotation_angle=meta["rotation_angle"],
            total_frames=meta["total_frames"],
            fps=meta["fps"]
        )
        self.editor_panel.video_path = self.core.video_path  # sets up audio

        frame0 = self.core.get_frame(0)
        if frame0 is not None:
            frame0 = self._apply_rotation(frame0)
            self.editor_panel.display_frame(frame0, 0)

        thumbs = self.core.generate_thumbnails(num_thumbs=16)
        self.editor_panel.add_thumbnails(thumbs)

        self.setWindowTitle(f"StopFilming — Editing: {vid_path}")
        screen = QApplication.primaryScreen()
        rect = screen.availableGeometry()
        self.showNormal()
        self.resize(rect.width(), rect.height())
        self.repaint()
        QApplication.processEvents()

    # ---------- Other functions remain unchanged...


    def _on_save_project(self):
        if not self.core.video_path:
            QMessageBox.information(self, "Export", "No video loaded.")
            return

        base, _ = os.path.splitext(os.path.basename(self.core.video_path))
        suggest_name = f"{base}_export"
        suggest_dir = os.path.dirname(self.core.video_path) or os.path.expanduser("~")

        dlg = ExportDialog(self, suggest_name=suggest_name, suggest_dir=suggest_dir)
        if dlg.exec_() != QDialog.Accepted:
            return

        cfg = dlg.result_values()
        out_path = cfg["path"]
        container = (cfg.get("container") or "mp4").lower()
        codec     = (cfg.get("codec") or "mp4v").lower()

        if container == "mp4" and codec in ("h264", "avc1", "x264"):
            codec = "mp4v"
            try:
                self.statusBar().showMessage("Using MPEG-4 (mp4v) for maximum compatibility.", 5000)
            except Exception:
                pass

        self.core.set_export_format(container)
        self.core.set_export_codec(codec)
        self.core.set_export_bitrate_mbps(int(cfg.get("bitrate_mbps", 12)))
        self.core.set_export_overrides(cfg.get("resolution", "Original"), cfg.get("fps", "Original"))

        self.proc = ProcessingDialog(self, title="Exporting – StopFilming",
                                     message="Exporting video…", total_steps=100)
        self.proc.setWindowModality(Qt.ApplicationModal)
        self.proc.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.proc.show()
        QApplication.processEvents()

        self.export_thread = ExportWorker(self.core, out_path)
        self.export_thread.progress.connect(
            lambda pct: self.proc.set_progress(int(max(0, min(100, pct))), label=f"{int(pct)}%")
        )

        def _done(ok, path):
            if getattr(self, "proc", None):
                self.proc.finish("Export complete" if ok else "Export failed")
                self.proc = None
            if ok:
                QMessageBox.information(self, "Export", f"Exported to:\n{path}")
            else:
                QMessageBox.warning(self, "Export", "Failed to export edited video.")
            try:
                self.export_thread.quit()
                self.export_thread.wait()
            except Exception:
                pass
            self.export_thread = None

        self.export_thread.done.connect(_done)
        self.export_thread.start()

    def _on_clear_blurs(self):
        self.core.blurred_frames.clear()
        self.core.blurred_cache.clear()
        self.editor_panel.clear_markers()
        self.editor_panel.blur_button.setEnabled(False)

    def _on_play_toggled(self, play: bool):
        if not self.core or not self.core.video_path:
            return

        if play:
            if self.core.cap is None or not self.core.cap.isOpened():
                self.core.cap = cv2.VideoCapture(self.core.video_path, cv2.CAP_FFMPEG)
                try:
                    self.core.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Smaller buffer for responsiveness
                except Exception:
                    pass

            start_at = max(0, int(self.editor_panel.current_frame_idx))
            self.core.cap.set(cv2.CAP_PROP_POS_FRAMES, start_at)

            self.play_start_frame = start_at
            self.play_clock.start()

            # Calculate proper timer interval based on FPS
            fps = max(1.0, float(self.core.fps))
            # Target slightly higher than FPS for smooth playback
            timer_interval = max(8, int(1000.0 / fps / 1.2))  # 20% faster than frame rate
            
            self.play_timer.start(timer_interval)
            self.editor_panel.toggle_button.setText("Pause")

            self.editor_panel.audio_play_from_frame(self.editor_panel.current_frame_idx, fps)
        else:
            self.play_timer.stop()
            self.editor_panel.toggle_button.setText("Play")
            self.editor_panel.audio_pause()

    def _apply_rotation(self, frame):
        if frame is None:
            return None
        ra = getattr(self.core, "rotation_angle", 0) or 0
        if ra == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif ra == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif ra == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _on_frame_changed(self, frame_idx: int):
        try:
            self.play_timer.stop()
            self.editor_panel.toggle_button.setText("Play")

            if self.core.cap is not None:
                self.core.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Check blurred cache first
            if frame_idx in self.core.blurred_cache:
                img = self.core.blurred_cache[frame_idx].copy()  # Make a copy to be safe
            else:
                img = self.core.get_frame(frame_idx)
                if img is not None:
                    img = self._apply_rotation(img)
            
            if img is not None:
                self.editor_panel.display_frame(img, frame_idx)
                self.editor_panel.audio_seek_to_frame(frame_idx, self.core.fps)
        except Exception as e:
            print(f"Error changing frame: {e}")

    def _on_timer_tick(self):
        cap = self.core.cap
        if cap is None or not cap.isOpened():
            self.play_timer.stop()
            self.editor_panel.toggle_button.setText("Play")
            self.editor_panel.audio_pause()
            return

        ret, frame = cap.read()
        if not ret or frame is None:
            self.play_timer.stop()
            self.editor_panel.toggle_button.setText("Play")
            self.editor_panel.audio_pause()
            return

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        display_frame = self._apply_rotation(frame)

        # --- Use blurred frame cache if available ---
        if hasattr(self, "blur_cache_dir") and hasattr(self, "core"):
            frame_path = os.path.join(self.blur_cache_dir, f"{frame_idx:06d}.jpg")
            if frame_idx in self.core.blurred_cache:
                display_frame = self.core.blurred_cache[frame_idx]
            elif os.path.exists(frame_path):
                blurred = cv2.imread(frame_path)
                self.core.blurred_cache[frame_idx] = blurred
                display_frame = blurred

            # Keep cache small (±15 frames around current)
            keys = sorted(self.core.blurred_cache.keys())
            for k in keys:
                if abs(k - frame_idx) > 15:
                    del self.core.blurred_cache[k]

        self.editor_panel.display_frame(display_frame, frame_idx)
        self.editor_panel.current_frame_idx = frame_idx



    def _on_detect_requested(self):
        """Detect gestures (wave + hand_over_face) and mark people to blur."""
        if not self.core.video_path:
            QMessageBox.information(self, "Detection", "No video loaded.")
            return

        video_path = self.core.video_path
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rotation = getattr(self.core, "rotation_angle", 0)
        cap.release()

        print("=" * 70)
        print("PASS 1: GESTURE DETECTION – CLEAN LOGIC")
        print("=" * 70)
        print(f"Video: {video_path}")
        print(f"Total Frames: {total_frames}, FPS: {fps:.2f}, Rotation: {rotation}°")

        # Processing dialog for progress
        self.proc = ProcessingDialog(self, title="Detecting Gestures", message="Analyzing video...", total_steps=100)
        self.proc.show()
        QApplication.processEvents()

        # Initialize tracker
        person_tracker = PersonTracker(max_disappeared=30, feature_threshold=0.3, motion_threshold=200)
        discovered_people = []
        people_to_blur = []
        gestures_to_check = ["wave", "hand_over_face"]

        # --- STEP 1: Discover all unique people ---
        print("\n📋 STEP 1: Discovering people...")
        cap = cv2.VideoCapture(video_path)
        for frame_idx in range(0, total_frames, DISCOVERY_FRAME_SKIP):
            ret, frame = cap.read()
            if not ret:
                break

            # Apply rotation if necessary
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            people_detected = detect_multiple_people_yolov8(frame, conf_threshold=0.5)
            if people_detected:
                current_people = person_tracker.update(frame, people_detected, frame_idx)
                for pid in current_people.keys():
                    if pid not in discovered_people:
                        discovered_people.append(pid)
                        print(f"  👤 Found new Person ID {pid} at frame {frame_idx}")

            progress = int((frame_idx / total_frames) * 40)
            self.proc.set_progress(progress)
            self.statusBar().showMessage(f"Discovering people... {progress}%")
            QApplication.processEvents()

        cap.release()
        print(f"\n✅ STEP 1 COMPLETE: {len(discovered_people)} people discovered.")

        # --- STEP 2: Analyze gestures per person ---
        print("\n🔍 STEP 2: Analyzing gestures for each discovered person...")
        cap = cv2.VideoCapture(video_path)
        total_tasks = len(discovered_people) * len(gestures_to_check)
        task_count = 0

        for gesture_type in gestures_to_check:
            print(f"\n🎯 Detecting gesture: {gesture_type}")
            for pid in discovered_people:
                print(f"  ➤ Analyzing Person ID {pid} for {gesture_type}...")
                gesture_detected = False

                for frame_num in range(0, total_frames, ANALYSIS_FRAME_SKIP):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if rotation == 90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif rotation == 180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif rotation == 270:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    people_detected = detect_multiple_people_yolov8(frame, conf_threshold=0.5)
                    if not people_detected:
                        continue

                    current_people = person_tracker.update(frame, people_detected, frame_num)
                    if pid in current_people:
                        person_data = current_people[pid]
                        bbox = person_data['bbox']
                        detected = detect_gesture_in_person_box(bbox, cap, gesture_type, fps, duration_seconds=GESTURE_DURATION)
                        if detected:
                            gesture_detected = True
                            print(f"    ✅ {gesture_type} detected for Person {pid}")
                            people_to_blur.append({
                                'person_id': pid,
                                'gesture': gesture_type,
                                'frame': frame_num,
                                'bbox': bbox
                            })
                            break

                if not gesture_detected:
                    print(f"    ❌ No {gesture_type} detected for Person {pid}")

                task_count += 1
                progress = 40 + int((task_count / total_tasks) * 60)
                self.proc.set_progress(progress)
                self.statusBar().showMessage(f"Analyzing gestures... {progress}%")
                QApplication.processEvents()

        cap.release()

        # Merge duplicates (one entry per person)
        unique_people = {p['person_id']: p for p in people_to_blur}
        people_to_blur = list(unique_people.values())

        print("\n🎉 GESTURE DETECTION COMPLETE:")
        print(f"- People to blur: {len(people_to_blur)}")
        print(f"- IDs: {[p['person_id'] for p in people_to_blur]}")

        # Update UI
        self.editor_panel.gesture_list.clear()
        for p in people_to_blur:
            item = QListWidgetItem(f"Person {p['person_id']} - {p['gesture'].title()}")
            item.setData(Qt.UserRole, p)
            self.editor_panel.gesture_list.addItem(item)

        has_items = self.editor_panel.gesture_list.count() > 0
        self.editor_panel.blur_button.setEnabled(has_items)
        self.statusBar().showMessage(f"Detected {len(people_to_blur)} gesture(s)")
        if hasattr(self, "proc"):
            self.proc.finish("Gesture detection complete")
            self.proc = None

        # Save for next pass
        self.people_to_blur = people_to_blur
        self.person_tracker = person_tracker
        print("=" * 70)
        print("PASS 1 COMPLETE — Ready for blurring.")
        print("=" * 70)


    def _on_blur_requested(self, _frame_idx_from_button: int):
        """Blur the selected person and keep all previously blurred people persistent."""
        import tempfile, gc, cv2, os

        # Ensure gesture detections exist
        if not hasattr(self, "people_to_blur") or not self.people_to_blur:
            QMessageBox.information(self, "Blur", "No detected gestures found.")
            return

        # Ensure a person is selected
        selected_item = self.editor_panel.gesture_list.currentItem()
        if not selected_item:
            QMessageBox.information(self, "Blur", "Please select a person to blur from the list.")
            return

        selected_data = selected_item.data(Qt.UserRole)
        if not selected_data or "person_id" not in selected_data:
            QMessageBox.warning(self, "Blur", "Invalid selection.")
            return

        selected_pid = selected_data["person_id"]
        print(f"🎯 Selected person for blurring: ID {selected_pid}")

        # Create the persistent set if not already
        if not hasattr(self, "persistent_blur_ids"):
            self.persistent_blur_ids = set()
        self.persistent_blur_ids.add(selected_pid)
        print(f"🧩 Current persistent blur IDs: {self.persistent_blur_ids}")

        video_path = self.core.video_path
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rotation = getattr(self.core, "rotation_angle", 0)
        fps = cap.get(cv2.CAP_PROP_FPS)

        self.editor_panel.start_blur_progress()
        QApplication.processEvents()

        # Reuse or create disk cache
        if not hasattr(self, "blur_cache_dir"):
            import tempfile
            self.blur_cache_dir = tempfile.mkdtemp(prefix="blur_cache_")
        cache_dir = self.blur_cache_dir
        print(f"📁 Using cache directory: {cache_dir}")

        self.core.blurred_cache.clear()
        print(f"Blurring all persistent Person IDs: {self.persistent_blur_ids}")

        # Reset tracker disappeared counts
        for pid in self.person_tracker.tracked_people:
            self.person_tracker.tracked_people[pid]["disappeared"] = 0

        detection_interval = YOLO_DETECTION_INTERVAL
        last_tracked_people = {}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply rotation
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Run YOLO every N frames
            if frame_idx % detection_interval == 0:
                people_detected = detect_multiple_people_yolov8(frame, conf_threshold=0.5)
                if people_detected:
                    current_people = self.person_tracker.update(frame, people_detected, frame_idx)
                    last_tracked_people = current_people.copy()
                else:
                    current_people = last_tracked_people.copy()
            else:
                current_people = last_tracked_people.copy()

            blurred_frame = frame.copy()

            # Blur all currently persistent people
            for pid, pdata in current_people.items():
                if pid in self.persistent_blur_ids:
                    bbox = pdata["bbox"]
                    blurred_frame = blur_faces_of_person(blurred_frame, bbox)

            # Save blurred frame to cache
            frame_path = os.path.join(cache_dir, f"{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, blurred_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # Keep small memory window
            self.core.blurred_cache[frame_idx] = blurred_frame
            if len(self.core.blurred_cache) > 20:
                oldest_key = min(self.core.blurred_cache.keys())
                del self.core.blurred_cache[oldest_key]

            # Progress feedback
            if frame_idx % 50 == 0:
                progress = int((frame_idx / total_frames) * 100)
                self.editor_panel.set_blur_progress(progress)
                self.statusBar().showMessage(
                    f"Blurring IDs {list(self.persistent_blur_ids)}... {progress}%"
                )
                QApplication.processEvents()

            if frame_idx % 300 == 0:
                gc.collect()

            frame_idx += 1

        cap.release()
        gc.collect()

        print(f"\n🎉 COMPLETE! Persistent blur IDs now: {self.persistent_blur_ids}")
        print(f"Frames stored in cache: {cache_dir}")

        self.editor_panel.finish_blur_progress(True)
        self.editor_panel.update()
        self.statusBar().showMessage(
            f"Blurring updated for {len(self.persistent_blur_ids)} person(s)"
        )
        QMessageBox.information(
            self, "Blur Updated",
            f"Blurring complete for Person {selected_pid}.\n"
            f"Now blurring {len(self.persistent_blur_ids)} person(s) in total."
        )


    def _on_gesture_detection_finished(self, people_with_gestures):
        """Handle completion of gesture detection."""
        # Close the processing dialog first
        if hasattr(self, "proc"):
            self.proc.finish("Detection complete")
            self.proc = None

        # Convert dictionary format to tuple format expected by sorting
        segment_starts = [
            (person_data['person_id'], 
             "wave", 
             person_data['frame'],
             person_data['bbox']) 
            for person_data in people_with_gestures
        ]
        
        # Sort by frame number
        segment_starts = sorted(segment_starts, key=lambda s: s[2])
        
        # Clear existing items
        self.editor_panel.gesture_list.clear()
        
        # Add detected gestures to UI
        for person_id, gesture, frame, bbox in segment_starts:
            item = QListWidgetItem(f"Person {person_id} - Frame {frame}")
            item.setData(Qt.UserRole, {
                "person_id": person_id,
                "frame": frame,
                "bbox": bbox
            })
            self.editor_panel.gesture_list.addItem(item)

        has_items = self.editor_panel.gesture_list.count() > 0
        self.editor_panel.blur_button.setEnabled(has_items)
        self.statusBar().showMessage(f"Detected {self.editor_panel.gesture_list.count()} gesture(s)")

    def _on_thumbnail_clicked(self, frame_idx: int):
        self.play_timer.stop()
        self.editor_panel.toggle_button.setText("Play")
        img = self.core.get_frame(frame_idx)
        self.editor_panel.display_frame(img, frame_idx)
        self.editor_panel.blur_button.setEnabled(True)

    def _on_gesture_item_clicked(self, payload):
        pid, gest, bbox = "?", "", None

        if isinstance(payload, dict):
            frame_idx = payload.get("frame", 0)
            pid = payload.get("person_id", "?")
            gest = str(payload.get("gesture", "")).capitalize()
            bbox = payload.get("bbox")
        elif hasattr(payload, "data"):
            data = payload.data(Qt.UserRole)
            if isinstance(data, dict):
                frame_idx = data.get("frame", 0)
                pid = data.get("person_id", "?")
                gest = str(data.get("gesture", "")).capitalize()
                bbox = data.get("bbox")
            else:
                frame_idx = int(data)
        else:
            frame_idx = int(payload) if isinstance(payload, (int, float)) else 0

        if isinstance(frame_idx, float):
            frame_idx = int(round(frame_idx * max(1, self.core.fps)))

        frame_idx = max(0, min(int(frame_idx), max(0, self.core.total_frames - 1)))

        self.play_timer.stop()
        self.editor_panel.toggle_button.setText("Play")

        if hasattr(self.editor_panel, "frame_slider"):
            sld = self.editor_panel.frame_slider
            try:
                sld.blockSignals(True)
                if sld.maximum() != max(0, self.core.total_frames - 1):
                    sld.setMaximum(max(0, self.core.total_frames - 1))
                sld.setValue(frame_idx)
            finally:
                sld.blockSignals(False)

        self._on_frame_changed(frame_idx)

        if gest:
            self.editor_panel.show_selection_badge(f"Selected: Person {pid} • {gest}")
            self.statusBar().showMessage(f"Selected Person {pid} • {gest} @ frame {frame_idx}")

    # ---------- misc ----------
    def _open_documentation(self):
        webbrowser.open("https://example.com/stopfilming/docs")

    def _check_for_updates(self):
        QMessageBox.information(self, "Check for Updates", "No updates available.")

    def _toggle_appearance(self):
        QMessageBox.information(self, "Appearance", "Toggle light/dark (not implemented).")

    def _show_shortcuts_reference(self):
        dlg = KeyboardShortcutsDialog(self)
        dlg.exec_()

    def _show_about_dialog(self):
        QMessageBox.information(self, "About StopFilming", "StopFilming v1.0\n© 2025")

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _show_video_settings_dialog(self):
        QMessageBox.information(self, "Video Settings", "Video-settings are not implemented yet.")

    def _show_detection_settings_dialog(self):
        QMessageBox.information(self, "Detection Settings", "Detection settings are not implemented yet.")

    def _show_blur_settings_dialog(self):
        QMessageBox.information(self, "Blur Settings", "Blur settings are not implemented yet.")


def create_eye_icon():
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor(70, 130, 180), 2))
    painter.setBrush(QBrush(QColor(100, 149, 237)))
    painter.drawEllipse(2, 10, 28, 12)
    painter.setPen(QPen(QColor(25, 25, 112), 2))
    painter.setBrush(QBrush(QColor(25, 25, 112)))
    painter.drawEllipse(13, 13, 6, 6)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(QColor(255, 255, 255, 180)))
    painter.drawEllipse(14, 14, 2, 2)
    painter.end()
    return QIcon(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    QApplication.setApplicationName("StopFilming")
    QApplication.setApplicationDisplayName("StopFilming – Privacy Protection Video Editor")
    QApplication.setApplicationVersion("1.0")
    app.setWindowIcon(create_eye_icon())

    window = MainWindow()
    screen = app.primaryScreen()
    rect = screen.availableGeometry()
    window.resize(rect.width(), rect.height())
    window.showMaximized()
    sys.exit(app.exec_())