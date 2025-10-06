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

        actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        
        # Debug prints to check cache during playback
        print(f"Current frame: {actual_pos}")
        print(f"Is frame in cache? {actual_pos in self.core.blurred_cache}")
        print(f"Total cached frames: {len(self.core.blurred_cache)}")
        
        if actual_pos in self.core.blurred_cache:
            print(f"Using blurred frame {actual_pos}")
            display_frame = self.core.blurred_cache[actual_pos]
        else:
            print(f"Using original frame {actual_pos}")
            display_frame = self._apply_rotation(frame)

        # Display whichever frame we're using
        self.editor_panel.display_frame(display_frame, actual_pos)
        self.editor_panel.current_frame_idx = actual_pos


    def _on_detect_requested(self):
        total = self.core.total_frames
        if total <= 0:
            return

        if hasattr(self.editor_panel, "get_detection_params"):
            det_params = self.editor_panel.get_detection_params()
            if hasattr(self.core, "set_detection_params"):
                self.core.set_detection_params(
                    confidence=det_params["confidence"],
                    frame_skip=det_params["frame_skip"],
                )
            else:
                setattr(self.core, "confidence", det_params["confidence"])
                setattr(self.core, "frame_skip", det_params["frame_skip"])

        self.proc = ProcessingDialog(
            self, title="Processing – StopFilming",
            message="Detecting gestures… Please wait…",
            total_steps=None
        )
        self.proc.show()

        self.detect_thread = GestureDetectWorker(self.core)
        self.detect_thread.finished.connect(self._on_gesture_detection_finished)
        self.detect_thread.start()

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

    def _on_blur_requested(self, _frame_idx_from_button: int):
        selected = self.editor_panel.gesture_list.selectedItems()
        if not selected:
            return
        payload = selected[0].data(Qt.UserRole)
        if not payload:
            return

        frame_idx = int(payload["frame"])
        sel_pid = payload.get("person_id", "?")

        self.editor_panel.show_selection_badge("")
        self.editor_panel.start_blur_progress()

        def _blur_progress_cb(pct: int):
            self.editor_panel.set_blur_progress(int(pct))

        ok = False
        try:
            # Initialize video properties
            cap = cv2.VideoCapture(self.core.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            person_tracker = PersonTracker(max_disappeared=60, feature_threshold=0.7, motion_threshold=200)
            
            # Initialize tracking with first detection
            initial_frame = self.core.get_frame(frame_idx)
            results = yolo_model(initial_frame)[0]
            boxes = results.boxes
            
            if len(boxes):
                # Convert YOLO detections to integer coordinates
                detections = []
                for box in boxes:
                    # Convert tensor to numpy and get coordinates
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, coords)
                    confidence = float(box.conf[0].cpu().numpy())
                    detections.append(([x1, y1, x2, y2], confidence))
                
                initial_people = person_tracker.update(initial_frame, detections, frame_idx)
                if sel_pid not in initial_people:
                    print(f"Warning: Could not find person {sel_pid} in initial frame")
                    return

            print(f"Starting tracking and blurring for person {sel_pid}")
            # Track and blur through video
            for current_frame in range(0, total_frames, 5):
                progress = (current_frame / total_frames) * 100
                _blur_progress_cb(progress)
                
                frame = self.core.get_frame(current_frame)
                if frame is None:
                    continue

                results = yolo_model(frame)[0]
                boxes = results.boxes
                
                if len(boxes):
                    # Convert YOLO detections to integer coordinates
                    detections = []
                    for box in boxes:
                        coords = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, coords)
                        confidence = float(box.conf[0].cpu().numpy())
                        detections.append(([x1, y1, x2, y2], confidence))
                    
                    current_people = person_tracker.update(frame, detections, current_frame)
                    
                    if sel_pid in current_people:
                        print(f"Found person {sel_pid} in frame {current_frame}")
                        blurred_frame = frame.copy()
                        bbox = [int(x) for x in current_people[sel_pid]['bbox']]
                        blurred_frame = blur_faces_of_person(blurred_frame, bbox)
                        self.core.blurred_cache[current_frame] = blurred_frame
                        print(f"Added blurred frame {current_frame} to cache")

            print(f"Final cache size: {len(self.core.blurred_cache)} frames")
            print(f"First 10 cached frames: {sorted(list(self.core.blurred_cache.keys()))[:10]}")
            ok = True

        except Exception as e:
            print(f"Error during blurring: {e}")
        finally:
            self.editor_panel.finish_blur_progress(bool(ok))
            self.editor_panel.show_selection_badge("")
            self.editor_panel.update()

        # Refresh display
        img = self.core.get_frame(frame_idx)
        if frame_idx in self.core.blurred_cache:
            img = self.core.blurred_cache[frame_idx]
        self.editor_panel.display_frame(img, frame_idx)
        self.editor_panel.current_frame_idx = frame_idx
            
            # Force a refresh of the display
        self.editor_panel.update()

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