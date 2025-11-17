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

import numpy as np
import sys
import webbrowser
import time
import cv2
from ultralytics import YOLO
from PyQt5.QtCore import QUrl
from moviepy import VideoFileClip

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QFileDialog, QMessageBox, QDialog,
    QListWidgetItem
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QElapsedTimer, QSettings
from PyQt5.QtGui import QPixmap, QIcon, QPainter, QColor, QPen, QBrush

from view import KeyboardShortcutsDialog
from view import EditorPanel, ProcessingDialog, ExportDialog
from model import EditorCore
import tempfile, shutil, gc

from utilities import (
    PersonTracker,
    detect_multiple_people_yolov8,
    detect_gesture_in_person_box,
    blur_faces_of_person,
    face_bbox_in_person,
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


# ---------------- Workers ----------------
class BlurPersonWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, core, sel_pid, blur_type, blur_strength):
        super().__init__()
        self.core = core
        self.sel_pid = sel_pid
        self.blur_type = blur_type
        self.blur_strength = blur_strength

    def run(self):
        import cv2

        cap = cv2.VideoCapture(self.core.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rotation = getattr(self.core, "rotation_angle", 0)

        # reference bbox for the selected person (from the detection pass)
        ref_bbox = None
        for person in getattr(self.core, "detected_people", []):
            if person.get("person_id") == self.sel_pid:
                ref_bbox = person.get("bbox")
                break

        def iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
            inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
            inter = iw * ih
            if inter == 0:
                return 0.0
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            return inter / float(area_a + area_b - inter + 1e-6)

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # apply rotation to match the rest of the app
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # detect people on this frame
            dets = detect_multiple_people_yolov8(frame, conf_threshold=0.5)

            # robustly pull bboxes whether det is (bbox, mask) or bbox-only
            bboxes = []
            for d in dets:
                if isinstance(d, (tuple, list)) and len(d) >= 1 and isinstance(d[0], (tuple, list)):
                    bb = d[0]
                else:
                    bb = d
                bboxes.append(tuple(map(int, bb)))

            blur_bbox = None
            if ref_bbox and bboxes:
                blur_bbox = max(bboxes, key=lambda bb: iou(ref_bbox, bb))
                if iou(ref_bbox, blur_bbox) < 0.05:
                    blur_bbox = None

            if blur_bbox is not None:
                # use cached blurred frame as base if exists, else original
                base = self.core.blurred_cache.get(frame_idx, frame)

                # compute a face bbox inside the person bbox for more precise blur
                try:
                    face_bb = face_bbox_in_person(frame, blur_bbox)
                except Exception:
                    face_bb = None
                target_bb = face_bb if face_bb else blur_bbox

                out = blur_faces_of_person(
                    base,
                    target_bb,
                    blur_type=self.blur_type,
                    strength=self.blur_strength,
                )

                self.core.blurred_cache[frame_idx] = out
                self.core.blurred_frames.add(frame_idx)

            # progress
            self.progress.emit(int(frame_idx / max(1, total_frames) * 100))

        cap.release()
        self.finished.emit()


class GestureDetectWorker(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)

    def __init__(self, core):
        super().__init__()
        self.core = core

    def run(self):
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

                person_data = person_tracker.tracked_people.get(person_id)
                if not person_data:
                    continue

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
    progress = pyqtSignal(int)
    done = pyqtSignal(bool, str)

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

        # Memory monitor
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self._check_memory)
        self.memory_timer.start(5000)

        self._mem_warned_once = False
        self._mem_next_warn_ts = 0.0
        self._settings = QSettings("StopFilming", "StopFilmingApp")
        self._mem_warn_suppressed = bool(self._settings.value("memory/warnings_suppressed", False, type=bool))

        # Menu bar styling
        menubar = self.menuBar()
        self.menuBar().setStyleSheet("""
            QMenuBar { background-color: #2D3748; color: #E2E8F0; spacing: 6px; padding: 2px 10px; }
            QMenuBar::item { background: transparent; padding: 4px 12px; }
            QMenuBar::item:selected { background-color: #4FD1C7; color: #1A202C; border-radius: 4px; }
            QMenu { background-color: #2D3748; color: #E2E8F0; border: 1px solid #4A5568; margin: 2px; }
            QMenu::item { padding: 6px 20px; }
            QMenu::item:selected { background-color: #4A5568; }
        """)

        # File menu
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

        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(lambda: None)
        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(lambda: None)
        edit_menu.addAction(undo_action)
        edit_menu.addAction(redo_action)
        edit_menu.addSeparator()

        # View menu
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

        # Memory warnings toggle
        self.warn_mem_action = QAction("Warn about Memory Usage", self, checkable=True)
        self.warn_mem_action.setChecked(not self._mem_warn_suppressed)
        self.warn_mem_action.toggled.connect(self._toggle_memory_warnings)
        view_menu.addAction(self.warn_mem_action)

        # Blur settings (defaults)
        self.blur_type = "gaussian"
        self.blur_strength = 50
        self._wire_blur_controls_from_panel()

        # Fullscreen
        fullscreen_action = QAction("Fullscreen", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        detect_action = QAction("Detect Gestures", self)
        detect_action.triggered.connect(self.editor_panel.detectRequested.emit)
        tools_menu.addAction(detect_action)

        blur_action = QAction("Blur Current Frame", self)
        blur_action.triggered.connect(
            lambda: self.editor_panel.blurRequested.emit(self.editor_panel.current_frame_idx)
        )
        tools_menu.addAction(blur_action)

        export_action = QAction("Export Blurred Video...", self)
        export_action.triggered.connect(self._on_save_project)
        tools_menu.addAction(export_action)

        clear_blurs_action = QAction("Clear All Blurs", self)
        clear_blurs_action.triggered.connect(self._on_clear_blurs)
        tools_menu.addAction(clear_blurs_action)

        # Help menu
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About StopFilming", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.triggered.connect(self._show_shortcuts_reference)
        help_menu.addAction(shortcuts_action)

        self._last_blur_profile = (self.blur_type, int(self.blur_strength))

    # ------------------------------------------------------------------
    # Blur settings handlers – keep UI and state in sync
    # ------------------------------------------------------------------
    def _on_blur_type_changed(self, t: str):
        self.blur_type = t
        p = getattr(self, "editor_panel", None)
        if p:
            if t == "gaussian" and hasattr(p, "rb_gauss"):
                p.rb_gauss.setChecked(True)
            elif t == "pixelate" and hasattr(p, "rb_pixel"):
                p.rb_pixel.setChecked(True)
            elif t == "solid" and hasattr(p, "rb_solid"):
                p.rb_solid.setChecked(True)

    def _on_blur_strength_changed(self, v: int):
        self.blur_strength = int(v)
        p = getattr(self, "editor_panel", None)
        if p and hasattr(p, "lbl_strength_pct"):
            p.lbl_strength_pct.setText(f"{int(v)}%")

    def _wire_blur_controls_from_panel(self):
        p = self.editor_panel

        # guard if the panel changed / not built yet
        for attr in ("rb_gauss", "rb_pixel", "rb_solid", "blur_strength", "lbl_strength_pct"):
            if not hasattr(p, attr):
                print("Blur controls not found on panel; skipping wiring.")
                return

        # initial sync from MainWindow -> UI
        if self.blur_type == "gaussian":
            p.rb_gauss.setChecked(True)
        elif self.blur_type == "pixelate":
            p.rb_pixel.setChecked(True)
        else:
            p.rb_solid.setChecked(True)

        p.blur_strength.setRange(0, 100)
        p.blur_strength.setValue(int(self.blur_strength))
        if hasattr(p, "lbl_strength_pct"):
            p.lbl_strength_pct.setText(f"{int(self.blur_strength)}%")

        # connect UI -> MainWindow state
        p.rb_gauss.toggled.connect(lambda c: c and self._on_blur_type_changed("gaussian"))
        p.rb_pixel.toggled.connect(lambda c: c and self._on_blur_type_changed("pixelate"))
        p.rb_solid.toggled.connect(lambda c: c and self._on_blur_type_changed("solid"))
        p.blur_strength.valueChanged.connect(self._on_blur_strength_changed)

    # ------------------------------------------------------------------
    # Memory warnings + misc small helpers
    # ------------------------------------------------------------------
    def _toggle_memory_warnings(self, enabled: bool):
        self._mem_warn_suppressed = not enabled
        self._settings.setValue("memory/warnings_suppressed", self._mem_warn_suppressed)

    def _cancelled(self) -> bool:
        return bool(getattr(self, "proc", None) and self.proc.was_cancelled())

    # ---------- Menus and Controller slots ----------
    def _check_memory(self):
        try:
            import psutil, gc, time
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024

            if memory_mb > 2048:
                if hasattr(self.core, '_frame_cache'):
                    self.core._frame_cache.clear()
                if hasattr(self.core, 'blurred_cache'):
                    cur = getattr(self.editor_panel, "current_frame_idx", 0)
                    for k in list(self.core.blurred_cache.keys()):
                        if abs(k - cur) > 8:
                            del self.core.blurred_cache[k]
                gc.collect()

            if memory_mb > 4096 and not self._mem_warn_suppressed:
                now = time.time()
                if (not self._mem_warned_once) or (now >= self._mem_next_warn_ts):
                    self._toast(
                        f"High memory usage detected ({memory_mb:.0f}MB). Continuing… freed caches.",
                        ms=2500
                    )
                    self._mem_warned_once = True
                    self._mem_next_warn_ts = now + 120

        except ImportError:
            pass

    def _on_import_requested(self):
        vid_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Video Files (*.mp4 *.MP4 *.avi *.AVI *.mov *.MOV);;All Files (*)"
        )
        if not vid_path:
            return

        self.play_timer.stop()
        try:
            self.editor_panel.audio_pause()
        except Exception:
            pass

        if getattr(self.core, "cap", None) is not None:
            try:
                self.core.cap.release()
            except Exception:
                pass
            self.core.cap = None

        meta = self.core.load_video(vid_path)

        # MoviePy rotation metadata
        self._moviepy_rotation = self._get_rotation_from_metadata(vid_path)
        self.core.rotation_angle = self._moviepy_rotation
        print(f"[IMPORT] MoviePy rotation metadata: {self._moviepy_rotation}°")

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

        self._wire_blur_controls_from_panel()

        self.editor_panel.set_video_info(
            rotation_angle=meta["rotation_angle"],
            total_frames=meta["total_frames"],
            fps=meta["fps"],
        )
        self.editor_panel.video_path = self.core.video_path

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

    def _toast(self, text: str, title: str = "StopFilming", ms: int = 1500):
        dlg = ProcessingDialog(self, title=title, message=text, total_steps=None)
        dlg.setWindowModality(Qt.NonModal)
        dlg.setWindowFlags(dlg.windowFlags() | Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)

        parent_geo = self.geometry()
        dlg.adjustSize()
        x = parent_geo.x() + parent_geo.width() - dlg.width() - 24
        y = parent_geo.y() + parent_geo.height() - dlg.height() - 24
        dlg.move(max(0, x), max(0, y))

        dlg.show()
        QTimer.singleShot(ms, dlg.accept)

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
        codec = (cfg.get("codec") or "mp4v").lower()

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

        self.proc = ProcessingDialog(
            self,
            title="Exporting Video",
            message="Writing output file...",
            total_steps=100,
            allow_cancel=False,
            show_cancel_button=False
        )
        self.proc.setWindowModality(Qt.NonModal)
        self.proc.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.proc.show()
        QApplication.processEvents()

        self.export_thread = ExportWorker(self.core, out_path)
        self.export_thread.progress.connect(
            lambda pct: (getattr(self, "proc", None) and
                         self.proc.set_progress(
                             int(max(0, min(100, pct))),
                             label=f"{int(pct)}%"
                         ))
        )

        def _done(ok, path):
            if getattr(self, "proc", None):
                try:
                    self.proc.finish("Export complete" if ok else "Export failed")
                except Exception:
                    try:
                        self.proc.accept()
                    except Exception:
                        pass
                self.proc = None

            if ok:
                self._toast(f"Exported to:\n{path}", title="Export Complete", ms=3000)
                self.statusBar().showMessage(f"Saved: {path}", 7000)
            else:
                self._toast("Failed to export edited video.", title="Export Failed", ms=3500)
                self.statusBar().showMessage("Export failed", 7000)

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
        self.editor_panel.clear_markers()
        self.editor_panel.blur_button.setEnabled(False)

    def _on_play_toggled(self, play: bool):
        if not self.core or not self.core.video_path:
            return

        if play:
            if self.core.cap is None or not self.core.cap.isOpened():
                self.core.cap = cv2.VideoCapture(self.core.video_path, cv2.CAP_FFMPEG)
                try:
                    self.core.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

            start_at = max(0, int(self.editor_panel.current_frame_idx))
            self.core.cap.set(cv2.CAP_PROP_POS_FRAMES, start_at)
            self.play_start_frame = start_at
            self.play_clock.start()

            fps = max(1.0, float(self.core.fps))
            timer_interval = max(8, int(1000.0 / fps / 1.2))
            self.play_timer.start(timer_interval)
            self.editor_panel.toggle_button.setText("Pause")
            self.editor_panel.audio_play_from_frame(self.editor_panel.current_frame_idx, fps)
        else:
            self.play_timer.stop()
            self.editor_panel.toggle_button.setText("Play")
            self.editor_panel.audio_pause()

    def _get_rotation_from_metadata(self, path):
        try:
            clip = VideoFileClip(path)
            rotation = int(getattr(clip, "rotation", 0) or 0)
            clip.close()
            if rotation not in (0, 90, 180, 270):
                rotation = 0
        except Exception:
            rotation = 0
        return rotation

    def _apply_rotation(self, frame):
        if frame is None:
            return None
        rotation = getattr(self, "_moviepy_rotation", 0)
        if rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _on_frame_changed(self, frame_idx: int):
        try:
            self.play_timer.stop()
            self.editor_panel.toggle_button.setText("Play")

            if self.core.cap is not None:
                self.core.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            if frame_idx in self.core.blurred_cache:
                img = self.core.blurred_cache[frame_idx].copy()
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

        if hasattr(self.core, "blur_cache_dir") and os.path.exists(self.core.blur_cache_dir):
            frame_path = os.path.join(self.core.blur_cache_dir, f"{frame_idx:06d}.jpg")
            if frame_idx in self.core.blurred_cache:
                display_frame = self.core.blurred_cache[frame_idx]
            elif os.path.exists(frame_path):
                blurred = cv2.imread(frame_path)
                self.core.blurred_cache[frame_idx] = blurred
                display_frame = blurred

            keys = sorted(self.core.blurred_cache.keys())
            for k in keys:
                if abs(k - frame_idx) > 15:
                    del self.core.blurred_cache[k]

        self.editor_panel.display_frame(display_frame, frame_idx)
        self.editor_panel.current_frame_idx = frame_idx

    def _on_detect_requested(self):
        self.editor_panel.start_detect_progress(total_steps=100)

        if not self.core.video_path:
            QMessageBox.information(self, "Detection", "No video loaded.")
            return

        video_path = self.core.video_path
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            clip = VideoFileClip(video_path)
            rotation = int(getattr(clip, "rotation", 0) or 0)
            clip.close()
            self._moviepy_rotation = rotation
            print(f"[DEBUG] MoviePy rotation metadata: {self._moviepy_rotation}°")
        except Exception:
            rotation = getattr(self.core, "rotation_angle", 0) or 0
            self._moviepy_rotation = rotation
            print(f"[DEBUG] MoviePy rotation metadata: {self._moviepy_rotation}°")
        cap.release()

        print("=" * 70)
        print("PASS 1: OPTIMIZED GESTURE DETECTION (Improved Skip Logic)")
        print("=" * 70)
        print(f"Video: {video_path}")
        print(f"Total Frames: {total_frames}, FPS: {fps:.2f}, Rotation: {rotation}°")

        person_tracker = PersonTracker(max_disappeared=30, feature_threshold=0.3, motion_threshold=200)
        discovered_people = []
        people_to_blur = []
        gestures_to_check = ["wave", "hand_over_face"]

        # STEP 1: Discover people
        print("\n📋 STEP 1: Discovering people...")
        cap = cv2.VideoCapture(video_path)
        for frame_idx in range(0, total_frames, DISCOVERY_FRAME_SKIP):
            dlg = getattr(self.editor_panel, "_dlg_detect", None)
            if dlg and dlg.was_cancelled():
                self.editor_panel.finish_detect_progress(False)
                return

            ret, frame = cap.read()
            if not ret:
                break

            frame = self._apply_rotation(frame)
            people_detected = detect_multiple_people_yolov8(frame, conf_threshold=0.5)
            if people_detected:
                current_people = person_tracker.update(frame, people_detected, frame_idx)
                for pid in current_people.keys():
                    if pid not in discovered_people:
                        discovered_people.append(pid)
                        print(f"  👤 New person detected: ID {pid} @ frame {frame_idx}")

            progress = int((frame_idx / total_frames) * 40)
            if not self.editor_panel.set_detect_progress(progress, f"Stage 1 – Discovering People: {progress}%"):
                self.editor_panel.finish_detect_progress(False)
                return

            self.statusBar().showMessage(f"Discovering people... {progress}%")
            QApplication.processEvents()

        cap.release()
        print(f"\n✅ STEP 1 COMPLETE: {len(discovered_people)} people discovered.")

        # STEP 2: Analyze gestures
        print("\n🔍 STEP 2: Analyzing gestures (adaptive clarity filter)...")
        gesture_found_for_person = {pid: False for pid in discovered_people}
        unclear_face_for_person = {pid: False for pid in discovered_people}

        total_to_analyze = len(discovered_people)
        analyzed_count = 0

        print("\n📊 Estimating average video clarity...")
        cap = cv2.VideoCapture(video_path)
        sharp_samples, bright_samples = [], []
        for i in range(0, min(total_frames, 300), max(1, total_frames // 50)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp_samples.append(cv2.Laplacian(gray, cv2.CV_64F).var())
            bright_samples.append(gray.mean())
        cap.release()

        avg_sharp = np.mean(sharp_samples) if sharp_samples else 50
        avg_bright = np.mean(bright_samples) if bright_samples else 100
        print(f"📈 Avg sharpness: {avg_sharp:.1f}, Avg brightness: {avg_bright:.1f}")

        sharp_thresh = max(10, avg_sharp * 0.4)
        bright_thresh = max(25, avg_bright * 0.5)
        area_thresh = 5000
        print(f"🔧 Using thresholds → sharpness<{sharp_thresh:.1f}, brightness<{bright_thresh:.1f}, area<{area_thresh}")

        for gesture_type in gestures_to_check:
            print(f"\n🎯 Checking gesture type: {gesture_type}")

            for pid in discovered_people:
                if gesture_found_for_person[pid]:
                    print(f"⏭️ Skipping Person {pid} (already has gesture)")
                    continue
                if unclear_face_for_person[pid]:
                    print(f"🚫 Skipping Person {pid} (face unclear or not visible)")
                    continue

                print(f"  ➤ Analyzing Person {pid} for {gesture_type}...")
                gesture_detected = False
                cap = cv2.VideoCapture(video_path)

                for frame_num in range(0, total_frames, ANALYSIS_FRAME_SKIP):
                    dlg = getattr(self.editor_panel, "_dlg_detect", None)
                    if dlg and dlg.was_cancelled():
                        self.editor_panel.finish_detect_progress(False)
                        return

                    if gesture_found_for_person[pid]:
                        break

                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = self._apply_rotation(frame)
                    people_detected = detect_multiple_people_yolov8(frame, conf_threshold=0.5)
                    if not people_detected:
                        continue

                    current_people = person_tracker.update(frame, people_detected, frame_num)
                    if pid in current_people:
                        person_data = current_people[pid]
                        bbox = person_data["bbox"]
                        x1, y1, x2, y2 = map(int, bbox)

                        roi = frame[y1:y2, x1:x2]
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                        brightness = gray.mean()
                        area = (x2 - x1) * (y2 - y1)

                        if sharpness < sharp_thresh or brightness < bright_thresh or area < area_thresh:
                            unclear_face_for_person[pid] = True
                            print(
                                f"🚫 Person {pid} face unclear "
                                f"(sharp={sharpness:.1f}/{sharp_thresh:.1f}, "
                                f"bright={brightness:.1f}/{bright_thresh:.1f}) – skipping."
                            )
                            break

                        detected = detect_gesture_in_person_box(
                            bbox, cap, gesture_type, fps, duration_seconds=GESTURE_DURATION
                        )
                        if detected:
                            gesture_detected = True
                            gesture_found_for_person[pid] = True
                            print(f"    ✅ {gesture_type} detected for Person {pid}")
                            people_to_blur.append({
                                "person_id": pid,
                                "gesture": gesture_type,
                                "frame": frame_num,
                                "bbox": bbox
                            })
                            break

                cap.release()
                analyzed_count += 1
                progress = max(0, min(100, 40 + int((analyzed_count / total_to_analyze) * 60)))
                if not self.editor_panel.set_detect_progress(
                    progress,
                    f"Stage 2 – Detecting Gestures: {progress}%"
                ):
                    self.editor_panel.finish_detect_progress(False)
                    return

                self.statusBar().showMessage(f"Analyzing gestures... {progress}%")
                QApplication.processEvents()

                all_done = all(
                    gesture_found_for_person[pid] or unclear_face_for_person[pid]
                    for pid in discovered_people
                )
                if all_done:
                    print("🎉 All people processed (gesture or skipped). Stopping early!")
                    break

            if all(
                gesture_found_for_person[pid] or unclear_face_for_person[pid]
                for pid in discovered_people
            ):
                break

        unique_people = {p["person_id"]: p for p in people_to_blur}
        people_to_blur = list(unique_people.values())

        print("\n🎉 DETECTION COMPLETE:")
        print(f"- Total people with gestures: {len(people_to_blur)}")
        print(f"- IDs: {[p['person_id'] for p in people_to_blur]}")

        self.editor_panel.gesture_list.clear()
        for p in people_to_blur:
            item = QListWidgetItem(f"Person {p['person_id']} - {p['gesture'].title()}")
            item.setData(Qt.UserRole, p)
            self.editor_panel.gesture_list.addItem(item)

        has_items = self.editor_panel.gesture_list.count() > 0
        self.editor_panel.blur_button.setEnabled(has_items)
        self.statusBar().showMessage(f"Detected {len(people_to_blur)} gesture(s)")

        self.people_to_blur = people_to_blur
        self.person_tracker = person_tracker
        print("=" * 70)
        print("PASS 1 COMPLETE — Ready for blurring.")
        print("=" * 70)
        self.editor_panel.finish_detect_progress(True)

    def _on_blur_requested(self, _frame_idx_from_button: int):
        """Blur the selected person and keep all previously blurred people persistent."""
        import tempfile, gc, cv2, os

        if not hasattr(self, "people_to_blur") or not self.people_to_blur:
            self._toast("No detected gestures found.", title="Blur – StopFilming")
            return

        selected_items = self.editor_panel.gesture_list.selectedItems()
        if not selected_items:
            self._toast("Select one or more people from the list.", title="Blur – StopFilming")
            return

        selected_pairs = []
        for it in selected_items:
            data = it.data(Qt.UserRole) or {}
            if "person_id" in data:
                pid = int(data["person_id"])
                bb = tuple(map(int, data.get("bbox", (0, 0, 0, 0))))
                selected_pairs.append((pid, bb))

        selected_ids = [pid for pid, _ in selected_pairs]
        print(f"🎯 Selected people for blurring: IDs {selected_ids}")

        prev_ids = set(getattr(self, "persistent_blur_ids", set()))

        if not hasattr(self, "persistent_blur_ids"):
            self.persistent_blur_ids = set()
        self.persistent_blur_ids.update(selected_ids)

        new_ids = set(self.persistent_blur_ids) - prev_ids

        print(f"Prev IDs: {sorted(prev_ids)}")
        print(f"Selected IDs this pass: {sorted(selected_ids)}")
        print(f"New IDs to blur this pass: {sorted(new_ids)}")

        # Decide whether to re-render from clean frames or build on cached frames,
        # based on blur profile changing or overlapping IDs
        last_profile = getattr(self, "_last_blur_profile", None)
        current_profile = (self.blur_type, int(self.blur_strength))

        changed_profile = (last_profile is not None) and (current_profile != last_profile)
        has_existing_overlap = len(set(selected_ids) & prev_ids) > 0

        re_render_all = changed_profile or has_existing_overlap
        ids_to_apply = set(self.persistent_blur_ids) if re_render_all else set(new_ids)

        print(f"🧠 Re-render all from clean frames? {re_render_all}")
        print(f"IDs that will be (re)applied this pass: {sorted(ids_to_apply)}")

        import shutil
        cancelled = False
        staged_cache = {}
        if not hasattr(self.core, "blur_cache_dir"):
            self.core.blur_cache_dir = tempfile.mkdtemp(prefix="blur_cache_")
        cache_dir = self.core.blur_cache_dir
        run_dir = os.path.join(cache_dir, f"run_{int(time.time())}")
        os.makedirs(run_dir, exist_ok=True)

        def _iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
            inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            return inter / float(area_a + area_b - inter + 1e-6)

        if not hasattr(self, "persistent_blur_ids"):
            self.persistent_blur_ids = set()
        for pid in selected_ids:
            self.persistent_blur_ids.add(pid)
        print(f"🧩 Current persistent blur IDs: {sorted(self.persistent_blur_ids)}")

        video_path = self.core.video_path
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rotation = getattr(self.core, "rotation_angle", 0)
        fps = cap.get(cv2.CAP_PROP_FPS)

        self.editor_panel.start_blur_progress()
        QApplication.processEvents()

        print(f"📁 Using cache directory: {cache_dir}")
        print(f"Blurring all persistent Person IDs (base): {self.persistent_blur_ids}")
        print(f"Re-render all from clean frames? {re_render_all}")

        for pid in self.person_tracker.tracked_people:
            self.person_tracker.tracked_people[pid]["disappeared"] = 0

        detection_interval = YOLO_DETECTION_INTERVAL
        last_tracked_people = {}
        frame_idx = 0

        if not hasattr(self, "ref_bbox_for"):
            self.ref_bbox_for = {}

        seed_map = {pid: bb for pid, bb in selected_pairs}
        for pid in self.persistent_blur_ids:
            if pid not in self.ref_bbox_for or self.ref_bbox_for[pid] == (0, 0, 0, 0):
                pdata = self.person_tracker.tracked_people.get(pid, {})
                self.ref_bbox_for[pid] = seed_map.get(
                    pid,
                    tuple(map(int, pdata.get("bbox", (0, 0, 0, 0))))
                )

        def _best_match_bbox(ref_bb, current_people_dict):
            if ref_bb == (0, 0, 0, 0) or not current_people_dict:
                return None, None, 0.0
            best_pid, best_bb, best_iou = None, None, 0.0
            for pid, pdata in current_people_dict.items():
                bb = tuple(map(int, pdata["bbox"]))
                i = _iou(ref_bb, bb)
                if i > best_iou:
                    best_pid, best_bb, best_iou = pid, bb, i
            return best_pid, best_bb, best_iou

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = self._apply_rotation(frame)

            if frame_idx % detection_interval == 0:
                people_detected = detect_multiple_people_yolov8(frame, conf_threshold=0.5)
                if people_detected:
                    current_people = self.person_tracker.update(frame, people_detected, frame_idx)
                    last_tracked_people = current_people.copy()
                else:
                    current_people = last_tracked_people.copy()
            else:
                current_people = last_tracked_people.copy()

            frame_path = os.path.join(cache_dir, f"{frame_idx:06d}.jpg")

            if re_render_all:
                base = frame.copy()
            else:
                if os.path.exists(frame_path):
                    base = cv2.imread(frame_path)
                else:
                    base = frame.copy()

            blurred_frame = base

            for pid in list(ids_to_apply):
                ref_bb = self.ref_bbox_for.get(pid, (0, 0, 0, 0))
                match_pid, match_bb, match_iou = _best_match_bbox(ref_bb, current_people)

                if (match_bb is None or match_iou < 0.02) and current_people and ref_bb != (0, 0, 0, 0):
                    (rx1, ry1, rx2, ry2) = ref_bb
                    rcx, rcy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0

                    def _center(b):
                        x1, y1, x2, y2 = b
                        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                    best = None
                    best_d = 1e18
                    for _, pdata in current_people.items():
                        b = tuple(map(int, pdata["bbox"]))
                        cx, cy = _center(b)
                        d = (cx - rcx) ** 2 + (cy - rcy) ** 2
                        if d < best_d:
                            best, best_d = b, d
                    match_bb = best

                if match_bb is not None:
                    # use face_bbox_in_person for tighter blur region
                    try:
                        face_bb = face_bbox_in_person(frame, match_bb)
                    except Exception:
                        face_bb = None

                    target_bb = face_bb if face_bb else match_bb

                    blurred_frame = blur_faces_of_person(
                        blurred_frame,
                        target_bb,
                        blur_type=self.blur_type,
                        strength=int(self.blur_strength),
                    )

                    self.ref_bbox_for[pid] = match_bb

            frame_path_stage = os.path.join(run_dir, f"{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path_stage, blurred_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            staged_cache[frame_idx] = blurred_frame

            self.editor_panel.set_blur_progress(int((frame_idx / total_frames) * 100))

            dlg = getattr(self.editor_panel, "_dlg_blur", None)
            if dlg is not None and dlg.was_cancelled():
                cancelled = True
                break

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

        if cancelled:
            self.persistent_blur_ids = set(prev_ids)
            try:
                shutil.rmtree(run_dir, ignore_errors=True)
            except Exception:
                pass
            self.editor_panel.finish_blur_progress(False)
            self.statusBar().showMessage("Blurring cancelled")
            self._toast("Blur cancelled. No changes applied.", title="Blur – StopFilming")
            return
        else:
            for name in os.listdir(run_dir):
                src = os.path.join(run_dir, name)
                dst = os.path.join(cache_dir, name)
                try:
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.replace(src, dst)
                except Exception:
                    pass
            try:
                os.rmdir(run_dir)
            except Exception:
                pass

        print(f"\n🎉 COMPLETE! Persistent blur IDs now: {sorted(self.persistent_blur_ids)}")
        print(f"Frames stored in cache: {cache_dir}")

        self.editor_panel.finish_blur_progress(True)
        self.editor_panel.update()
        self.statusBar().showMessage(
            f"Blurring updated for {len(self.persistent_blur_ids)} person(s)"
        )

        try:
            sel_str = ", ".join(map(str, selected_ids))
        except NameError:
            sel_str = "selected people"

        self._toast(
            f"Blurring complete for {sel_str}. "
            f"Now blurring {len(self.persistent_blur_ids)} person(s) in total.",
            title="Blur Updated – StopFilming"
        )

        print(f"DEBUG: blurred_frames has {len(self.core.blurred_frames)} frames")
        print(f"DEBUG: blurred_cache has {len(self.core.blurred_cache)} frames")
        print(f"DEBUG: Sample frame indices: {list(self.core.blurred_frames)[:10]}")

        self._last_blur_profile = (self.blur_type, int(self.blur_strength))

    def _on_gesture_detection_finished(self, people_with_gestures):
        if hasattr(self, "proc"):
            self.proc = None

        segment_starts = [
            (
                person_data['person_id'],
                "wave",
                person_data['frame'],
                person_data['bbox']
            )
            for person_data in people_with_gestures
        ]

        segment_starts = sorted(segment_starts, key=lambda s: s[2])

        self.editor_panel.gesture_list.clear()

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
        img = self._apply_rotation(img)
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

        if bbox:
            if hasattr(self.core, 'rotation_angle') and self.core.rotation_angle != 0:
                frame = self.core.get_frame(frame_idx)
                if frame is not None:
                    h, w = frame.shape[:2]
                    bbox = self._rotate_bbox(bbox, w, h, self.core.rotation_angle)

            self.editor_panel.highlight_person_on_frame(bbox)

        if gest:
            self.editor_panel.show_selection_badge(f"Selected: Person {pid} • {gest}")
            self.statusBar().showMessage(f"Selected Person {pid} • {gest} @ frame {frame_idx}")

    def _rotate_bbox(self, bbox, width, height, rotation):
        x1, y1, x2, y2 = bbox

        if rotation == 90:
            new_x1 = y1
            new_y1 = width - x2
            new_x2 = y2
            new_y2 = width - x1
            return (new_x1, new_y1, new_x2, new_y2)
        elif rotation == 180:
            new_x1 = width - x2
            new_y1 = height - y2
            new_x2 = width - x1
            new_y2 = height - y1
            return (new_x1, new_y1, new_x2, new_y2)
        elif rotation == 270:
            new_x1 = height - y2
            new_y1 = x1
            new_x2 = height - y1
            new_y2 = x2
            return (new_x1, new_y1, new_x2, new_y2)

        return bbox

    # ---------- misc ----------
    def _open_documentation(self):
        webbrowser.open(self._docs_local_url())

    def _docs_local_url(self):
        base = os.path.dirname(os.path.abspath(sys.argv[0]))
        candidates = [
            os.path.join(base, "docs", "index.html"),
            os.path.join(base, "docs.html"),
            os.path.join(base, "StopFilmingDocs.html"),
            os.path.join(base, "help", "index.html"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return QUrl.fromLocalFile(p).toString()

        import os as _os
        url = _os.environ.get("STOPFILMING_DOCS_URL")
        if url:
            return url

        return "https://example.com/stopfilming/docs"

    def _check_for_updates(self):
        QMessageBox.information(self, "Check for Updates", "No updates available.")

    def _show_shortcuts_reference(self):
        dlg = KeyboardShortcutsDialog(self)
        dlg.exec_()

    def _show_quick_help(self):
        try:
            from view import HelpDialog
            dlg = HelpDialog(self)
            dlg.exec_()
        except Exception as e:
            QMessageBox.information(self, "Help", f"Unable to open help: {e}")

    def _show_about_dialog(self):
        try:
            from view import AboutDialog
            dlg = AboutDialog(self, version="v1.0", year="2025")
            dlg.exec_()
        except Exception as e:
            QMessageBox.information(self, "About StopFilming", f"StopFilming v1.0\n© 2025\n\n{e}")

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _show_documentation(self):
        try:
            base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            doc_path = os.path.join(base_dir, "documentation.html")

            if os.path.exists(doc_path):
                webbrowser.open(f"file:///{os.path.abspath(doc_path)}")
            else:
                QMessageBox.warning(
                    self,
                    "Documentation Not Found",
                    f"Could not find documentation.html in the application directory:\n{base_dir}"
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while opening documentation:\n{e}")

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
