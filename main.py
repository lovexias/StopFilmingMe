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

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QFileDialog, QMessageBox, QDialog
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QElapsedTimer
from PyQt5.QtGui import QPixmap, QIcon, QPainter, QColor, QPen, QBrush

from view import KeyboardShortcutsDialog
from view import EditorPanel, ProcessingDialog, ExportDialog
from model import EditorCore


# ---------------- Workers ----------------

class GestureDetectWorker(QThread):
    finished = pyqtSignal(object)  # segment_starts

    def __init__(self, core):
        super().__init__()
        self.core = core

    def run(self):
        segment_starts = self.core.detect_and_blur_hand_segments()
        self.finished.emit(segment_starts)


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
        self._create_menu_bar()
        self.menuBar().setStyleSheet("""
            QMenuBar { background-color: #2D3748; color: #E2E8F0; spacing: 6px; padding: 2px 10px; }
            QMenuBar::item { background: transparent; padding: 4px 12px; }
            QMenuBar::item:selected { background-color: #4FD1C7; color: #1A202C; border-radius: 4px; }
            QMenu { background-color: #2D3748; color: #E2E8F0; border: 1px solid #4A5568; margin: 2px; }
            QMenu::item { padding: 6px 20px; }
            QMenu::item:selected { background-color: #4A5568; }
        """)

    # ---------- Menus ----------
    def _create_menu_bar(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2D3748, stop:1 #1A202C);
                color: #E2E8F0; spacing: 8px; padding: 4px 12px; border-bottom: 2px solid #4FD1C7;
                font-weight: 500; font-size: 13px;
            }
            QMenuBar::item { background: transparent; padding: 6px 14px; border-radius: 6px; margin: 2px; }
            QMenuBar::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4FD1C7, stop:1 #38B2AC);
                color: #1A202C; font-weight: 600;
            }
            QMenuBar::item:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #319795, stop:1 #2C7A7B);
                color: #E6FFFA;
            }
            QMenu {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2D3748, stop:1 #1A202C);
                color: #E2E8F0; border: 2px solid #4A5568; border-radius: 8px; padding: 6px; margin: 2px;
            }
            QMenu::item { padding: 8px 24px; border-radius: 4px; margin: 1px; }
            QMenu::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4A5568, stop:1 #2D3748);
                color: #4FD1C7;
            }
            QMenu::separator { height: 1px; background: #4A5568; margin: 6px 12px; }
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

    def _check_memory(self):
        """Monitor memory usage and show warnings"""
        try:
            import psutil
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            
            if memory_mb > 2048:  # More than 2GB
                print(f"⚠️ High memory usage: {memory_mb:.0f}MB")
                
                # Trigger cleanup in core
                if hasattr(self.core, '_frame_cache'):
                    self.core._frame_cache.clear()
                    print("🧹 Cleared frame cache")
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Show user warning if very high
                if memory_mb > 4096:  # More than 4GB
                    QMessageBox.warning(
                        self, 
                        "Memory Warning", 
                        f"High memory usage detected ({memory_mb:.0f}MB).\n"
                        "Consider restarting the application or using a smaller video."
                    )
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

        # clean audio from old panel
        old_panel = self.editor_panel
        if hasattr(old_panel, 'cleanup_audio_resources'):
            old_panel.cleanup_audio_resources()
        old_panel.deleteLater()

        # fresh panel
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

        # init panel
        self.editor_panel.set_video_info(
            rotation_angle=meta["rotation_angle"],
            total_frames=meta["total_frames"],
            fps=meta["fps"]
        )
        self.editor_panel.video_path = self.core.video_path  # sets up audio

        # first frame
        frame0 = self.core.get_frame(0)
        if frame0 is not None:
            frame0 = self._apply_rotation(frame0)
            self.editor_panel.display_frame(frame0, 0)

        # thumbnails
        thumbs = self.core.generate_thumbnails(num_thumbs=16)
        self.editor_panel.add_thumbnails(thumbs)

        # window title & fit
        self.setWindowTitle(f"StopFilming — Editing: {vid_path}")
        screen = QApplication.primaryScreen()
        rect = screen.availableGeometry()
        self.showNormal()
        self.resize(rect.width(), rect.height())
        self.repaint()
        QApplication.processEvents()

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
        self.play_timer.stop()
        self.editor_panel.toggle_button.setText("Play")

        if self.core.cap is not None:
            self.core.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        img = self.core.get_frame(frame_idx)
        if frame_idx not in self.core.blurred_cache:
            img = self._apply_rotation(img)
        self.editor_panel.display_frame(img, frame_idx)
        self.editor_panel.audio_seek_to_frame(frame_idx, self.core.fps)

    def _on_timer_tick(self):
        cap = self.core.cap
        if cap is None or not cap.isOpened():
            self.play_timer.stop()
            self.editor_panel.toggle_button.setText("Play")
            self.editor_panel.audio_pause()
            return

        fps = max(1.0, float(self.core.fps))
        elapsed_s = self.play_clock.elapsed() / 1000.0
        target_idx = int(self.play_start_frame + elapsed_s * fps)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if target_idx >= total:
            self.play_timer.stop()
            self.editor_panel.toggle_button.setText("Play")
            self.editor_panel.audio_pause()
            return

        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Only seek if we're significantly off target (reduces expensive seeking)
        if abs(current_pos - target_idx) > 2:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        
        ret, frame = cap.read()
        if not ret or frame is None:
            # Try reading the next frame instead of failing
            ret, frame = cap.read()
            if not ret or frame is None:
                self.play_timer.stop()
                self.editor_panel.toggle_button.setText("Play")
                self.editor_panel.audio_pause()
                return

        frame = self._apply_rotation(frame)
        actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        display_frame = self.core.blurred_cache.get(actual_pos, frame)
        self.editor_panel.display_frame(display_frame, actual_pos)

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

    def _on_gesture_detection_finished(self, segment_starts):
        if hasattr(self, "proc") and self.proc:
            self.proc.finish("Detection complete")
            self.proc = None

        segment_starts = sorted(segment_starts, key=lambda s: s[2])

        self.editor_panel.gesture_list.clear()
        for person_id, gesture_type, raw_frame, bbox in segment_starts:
            if isinstance(raw_frame, float):
                frame_idx = int(round(raw_frame * max(1, self.core.fps)))
            else:
                frame_idx = int(raw_frame)

            frame_idx = max(0, min(frame_idx, max(0, self.core.total_frames - 1)))

            t = frame_idx / max(1, self.core.fps)
            mm = int(t // 60); ss = int(t % 60); msec = int((t - int(t)) * 1000)
            time_str = f"{mm:02}:{ss:02}.{msec:03d}"
            emoji = "👋" if gesture_type == "wave" else "🫣"
            label = f"{emoji} Person {person_id} {gesture_type.capitalize()}  –  ({time_str})"

            from PyQt5.QtWidgets import QListWidgetItem
            item = QListWidgetItem(label)
            payload = {"person_id": int(person_id), "gesture": str(gesture_type),
                       "frame": frame_idx, "bbox": bbox}
            item.setData(Qt.UserRole, payload)
            self.editor_panel.gesture_list.addItem(item)

        has_items = self.editor_panel.gesture_list.count() > 0
        self.editor_panel.blur_button.setEnabled(has_items)
        self.statusBar().showMessage(f"Detected {self.editor_panel.gesture_list.count()} gesture(s)")

    # Alternative fix in main.py - Remove the parameters from the call:
    def _on_blur_requested(self, _frame_idx_from_button: int):
        # 1) read the selected gesture payload
        selected = self.editor_panel.gesture_list.selectedItems()
        if not selected:
            return
        payload = selected[0].data(Qt.UserRole)
        if not payload:
            return

        frame_idx = int(payload["frame"])
        bbox = payload["bbox"]
        sel_pid = payload.get("person_id", "?")

        self.editor_panel.show_selection_badge(f"Blurring Person {sel_pid}…")


        def _blur_progress_cb(pct: int):
            self.editor_panel.set_blur_progress(int(pct))

        # 4) run your blur routine (pick ONE path)
        #    A) if you blur directly into the preview buffer:
        self.editor_panel.start_blur_progress()
        ok = False
        try:
            ok = self.core.blur_person_in_video(
                bbox, start_frame=frame_idx, progress_callback=_blur_progress_cb
            )
        finally:
            # even if an exception is raised, close the dialog
            self.editor_panel.finish_blur_progress(bool(ok))


        # 6) optional: refresh the preview at the current frame
        img = self.core.get_frame(frame_idx)
        self.editor_panel.display_frame(img, frame_idx)
        self.editor_panel.current_frame_idx = frame_idx

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
