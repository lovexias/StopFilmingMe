# main.py

import sys
import webbrowser
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QAction,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QListWidgetItem
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

import cv2
import time

from view import EditorPanel
from model import EditorCore
from utils import blur_faces_of_person

class GestureDetectWorker(QThread):
    finished = pyqtSignal(object)  # Will emit segment_starts

    def __init__(self, core):
        super().__init__()
        self.core = core

    def run(self):
        segment_starts = self.core.detect_and_blur_hand_segments()
        self.finished.emit(segment_starts)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # ─── Model ────────────────────────────────────────────────────────
        self.core = EditorCore()

        # ─── Single View: EditorPanel ─────────────────────────────────────
        self.editor_panel = EditorPanel()
        self.setCentralWidget(self.editor_panel)
        self.setWindowTitle("StopFilming")

        # ─── Connect EditorPanel signals to controller slots ─────────────
        self.editor_panel.importRequested.connect(self._on_import_requested)
        self.editor_panel.playToggled.connect(self._on_play_toggled)
        self.editor_panel.frameChanged.connect(self._on_frame_changed)
        self.editor_panel.detectRequested.connect(self._on_detect_requested)
        self.editor_panel.blurRequested.connect(self._on_blur_requested)
        self.editor_panel.thumbnailClicked.connect(self._on_thumbnail_clicked)
        self.editor_panel.gestureItemClicked.connect(self._on_gesture_item_clicked)

        # ─── Playback timer (used when playing back video) ───────────────
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._on_timer_tick)

        # ─── Build the menubar ───────────────────────────────────────────
        self._create_menu_bar()

    def _create_menu_bar(self):
        menubar = self.menuBar()

        # ─── File Menu ─────────────────────────────────────────────────
        file_menu = menubar.addMenu("File")

        open_vid_action = QAction("Open Video…", self)
        open_vid_action.setShortcut("Ctrl+O")
        open_vid_action.triggered.connect(self._on_import_requested)
        file_menu.addAction(open_vid_action)

        save_action = QAction("Save Project…", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # ─── Edit Menu (placeholders) ──────────────────────────────────
        edit_menu = menubar.addMenu("Edit")

        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(lambda: None)
        edit_menu.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(lambda: None)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        preferences_action = QAction("Preferences…", self)
        preferences_action.triggered.connect(self._toggle_appearance)
        edit_menu.addAction(preferences_action)

        # ─── View Menu ─────────────────────────────────────────────────
        view_menu = menubar.addMenu("View")

        toggle_thumbs_action = QAction("Toggle Thumbnails", self, checkable=True)
        toggle_thumbs_action.setChecked(True)
        toggle_thumbs_action.triggered.connect(lambda checked: self.editor_panel.thumbnail_scroll.setVisible(checked))
        view_menu.addAction(toggle_thumbs_action)

        toggle_markers_action = QAction("Toggle Markers Panel", self, checkable=True)
        toggle_markers_action.setChecked(True)
        toggle_markers_action.triggered.connect(lambda checked: self.editor_panel.gesture_list.parentWidget().setVisible(checked))
        view_menu.addAction(toggle_markers_action)

        fullscreen_action = QAction("Fullscreen", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        # ─── Tools Menu ────────────────────────────────────────────────
        tools_menu = menubar.addMenu("Tools")

        detect_action = QAction("Detect Gestures Now", self)
        detect_action.triggered.connect(self.editor_panel.detectRequested.emit)
        tools_menu.addAction(detect_action)

        blur_action = QAction("Blur Current Frame", self)
        blur_action.triggered.connect(lambda: self.editor_panel.blurRequested.emit(self.editor_panel.current_frame_idx))
        tools_menu.addAction(blur_action)

        export_action = QAction("Export Blurred Video…", self)
        export_action.triggered.connect(self._on_save_project)
        tools_menu.addAction(export_action)

        clear_blurs_action = QAction("Clear All Blurs", self)
        clear_blurs_action.triggered.connect(self._on_clear_blurs)
        tools_menu.addAction(clear_blurs_action)

        # ─── Settings Menu ─────────────────────────────────────────────
        settings_menu = menubar.addMenu("Settings")

        appearance_action = QAction("Appearance (Light/Dark)", self)
        appearance_action.triggered.connect(self._toggle_appearance)
        settings_menu.addAction(appearance_action)

        shortcuts_action = QAction("Keyboard Shortcuts…", self)
        shortcuts_action.triggered.connect(self._show_shortcuts_reference)
        settings_menu.addAction(shortcuts_action)

        # ─── Help Menu ─────────────────────────────────────────────────
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About StopFilming", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

        doc_action = QAction("Documentation", self)
        doc_action.triggered.connect(self._open_documentation)
        help_menu.addAction(doc_action)

        check_updates_action = QAction("Check for Updates…", self)
        check_updates_action.triggered.connect(self._check_for_updates)
        help_menu.addAction(check_updates_action)


    # ─── Controller Slots ─────────────────────────────────────────────────

    def _on_import_requested(self):
        vid_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.MP4 *.avi *.AVI *.mov *.MOV);;All Files (*)"
        )
        if not vid_path:
            return

        # 1) Load the new video into the core (this still clears blurred_cache)
        meta = self.core.load_video(vid_path)

        # 2) Delete the old EditorPanel entirely:
        old_panel = self.editor_panel
        old_panel.deleteLater()

        # 3) Create a fresh EditorPanel and hook up all signals again:
        self.editor_panel = EditorPanel()
        self.setCentralWidget(self.editor_panel)
        self.editor_panel.importRequested.connect(self._on_import_requested)
        self.editor_panel.playToggled.connect(self._on_play_toggled)
        self.editor_panel.frameChanged.connect(self._on_frame_changed)
        self.editor_panel.detectRequested.connect(self._on_detect_requested)
        self.editor_panel.blurRequested.connect(self._on_blur_requested)
        self.editor_panel.thumbnailClicked.connect(self._on_thumbnail_clicked)
        self.editor_panel.gestureItemClicked.connect(self._on_gesture_item_clicked)

        # 4) Initialize the new panel with the video info:
        self.editor_panel.set_video_info(
            rotation_angle=meta["rotation_angle"],
            total_frames=meta["total_frames"],
            fps=meta["fps"]
        )

        # 5) Immediately grab frame 0 from the new core and display it:
        frame0 = self.core.get_frame(0)
        if frame0 is not None:
            self.editor_panel.display_frame(frame0, 0)

        # 6) Build fresh thumbnails on the brand‐new panel:
        thumbs = self.core.generate_thumbnails(num_thumbs=16)
        self.editor_panel.add_thumbnails(thumbs)

        # 7) Update the window title:
        self.setWindowTitle(f"StopFilming – Editing: {vid_path}")

        # 8) Ensure the window does not resize beyond screen dimensions
        screen = QApplication.primaryScreen()  # Get the primary screen
        rect = screen.availableGeometry()  # Get screen's available geometry

        self.showNormal()  # Ensure window is not maximized
        self.resize(rect.width(), rect.height())  # Resize the window to fit the screen
        self.setFixedSize(self.size())  # Make the window size fixed (not resizable)

        self.repaint()  # Repaint to apply the resize
        QApplication.processEvents()  # Force UI update

    def _on_save_project(self):
        """
        Called from menu (“Save Project…”) or Tools → Export.
        If no video loaded, show an info dialog. Otherwise, ask where to save MP4
        and call core.export_video(...).
        """
        if not self.core.video_path:
            QMessageBox.information(self, "Save Project", "No project to save (no video loaded).")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Blurred Video As",
            "",
            "MP4 Video (*.mp4);;All Files (*)"
        )
        if not out_path:
            return

        if not out_path.lower().endswith(".mp4"):
            out_path += ".mp4"

        success = self.core.export_video(out_path)
        if success:
            QMessageBox.information(self, "Save Project", f"Exported to:\n{out_path}")
        else:
            QMessageBox.warning(self, "Save Project", "Failed to export edited video.")


    def _on_clear_blurs(self):
        """
        Clear all cached blurred frames in the model, clear the markers in the UI,
        disable Blur Frame button.
        """
        self.core.blurred_frames.clear()
        self.core.blurred_cache.clear()
        self.editor_panel.clear_markers()
        self.editor_panel.blur_button.setEnabled(False)


    def _on_play_toggled(self, play: bool):
        """
        User clicked Play/Pause.  If play=True, start the timer; if play=False, stop it.
        """
        if play:
            interval = int(1000 / self.core.fps) if self.core.fps > 0 else 33
            self.play_timer.start(interval)
        else:
            self.play_timer.stop()


    def _on_frame_changed(self, frame_idx: int):
        """
        User dragged the slider to frame_idx.  Stop playback, show that frame (blurred if cached).
        """
        self.play_timer.stop()
        self.editor_panel.toggle_button.setText("Play")

        if self.core.cap is not None:
            self.core.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        # Now fetch (blurred or raw) and display
        img = self.core.get_frame(frame_idx)
        self.editor_panel.display_frame(img, frame_idx)



    def _on_timer_tick(self):
        """
        Called every ~1000/fps ms while playing.  Grab the next frame from cv2.VideoCapture,
        overlay blur if needed, and display it in the UI.
        """
        ret, frame = self.core.cap.read()
        if not ret:
            self.play_timer.stop()
            self.editor_panel.toggle_button.setText("Play")
            return

        pos = int(self.core.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        img = self.core.blurred_cache.get(pos, frame)
        self.editor_panel.display_frame(img, pos)


    def _on_detect_requested(self):
        """
        User clicked “Detect Gestures”. Show a dark‐themed, indeterminate QProgressDialog
        with a moving color chunk, run core.detect_and_blur_hand_segments() in a thread,
        then populate gesture_list/timestamps with person ID and gesture type.
        """
        total = self.core.total_frames
        if total <= 0:
            # No video loaded or empty video
            return

        # ─── Create an indeterminate, dark‐themed QProgressDialog ──────────────
        self.progress = QProgressDialog("Detecting gestures… Please wait…", None, 0, 0, self.editor_panel)
        self.progress.setWindowTitle("Processing")
        self.progress.setWindowModality(Qt.ApplicationModal)
        self.progress.setCancelButton(None)
        self.progress.setMinimumDuration(0)
        self.progress.setAutoClose(False)
        self.progress.setAutoReset(False)
        self.progress.setMinimumSize(300, 100)
        self.progress.setWindowFlags(
            self.progress.windowFlags()
            & ~Qt.WindowContextHelpButtonHint
        )
        self.progress.setStyleSheet("""
            QProgressDialog {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border-radius: 8px;
                padding: 0px;
                margin: 0px;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 5px;
                text-align: center;
                background-color: #3c3c3c;
                color: #ffffff;
                height: 25px;
                min-width: 0;
                width: 100%;
                margin-left: 0px;
                margin-right: 0px;
                padding-left: 0px;
                padding-right: 0px;
            }
            QProgressBar::chunk {
                background-color: #6cace4;
                animation: busybar 1s linear infinite;
            }
            QLabel {
                color: #e0e0e0;
            }
            @keyframes busybar {
                0% { margin-left: 0px; }
                100% { margin-left: 100%; }
            }
        """)

        self.progress.show()
        QApplication.processEvents()

        for child in self.progress.findChildren(QProgressDialog):
            child.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            child.setMinimumWidth(500)

        # ─── Run detection in a background thread ─────────────
        self.detect_thread = GestureDetectWorker(self.core)
        self.detect_thread.finished.connect(self._on_gesture_detection_finished)
        self.detect_thread.start()

    def _on_gesture_detection_finished(self, segment_starts):
        self.progress.close()

        # ─── Populate the gesture lists with person_id and gesture_type ─────
        self.editor_panel.gesture_list.clear()
        for person_id, gesture_type, frame_idx, landmarks in segment_starts:
            t = frame_idx / self.core.fps
            mm = int(t // 60)
            ss = int(t % 60)
            msec = int((t - int(t)) * 1000)
            time_str = f"{mm:02}:{ss:02}.{msec:03}"

            if gesture_type == "wave":
                emoji = "👋"
            elif gesture_type == "cover_face":
                emoji = "🫣"
            else:
                emoji = "❓"

            item_str = f"{emoji} Person {person_id} {gesture_type.capitalize()} - ({time_str})"
            item = QListWidgetItem(item_str)
            item.setData(Qt.UserRole, (frame_idx, landmarks))
            self.editor_panel.gesture_list.addItem(item)

        cur = self.editor_panel.current_frame_idx
        if cur in self.core.blurred_cache:
            img = self.core.blurred_cache[cur]
            self.editor_panel.display_frame(img, cur)

        if segment_starts:
            self.editor_panel.blur_button.setEnabled(True)



    def _on_blur_requested(self, frame_idx: int):
        """
        User clicked “Blur Frame” on a selected gesture item.
        Blur that person's face throughout the entire video with progress bar.
        Jump to the selected frame after completion.
        """
        # Retrieve selected gesture item and its stored data
        selected_items = self.editor_panel.gesture_list.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        frame_idx, target_landmarks = item.data(Qt.UserRole)

        # ─── Create progress dialog ───────────────────────────────────────
        total = self.core.total_frames
        progress = QProgressDialog("Blurring person in all frames… Please wait…", None, 0, total, self.editor_panel)
        progress.setWindowTitle("Processing")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        progress.setMinimumSize(300, 100)
        progress.setWindowFlags(
            progress.windowFlags()
            & ~Qt.WindowContextHelpButtonHint
        )
        progress.setStyleSheet("""
            QProgressDialog {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border-radius: 8px;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 5px;
                text-align: center;
                background-color: #3c3c3c;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #6cace4;
                width: 20px;
            }
            QLabel {
                color: #e0e0e0;
            }
        """)

        progress.show()
        QApplication.processEvents()

        # ─── Blur all frames for the person ──────────────────────────────
        cap = cv2.VideoCapture(self.core.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(total):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            blurred = blur_faces_of_person(frame, target_landmarks)
            self.core.blurred_frames.add(i)
            self.core.blurred_cache[i] = blurred

            if i % 10 == 0:
                progress.setValue(i + 1)
                QApplication.processEvents()

        cap.release()
        progress.close()

        # ─── Jump to preview of selected frame ───────────────────────────
        if frame_idx in self.core.blurred_cache:
            img = self.core.blurred_cache[frame_idx]
        else:
            img = self.core.get_frame(frame_idx)
        self.editor_panel.display_frame(img, frame_idx)


    def _on_thumbnail_clicked(self, frame_idx: int):
        """
        User clicked one of the 16 thumbnails. Jump to that frame.
        """
        self.play_timer.stop()
        self.editor_panel.toggle_button.setText("Play")

        img = self.core.get_frame(frame_idx)
        self.editor_panel.display_frame(img, frame_idx)
        self.editor_panel.blur_button.setEnabled(True)


    def _on_gesture_item_clicked(self, frame_idx: int):
        """
        User clicked an item in the Detected Gestures list. Jump to that frame.
        """
        self.play_timer.stop()
        self.editor_panel.toggle_button.setText("Play")

        img = self.core.get_frame(frame_idx)
        self.editor_panel.display_frame(img, frame_idx)
        self.editor_panel.blur_button.setEnabled(True)


    def _open_documentation(self):
        webbrowser.open("https://example.com/stopfilming/docs")


    def _check_for_updates(self):
        QMessageBox.information(self, "Check for Updates", "No updates available.")


    def _toggle_appearance(self):
        """
        (Placeholder) Toggle between light & dark theme. Currently not implemented.
        """
        QMessageBox.information(self, "Appearance", "Toggle light/dark (not implemented).")


    def _show_shortcuts_reference(self):
        shortcuts_text = (
            "Ctrl+O: Open Video\n"
            "Ctrl+S: Save Project\n"
            "Ctrl+Q: Quit\n"
            "F11: Fullscreen\n"
        )
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts_text)


    def _show_about_dialog(self):
        QMessageBox.information(self, "About StopFilming", "StopFilming v1.0\n© 2025")


    def resizeEvent(self, event):
        """
        If you want to do something on window‐resize, override here.
        """
        super().resizeEvent(event)
        # (No extra behavior at the moment)


    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()

    # resize window to fit the device's screen size
    screen = app.primaryScreen()
    rect = screen.availableGeometry()
    window.resize(rect.width(), rect.height())
    window.showMaximized()

    sys.exit(app.exec_())