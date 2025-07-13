import sys
import webbrowser
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QAction,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QListWidgetItem,
    QLabel,
    QProgressBar
)

from PyQt5.QtGui import QPixmap, QIcon, QPainter, QFont, QColor, QPen, QBrush
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

from view import KeyboardShortcutsDialog

import cv2
import time

from view import EditorPanel
from model import EditorCore
from utilities import blur_faces_of_person
from utilities import detect_and_blur_multiple_people
from utilities import detect_multiple_people_yolov8
from utilities import match_person_id

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
        #self.frame_skip = 60
        self.setWindowIcon(create_eye_icon())
        self.setWindowTitle("StopFilming - Privacy Protection Video Editor")

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
        self.editor_panel.exportRequested.connect(self._on_save_project)

        # ─── Playback timer (used when playing back video) ───────────────
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._on_timer_tick)

        # ─── Build the menubar ───────────────────────────────────────────
        self._create_menu_bar()


    # after _create_menu_bar()
        self.menuBar().setStyleSheet("""
            QMenuBar {
                background-color: #2D3748;
                color: #E2E8F0;
                spacing: 6px;          /* space between menu titles */
                padding: 2px 10px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 4px 12px;
            }
            QMenuBar::item:selected {
                background-color: #4FD1C7;
                color: #1A202C;
                border-radius: 4px;
            }

            QMenu {
                background-color: #2D3748;
                color: #E2E8F0;
                border: 1px solid #4A5568;
                margin: 2px;
            }
            QMenu::item {
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #4A5568;
            }
        """)


    def _create_menu_bar(self):
        menubar = self.menuBar()
        
        # Enhanced menu bar styling - REPLACE YOUR EXISTING menubar.setStyleSheet() with this:
        menubar.setStyleSheet("""
            QMenuBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2D3748, stop:1 #1A202C);
                color: #E2E8F0;
                spacing: 8px;
                padding: 4px 12px;
                border-bottom: 2px solid #4FD1C7;
                font-weight: 500;
                font-size: 13px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 6px 14px;
                border-radius: 6px;
                margin: 2px;
            }
            QMenuBar::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4FD1C7, stop:1 #38B2AC);
                color: #1A202C;
                font-weight: 600;
            }
            QMenuBar::item:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #319795, stop:1 #2C7A7B);
                color: #E6FFFA;
            }

            QMenu {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2D3748, stop:1 #1A202C);
                color: #E2E8F0;
                border: 2px solid #4A5568;
                border-radius: 8px;
                padding: 6px;
                margin: 2px;
            }
            QMenu::item {
                padding: 8px 24px;
                border-radius: 4px;
                margin: 1px;
            }
            QMenu::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4A5568, stop:1 #2D3748);
                color: #4FD1C7;
            }
            QMenu::separator {
                height: 1px;
                background: #4A5568;
                margin: 6px 12px;
            }
        """)

        # ─── File Menu (ADD EMOJIS TO YOUR EXISTING MENU ITEMS) ─────────────
        file_menu = menubar.addMenu("📁 File")

        open_vid_action = QAction("🎬 Open Video…", self)
        open_vid_action.setShortcut("Ctrl+O")
        open_vid_action.triggered.connect(self._on_import_requested)
        file_menu.addAction(open_vid_action)

        save_action = QAction("💾 Save Project…", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        quit_action = QAction("🚪 Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # ─── Edit Menu (ADD EMOJIS) ──────────────────────────────────────
        edit_menu = menubar.addMenu("✏️ Edit")

        undo_action = QAction("↶ Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(lambda: None)
        edit_menu.addAction(undo_action)

        redo_action = QAction("↷ Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(lambda: None)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        preferences_action = QAction("⚙️ Preferences…", self)
        preferences_action.triggered.connect(self._toggle_appearance)
        edit_menu.addAction(preferences_action)

        # ─── View Menu (ADD EMOJIS) ─────────────────────────────────────────
        view_menu = menubar.addMenu("👁️ View")

        toggle_thumbs_action = QAction("🖼️ Toggle Thumbnails", self, checkable=True)
        toggle_thumbs_action.setChecked(True)
        toggle_thumbs_action.triggered.connect(lambda checked: self.editor_panel.thumbnail_scroll.setVisible(checked))
        view_menu.addAction(toggle_thumbs_action)

        toggle_markers_action = QAction("🏷️ Toggle Markers Panel", self, checkable=True)
        toggle_markers_action.setChecked(True)
        toggle_markers_action.triggered.connect(lambda checked: self.editor_panel.gesture_list.parentWidget().setVisible(checked))
        view_menu.addAction(toggle_markers_action)

        fullscreen_action = QAction("⛶ Fullscreen", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        # ─── Tools Menu (ADD EMOJIS) ────────────────────────────────────────
        tools_menu = menubar.addMenu("🔧 Tools")

        detect_action = QAction("🔍 Detect Gestures Now", self)
        detect_action.triggered.connect(self.editor_panel.detectRequested.emit)
        tools_menu.addAction(detect_action)

        blur_action = QAction("🫥 Blur Current Frame", self)
        blur_action.triggered.connect(lambda: self.editor_panel.blurRequested.emit(self.editor_panel.current_frame_idx))
        tools_menu.addAction(blur_action)

        export_action = QAction("📤 Export Blurred Video…", self)
        export_action.triggered.connect(self._on_save_project)
        tools_menu.addAction(export_action)

        clear_blurs_action = QAction("🧹 Clear All Blurs", self)
        clear_blurs_action.triggered.connect(self._on_clear_blurs)
        tools_menu.addAction(clear_blurs_action)

        # ─── Settings Menu (ADD EMOJIS) ─────────────────────────────────────
        settings_menu = menubar.addMenu("⚙️ Settings")

        video_settings_action = QAction("🎥 Video Settings…", self)
        video_settings_action.triggered.connect(self._show_video_settings_dialog)
        settings_menu.addAction(video_settings_action)

        detection_settings_action = QAction("🎯 Detection Settings…", self)
        detection_settings_action.triggered.connect(self._show_detection_settings_dialog)
        settings_menu.addAction(detection_settings_action)

        blur_settings_action = QAction("🌀 Blur Settings…", self)
        blur_settings_action.triggered.connect(self._show_blur_settings_dialog)
        settings_menu.addAction(blur_settings_action)

        settings_menu.addSeparator()

        shortcuts_action = QAction("⌨️ Keyboard Shortcuts…", self)
        shortcuts_action.triggered.connect(self._show_shortcuts_reference)
        settings_menu.addAction(shortcuts_action)

        # ─── Help Menu (ADD EMOJIS) ─────────────────────────────────────────
        help_menu = menubar.addMenu("❓ Help")

        about_action = QAction("ℹ️ About StopFilming", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

        doc_action = QAction("📚 Documentation", self)
        doc_action.triggered.connect(self._open_documentation)
        help_menu.addAction(doc_action)

        check_updates_action = QAction("🔄 Check for Updates…", self)
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
        self.editor_panel.exportRequested.connect(self._on_save_project)

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
        Called every ~1000/fps ms while playing. Grab the next frame from cv2.VideoCapture,
        overlay blur if needed, and display it in the UI.
        """
        ret, frame = self.core.cap.read()
        if not ret:
            self.play_timer.stop()
            self.editor_panel.toggle_button.setText("Play")
            return

        pos = int(self.core.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # Don't skip frames during playback: just display the current frame
        img = self.core.blurred_cache.get(pos, frame)
        self.editor_panel.display_frame(img, pos)

        # Apply blurring and detection only for the current frame
        frame = detect_and_blur_multiple_people(frame, frame_count=pos)  # Process every frame (frame_skip=1)
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

        self.elapsed_label = QLabel("00:00", self.progress)
        self.elapsed_label.setStyleSheet("""
            color: #fff;
            font-size: 12px;
            font-weight: normal;
            background: transparent;
        """)
        self.elapsed_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # Position it precisely over the right edge of the progress bar
        progress_bar = self.progress.findChild(QProgressBar)
        if progress_bar:
            progress_bar_rect = progress_bar.geometry()
            label_width = 40
            label_height = 20
            self.elapsed_label.setGeometry(
                progress_bar_rect.right() - label_width + 2,
                progress_bar_rect.top() + (progress_bar_rect.height() - label_height) // 2,
                label_width,
                label_height
            )
        self.elapsed_label.show()

        # Timer for updating elapsed time
        self._detect_start_time = time.time()
        self._detect_timer = QTimer(self.progress)
        self._detect_timer.timeout.connect(self._update_detect_elapsed)
        self._detect_timer.start(500)

        self.progress.show()
        QApplication.processEvents()

        for child in self.progress.findChildren(QProgressDialog):
            child.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            child.setMinimumWidth(500)

        # ─── Run detection in a background thread ─────────────
        self.detect_thread = GestureDetectWorker(self.core)
        self.detect_thread.finished.connect(self._on_gesture_detection_finished)
        self.detect_thread.start()

    def _update_detect_elapsed(self):
        if hasattr(self, "_detect_start_time") and hasattr(self, "elapsed_label"):
            elapsed = int(time.time() - self._detect_start_time)
            mm = elapsed // 60
            ss = elapsed % 60
            self.elapsed_label.setText(f"{mm:02d}:{ss:02d}")
    
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

        # Loop through frames and blur the selected person's face
        cap = cv2.VideoCapture(self.core.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(total):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            # Apply blurring to the selected person
            blurred = blur_faces_of_person(frame, target_landmarks_list=[target_landmarks])
            self.core.blurred_frames.add(i)
            self.core.blurred_cache[i] = blurred

            progress.setValue(i + 1)
            QApplication.processEvents()

        cap.release()
        progress.close()

        # Display the selected frame after blurring
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
        dlg = KeyboardShortcutsDialog(self)
        dlg.exec_()



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

    #Settings---------------------------------
    def _show_video_settings_dialog(self):
        QMessageBox.information(
            self,
            "Video Settings",
            "Video‐settings are not implemented yet."
        )
    def _show_detection_settings_dialog(self):
        QMessageBox.information(
            self,
            "Detection Settings",
            "Detection settings are not implemented yet."
        )
    def _show_blur_settings_dialog(self):
        QMessageBox.information(
            self,
            "Blur Settings",
            "Blur settings are not implemented yet."
        )


    def _on_export_video_requested(self):
        """
        Called when the 'Export Video' button is clicked.
        Calls the same method as Save Project to export the video.
        """
        self._on_save_project()  # This method already handles exporting the video

def create_eye_icon():
    """Create a custom eye icon for the application"""
    # Create a 32x32 pixmap
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Draw eye shape (outer ellipse)
    painter.setPen(QPen(QColor(70, 130, 180), 2))  # Steel blue
    painter.setBrush(QBrush(QColor(100, 149, 237)))  # Cornflower blue
    painter.drawEllipse(2, 10, 28, 12)
    
    # Draw pupil (inner circle)
    painter.setPen(QPen(QColor(25, 25, 112), 2))  # Midnight blue
    painter.setBrush(QBrush(QColor(25, 25, 112)))
    painter.drawEllipse(13, 13, 6, 6)
    
    # Draw highlight on pupil
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(QColor(255, 255, 255, 180)))
    painter.drawEllipse(14, 14, 2, 2)
    
    painter.end()
    return QIcon(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Set application-wide properties (ADD THESE LINES)
    app.setApplicationName("StopFilming")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Privacy Tools")
    app.setApplicationDisplayName("StopFilming - Privacy Protection Video Editor")
    app.setWindowIcon(create_eye_icon())  # Global app icon

    window = MainWindow()

    # resize window to fit the device's screen size
    screen = app.primaryScreen()
    rect = screen.availableGeometry()
    window.resize(rect.width(), rect.height())
    window.showMaximized()

    sys.exit(app.exec_())