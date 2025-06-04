# main.py

import sys
import webbrowser
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QAction,
    QMenu,
    QStackedWidget,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import Qt

from home_page import HomePage
from editor_page import EditorPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # ─── Set up stacked pages ───────────────────────────────────────────
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home_page = HomePage(
            import_callback=self.start_editing,
            continue_callback=self.continue_project
        )
        self.editor_page = EditorPage()

        self.stack.addWidget(self.home_page)    # idx 0
        self.stack.addWidget(self.editor_page)  # idx 1

        self.stack.setCurrentWidget(self.home_page)
        self.setWindowTitle("StopFilming")

        # ─── Build the menubar ──────────────────────────────────────────────
        self._create_menu_bar()

    def _create_menu_bar(self):
        menubar = self.menuBar()

        # ─── File Menu ────────────────────────────────────────────────────
        file_menu = menubar.addMenu("File")

        open_vid_action = QAction("Open Video…", self)
        open_vid_action.setShortcut("Ctrl+O")
        open_vid_action.triggered.connect(self._menu_open_video)
        file_menu.addAction(open_vid_action)

        save_action = QAction("Save Project…", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._menu_save_project)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        close_proj_action = QAction("Close Project", self)
        close_proj_action.setShortcut("Ctrl+W")
        close_proj_action.triggered.connect(self._menu_back_to_home)
        file_menu.addAction(close_proj_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # ─── Edit Menu ────────────────────────────────────────────────────
        edit_menu = menubar.addMenu("Edit")

        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(lambda: None)  # TODO: hook into undo logic
        edit_menu.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(lambda: None)  # TODO: hook into redo logic
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        cut_action = QAction("Cut", self)
        cut_action.setShortcut("Ctrl+X")
        cut_action.triggered.connect(lambda: None)   # Optional
        edit_menu.addAction(cut_action)

        copy_action = QAction("Copy", self)
        copy_action.setShortcut("Ctrl+C")
        copy_action.triggered.connect(lambda: None)  # Optional
        edit_menu.addAction(copy_action)

        paste_action = QAction("Paste", self)
        paste_action.setShortcut("Ctrl+V")
        paste_action.triggered.connect(lambda: None)  # Optional
        edit_menu.addAction(paste_action)

        edit_menu.addSeparator()

        preferences_action = QAction("Preferences…", self)
        preferences_action.triggered.connect(self._open_preferences_dialog)
        edit_menu.addAction(preferences_action)

        # ─── View Menu ────────────────────────────────────────────────────
        view_menu = menubar.addMenu("View")

        toggle_thumbs_action = QAction("Toggle Thumbnails", self, checkable=True)
        toggle_thumbs_action.setChecked(True)
        toggle_thumbs_action.triggered.connect(self._toggle_thumbnails)
        view_menu.addAction(toggle_thumbs_action)

        toggle_markers_action = QAction("Toggle Markers Panel", self, checkable=True)
        toggle_markers_action.setChecked(True)
        toggle_markers_action.triggered.connect(self._toggle_markers_panel)
        view_menu.addAction(toggle_markers_action)

        view_menu.addSeparator()

        fullscreen_action = QAction("Fullscreen", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        # ─── Tools Menu ────────────────────────────────────────────────────
        tools_menu = menubar.addMenu("Tools")

        detect_action = QAction("Detect Gestures Now", self)
        detect_action.triggered.connect(self.editor_page._on_detect_clicked)
        tools_menu.addAction(detect_action)

        blur_action = QAction("Blur Current Frame", self)
        blur_action.triggered.connect(self.editor_page._on_blur_clicked)
        tools_menu.addAction(blur_action)

        tools_menu.addSeparator()

        export_action = QAction("Export Blurred Video…", self)
        export_action.triggered.connect(self._menu_save_project)
        tools_menu.addAction(export_action)

        clear_blurs_action = QAction("Clear All Blurs", self)
        clear_blurs_action.triggered.connect(self._clear_blurred_cache)
        tools_menu.addAction(clear_blurs_action)

        # ─── Settings Menu ─────────────────────────────────────────────────
        settings_menu = menubar.addMenu("Settings")

        appearance_action = QAction("Appearance (Light/Dark)", self)
        appearance_action.triggered.connect(self._toggle_appearance)
        settings_menu.addAction(appearance_action)

        shortcuts_action = QAction("Keyboard Shortcuts…", self)
        shortcuts_action.triggered.connect(self._show_shortcuts_reference)
        settings_menu.addAction(shortcuts_action)

        # ─── Help Menu ─────────────────────────────────────────────────────
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

    # ─── Slot Methods for File / Edit / View / Tools / Settings / Help ───────

    def _menu_open_video(self):
        vid_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Video Files (*.mp4 *.MP4 *.avi *.AVI *.mov *.MOV);;All Files (*)"
        )
        if not vid_path:
            return

        if self.stack.currentWidget() == self.editor_page:
            # Already in editor: just reload
            self.editor_page.load_video(vid_path)
        else:
            # From home: start editing
            self.start_editing(vid_path)

    def _menu_save_project(self):
        if self.stack.currentWidget() != self.editor_page:
            QMessageBox.information(self, "Save Project", "Nothing to save (no project open).")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Edited Video As",
            "",
            "MP4 Video (*.mp4);;All Files (*)"
        )
        if not out_path:
            return

        if not out_path.lower().endswith(".mp4"):
            out_path += ".mp4"

        success = self.editor_page.export_video(out_path)
        if success:
            QMessageBox.information(self, "Save Project", f"Exported to:\n{out_path}")
        else:
            QMessageBox.warning(self, "Save Project", "Failed to export edited video.")

    def _menu_back_to_home(self):
        self.stack.setCurrentWidget(self.home_page)
        self.setWindowTitle("StopFilming")

    def start_editing(self, video_path):
        self.editor_page.load_video(video_path)
        self.stack.setCurrentWidget(self.editor_page)
        self.setWindowTitle(f"StopFilming – Editing: {video_path}")

    def continue_project(self):
        # … implement if you want “Open .sfproj” or similar …
        pass

    # ─── EDIT Menu Slots ─────────────────────────────────────────────────────

    def _open_preferences_dialog(self):
        # TODO: Show a QDialog letting user specify default settings (e.g. thumbnail count).
        QMessageBox.information(self, "Preferences", "Preferences dialog (not implemented).")

    # ─── VIEW Menu Slots ─────────────────────────────────────────────────────

    def _toggle_thumbnails(self, checked: bool):
        """
        Show/hide the bottom thumbnail strip.
        """
        # The EditorPage thumbnail_scroll is the bottom QScrollArea
        if checked:
            self.editor_page.thumbnail_scroll.show()
        else:
            self.editor_page.thumbnail_scroll.hide()

    def _toggle_markers_panel(self, checked: bool):
        """
        Show/hide the right‐hand “Markers” pane.
        """
        # The markers pane is the parent widget of gesture_list in EditorPage
        markers_pane = self.editor_page.gesture_list.parentWidget()
        if checked:
            markers_pane.show()
        else:
            markers_pane.hide()

    def _toggle_fullscreen(self):
        """
        Toggle between fullscreen & normal windowed mode.
        """
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ─── SETTINGS Menu Slots ─────────────────────────────────────────────────

    def _toggle_appearance(self):
        """
        Switch between light and dark theme by re‐applying a different stylesheet.
        """
        QMessageBox.information(self, "Appearance", "Toggle light/dark (not implemented).")

    def _show_shortcuts_reference(self):
        """
        Display a small dialog listing all keyboard shortcuts (Ctrl+O, Ctrl+S, etc.).
        """
        shortcuts_text = (
            "Ctrl+O: Open Video\n"
            "Ctrl+S: Save Project\n"
            "Ctrl+W: Close Project\n"
            "Ctrl+Q: Quit\n"
            "Ctrl+Z: Undo\n"
            "Ctrl+Y: Redo\n"
            "F11: Fullscreen\n"
        )
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts_text)

    # ─── HELP Menu Slots ─────────────────────────────────────────────────────

    def _show_about_dialog(self):
        QMessageBox.information(self, "About StopFilming", "StopFilming v1.0\n© 2025")

    def _open_documentation(self):
        """
        Open a local or online documentation URL.
        """
        # Example: open a web page. Replace with your actual docs URL if you have one.
        webbrowser.open("https://example.com/stopfilming/docs")

    def _check_for_updates(self):
        """
        (Optional) If you host a version‐check endpoint, call it here.
        """
        QMessageBox.information(self, "Check for Updates", "No updates available.")

    # ─── TOOLS Menu Slot ─────────────────────────────────────────────────────
    def _clear_blurred_cache(self):
        """
        Clear all the cached blurred frames, and reset gesture/timestamp lists.
        """
        self.editor_page.core.blurred_frames.clear()
        self.editor_page.core.blurred_cache.clear()
        self.editor_page.gesture_list.clear()
        self.editor_page.log_list.clear()
        self.editor_page.blur_button.setEnabled(False)

    # ─── OPTIONAL OVERRIDES ──────────────────────────────────────────────────

    def resizeEvent(self, event):
        """
        If you want to update window title or do something on resize, override here.
        """
        super().resizeEvent(event)
        # For now, we do nothing special on resize in MainWindow.


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    sys.exit(app.exec_())
