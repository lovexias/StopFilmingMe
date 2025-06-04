# ─── In combined_view.py ────────────────────────────────────────────

import cv2
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QSlider,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QLabel,
    QFrame,
    QSizePolicy,
    QProgressDialog,
    QApplication,
)
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QCursor, QPainter, QColor, QFont


class TimeRuler(QWidget):
    """
    Horizontal ruler with tick marks and mm:ss labels, plus a caret for current frame.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame = 0

    def setVideoInfo(self, total_frames: int, fps: float):
        self.total_frames = total_frames
        self.fps = fps
        self.current_frame = 0
        self.update()

    def setCurrentFrame(self, frame_idx: int):
        self.current_frame = frame_idx
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, QColor("#3C4A5A"))
        if self.total_frames <= 0 or self.fps <= 0:
            painter.end()
            return

        total_seconds = self.total_frames / self.fps
        num_ticks = 10
        interval_sec = total_seconds / num_ticks

        pen = painter.pen()
        pen.setColor(QColor("#ECECEC"))
        painter.setPen(pen)

        font = QFont()
        font.setPointSize(7)
        painter.setFont(font)

        for i in range(num_ticks + 1):
            sec = i * interval_sec
            x = int((sec / total_seconds) * w)
            painter.drawLine(x, 0, x, int(h * 0.4))

            mm = int(sec // 60)
            ss = int(sec % 60)
            label = f"{mm:02}:{ss:02}"
            painter.drawText(x + 2, int(h * 0.8), label)

        cur_sec = self.current_frame / self.fps
        if cur_sec > total_seconds:
            cur_sec = total_seconds
        x_cur = int((cur_sec / total_seconds) * w)

        pen2 = painter.pen()
        pen2.setColor(QColor("#FFCC00"))
        pen2.setWidth(2)
        painter.setPen(pen2)
        painter.drawLine(x_cur, 0, x_cur, h)

        painter.end()


class EditorPanel(QWidget):
    """
    Pure UI “Editor” panel.  Exposes signals for any user action.
    Until a video is loaded, a small “Import Video…” pop‐up appears inside
    the entire left pane. Once a video is imported, that pop‐up is hidden.
    """
    importRequested    = pyqtSignal()     # User clicked “Import Video…” (overlay)
    playToggled        = pyqtSignal(bool) # True=Play, False=Pause
    frameChanged       = pyqtSignal(int)  # Slider moved to frameIdx
    detectRequested    = pyqtSignal()     # User clicked “Detect Gestures”
    blurRequested      = pyqtSignal(int)  # User clicked “Blur Frame” on current frame
    thumbnailClicked   = pyqtSignal(int)  # User clicked a thumbnail (frameIdx)
    gestureItemClicked = pyqtSignal(int)  # User clicked a gesture‐list item (frameIdx)

    def __init__(self, parent=None):
        super().__init__(parent)

        # ─── Ensure this flag exists immediately ─────────────────────────
        self.core_has_video = False

        # ─── Internal state (set by Controller) ─────────────────────────
        self.cap = None
        self.video_path = None
        self.rotation_angle = 0
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame_idx = 0
        self.is_playing = False

        # ─── Dark styling (you can tweak colors/fonts here) ─────────────
        self.setStyleSheet("""
            QWidget { background-color: #2F3E4E; color: #ECECEC; font-family: Arial; }
            QLabel#video_display { background-color: #1E2834; border: 1px solid #22303D; }
            QSlider::groove:horizontal { background: #3C4A5A; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #A1BFCF; border: 1px solid #789BAF; width: 14px; margin: -4px 0; border-radius: 7px; }
            QPushButton { background-color: #507B9C; border: none; padding: 6px 12px; border-radius: 4px; color: #FFF; }
            QPushButton:hover { background-color: #4A738D; }
            QPushButton:pressed { background-color: #3A5A76; }
            QPushButton:disabled { background-color: #415262; color: #888; }
            QPushButton#importPopupBtn { font-size: 18px; padding: 12px 24px; }
            QListWidget { background-color: #324154; border: 1px solid #22303D; border-radius: 4px; }
            QListWidget::item { padding: 4px; }
            QListWidget::item:selected { background-color: #507B9C; color: #FFF; }
            QLabel.thumbLabel { border: 1px solid #22303D; border-radius: 3px; }
            QLabel.thumbLabel:hover { border: 1px solid #A1BFCF; }
            QFrame#bottom_bar { background-color: #3C4A5A; padding: 4px; border-top: 1px solid #22303D; }
        """)

        self._build_ui()


    def _build_ui(self):
        """
        Build the combined “Import + Editor” UI in one single QWidget.
        The import‐overlay covers the entire left pane (video + controls)
        until a file is loaded.
        """
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ─── Middle: Video / Markers split ───────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle { background: #22303D; }")

        #
        #  1) Left side: “video_container” (video_display + slider + buttons)
        #
        video_container = QFrame()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)

        # ─── The QLabel where we will paint each frame (or show a blank background) ─
        self.video_display = QLabel()
        self.video_display.setObjectName("video_display")
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_display.setScaledContents(True)
        video_layout.addWidget(self.video_display)

        # ─── PLACEHOLDER “Import Video…” pop‐up, but now parented to video_container ─
        # That way, it covers both the QLabel *and* the slider/buttons below it.
        self.import_popup = QFrame(video_container)
        self.import_popup.setObjectName("importPlaceholder")
        self.import_popup.setStyleSheet("""
            background-color: rgba(50, 50, 50, 220);
            border: 2px solid #789BAF;
            border-radius: 8px;
        """)
        self.import_popup.raise_()

        popup_layout = QVBoxLayout(self.import_popup)
        popup_layout.setContentsMargins(0, 0, 0, 0)
        popup_layout.setSpacing(0)
        popup_layout.addStretch()

        hbox = QHBoxLayout()
        hbox.addStretch()

        self.importPopupBtn = QPushButton("Import Video…", self.import_popup)
        self.importPopupBtn.setObjectName("importPopupBtn")
        self.importPopupBtn.setFixedSize(200, 60)
        self.importPopupBtn.clicked.connect(self.importRequested.emit)
        hbox.addWidget(self.importPopupBtn)
        hbox.addStretch()

        popup_layout.addLayout(hbox)
        popup_layout.addStretch()

        # ─── Make it cover exactly the entire video_container to start ─────────
        # Replace this with a fixed rectangle (for example: x=20, y=20, width=800, height=600):
        self.import_popup.setGeometry(1000, 20, 2000, 1000)
        self.import_popup.show()

        # ─── Slider directly below the video area ────────────────────────────
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(lambda v: self.frameChanged.emit(v))
        video_layout.addWidget(self.slider)

        # ─── Row of buttons (Play / Detect / Blur) ──────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 4, 0, 4)
        btn_row.setSpacing(8)

        self.toggle_button = QPushButton("Play")
        self.toggle_button.setEnabled(False)
        self.toggle_button.clicked.connect(self._on_toggle_clicked)
        btn_row.addStretch()
        btn_row.addWidget(self.toggle_button)
        btn_row.addStretch()

        self.detect_button = QPushButton("Detect Gestures")
        self.detect_button.setEnabled(False)
        self.detect_button.clicked.connect(lambda: self.detectRequested.emit())
        btn_row.addWidget(self.detect_button)

        self.blur_button = QPushButton("Blur Frame")
        self.blur_button.setEnabled(False)
        self.blur_button.clicked.connect(lambda: self.blurRequested.emit(self.current_frame_idx))
        btn_row.addWidget(self.blur_button)

        video_layout.addLayout(btn_row)
        splitter.addWidget(video_container)

        #
        #  2) Right side: Markers pane (“Detected Gestures” + “Timestamps”)
        #
        markers_container = QFrame()
        markers_container.setFixedWidth(260)
        markers_layout = QVBoxLayout(markers_container)
        markers_layout.setContentsMargins(10, 10, 10, 10)
        markers_layout.setSpacing(8)

        markers_title = QLabel("Detected Gestures")
        markers_title.setStyleSheet("font-weight: bold; color: #A1BFCF; font-size: 14px;")
        markers_layout.addWidget(markers_title)

        self.gesture_list = QListWidget()
        self.gesture_list.setFixedHeight(180)
        self.gesture_list.itemClicked.connect(
            lambda item: self.gestureItemClicked.emit(item.data(Qt.UserRole))
        )
        markers_layout.addWidget(self.gesture_list)

        log_title = QLabel("Timestamps")
        log_title.setStyleSheet("font-weight: bold; color: #A1BFCF; font-size: 14px;")
        markers_layout.addWidget(log_title)

        self.log_list = QListWidget()
        self.log_list.setFixedHeight(180)
        markers_layout.addWidget(self.log_list)

        clear_btn = QPushButton("Clear Markers")
        clear_btn.clicked.connect(self._clear_markers)
        clear_btn.setFixedHeight(40)
        markers_layout.addWidget(clear_btn)

        markers_layout.addStretch()
        splitter.addWidget(markers_container)

        # ─── Let the left “video_container” expand, and the right pane stay fixed ─
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        # ─── Add the splitter → NOTE the “stretch=1” so it fills all vertical space ─
        main_layout.addWidget(splitter, stretch=1)

        #
        #  3) Bottom bar: TimeRuler + Thumbnails
        #
        bottom_bar = QFrame()
        bottom_bar.setObjectName("bottom_bar")
        bottom_layout = QVBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(10, 4, 10, 4)
        bottom_layout.setSpacing(4)

        self.time_ruler = TimeRuler()
        self.time_ruler.setFixedHeight(30)
        bottom_layout.addWidget(self.time_ruler)

        self.thumbnail_scroll = QScrollArea()
        self.thumbnail_scroll.setFixedHeight(100)
        self.thumbnail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.thumbnail_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.thumbnail_scroll.setWidgetResizable(True)

        thumb_container = QWidget()
        self.thumbnail_layout = QHBoxLayout(thumb_container)
        self.thumbnail_layout.setContentsMargins(0, 0, 0, 0)
        self.thumbnail_layout.setSpacing(6)
        self.thumbnail_scroll.setWidget(thumb_container)

        bottom_layout.addWidget(self.thumbnail_scroll)

        # ─── Add bottom bar with stretch=0 so it stays at its natural height ───
        main_layout.addWidget(bottom_bar, stretch=0)

        # ─── Playback Timer (Controller hooks this up later) ────────────
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer_tick)


    def _update_import_popup_visibility(self):
        """
        Hide the “Import Video…” overlay once a video is imported.
        """
        if self.core_has_video:
            self.import_popup.hide()
        else:
            self.import_popup.show()


    def clear_thumbnails(self):
        """
        Remove all existing thumbnail widgets from thumbnail_layout,
        and reset the lists that track them.
        """
        if hasattr(self, "thumbnail_labels"):
            for thumb in self.thumbnail_labels:
                # remove the widget from the layout and delete it
                self.thumbnail_layout.removeWidget(thumb)
                thumb.deleteLater()

            # reset the tracking lists
            self.thumbnail_labels = []
            self.thumbnail_frame_indices = []


    # ─── Public methods that Controller will call ────────────────────────

    
    def set_video_info(self, rotation_angle, total_frames, fps):
        """
        Called by Controller after core.load_video().
        We enable all controls, then hide the import-overlay.
        """
        self.rotation_angle = rotation_angle
        self.total_frames = total_frames
        self.fps = fps
        self.current_frame_idx = 0

        # Enable slider & buttons:
        self.slider.setMaximum(max(0, total_frames - 1))
        self.slider.setEnabled(True)

        self.toggle_button.setEnabled(True)
        self.toggle_button.setText("Play")
        self.detect_button.setEnabled(True)
        self.blur_button.setEnabled(True)

        self.time_ruler.setVideoInfo(total_frames, fps)

        # Hide the import pop‐up (because now a video is loaded)
        self.core_has_video = True
        self._update_import_popup_visibility()


    def display_frame(self, img_bgr, frame_idx: int):
        """
        Called by MainWindow whenever we want to show a frame.
        If img_bgr is None, force‐clear the QLabel so no old frame remains.
        """
        # ─── If no image, force a blank pixmap and return ────────────────────
        if img_bgr is None:
            w = self.video_display.width()
            h = self.video_display.height()
            blank = QPixmap(w, h)
            blank.fill(Qt.black)               # or Qt.white, etc.
            self.video_display.setPixmap(blank)
            self.video_display.repaint()
            self.current_frame_idx = -1
            self.time_ruler.setCurrentFrame(-1)
            return

        # ─── Otherwise, convert BGR→RGB→QImage→QPixmap and show it ─────────
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        bytes_per_line = w * 3
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        self.video_display.setPixmap(pix)
        self.current_frame_idx = frame_idx

        # Update ruler and slider
        self.time_ruler.setCurrentFrame(frame_idx)
        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)
        self.video_display.repaint()

    def add_thumbnails(self, thumbs):
        """
        Called by Controller with list of (idx, RGB‐thumb_numpy). Create labels.
        """
        if not hasattr(self, "thumbnail_labels"):
            self.thumbnail_labels = []
            self.thumbnail_frame_indices = []

        for idx, thumb_rgb in thumbs:
            h, w, _ = thumb_rgb.shape
            bytes_per_line = w * 3
            qimg = QImage(thumb_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaledToHeight(80, Qt.SmoothTransformation)

            thumb_label = QLabel()
            thumb_label.setProperty("class", "thumbLabel")
            thumb_label.setPixmap(pix)
            thumb_label.setFixedSize(QSize(pix.width(), pix.height()))
            thumb_label.setCursor(QCursor(Qt.PointingHandCursor))
            thumb_label.mousePressEvent = lambda e, i=idx: self.thumbnailClicked.emit(i)

            self.thumbnail_layout.addWidget(thumb_label)
            self.thumbnail_labels.append(thumb_label)
            self.thumbnail_frame_indices.append(idx)


    def add_gesture_items(self, segment_starts):
        """
        Called by Controller after detect_and_blur. Populate gesture_list & log_list.
        """
        self.gesture_list.clear()
        self.log_list.clear()
        for idx in segment_starts:
            t = idx / self.fps
            mm = int(t // 60)
            ss = int(t % 60)
            msec = int((t - int(t)) * 1000)
            time_str = f"{mm:02}:{ss:02}.{msec:03}"

            item = QListWidgetItem(f"✋  {time_str}")
            item.setData(Qt.UserRole, idx)
            self.gesture_list.addItem(item)
            self.log_list.addItem(time_str)


    def clear_markers(self):
        self.gesture_list.clear()
        self.log_list.clear()
        self.blur_button.setEnabled(False)


    def show_progress_dialog(self, title="Processing", message="Please wait…"):
        """
        Show a modal indeterminate QProgressDialog until hide_progress_dialog() is called.
        """
        self._progress = QProgressDialog(message, None, 0, 0, self)
        self._progress.setWindowTitle(title)
        self._progress.setWindowModality(Qt.WindowModal)
        self._progress.setCancelButton(None)
        self._progress.setMinimumDuration(0)
        self._progress.setAutoClose(False)
        self._progress.setAutoReset(False)
        self._progress.show()
        QApplication.processEvents()


    def hide_progress_dialog(self):
        if hasattr(self, "_progress"):
            self._progress.close()
            del self._progress


    # ─── Internal slots (connected to UI controls) ───────────────────────

    def _on_toggle_clicked(self):
        """
        Emit playToggled signal with new state (True=play, False=pause).
        """
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.toggle_button.setText("Pause")
        else:
            self.toggle_button.setText("Play")
        self.playToggled.emit(self.is_playing)

    def _on_timer_tick(self):
        """
        Should be connected by Controller to a QTimer. Controller will pull next frame from core.
        """
        pass  # Controller will manage playback loop—UI does not call core here.

    def _clear_markers(self):
        self.gesture_list.clear()
        self.log_list.clear()
        self.blur_button.setEnabled(False)


    # ─── Re‐center the import_popup whenever the widget is resized ─────────
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Replace this with the same hard-coded rectangle used above:
        self.import_popup.setGeometry(70, 180, 1500, 600)
