# editor_page.py

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
    QApplication
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QCursor, QPainter, QColor, QFont

from editor_core import EditorCore


class TimeRuler(QWidget):
    """
    A simple horizontal ruler showing tick marks and mm:ss labels,
    plus a vertical indicator for the current frame position.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame = 0

    def setVideoInfo(self, total_frames: int, fps: float):
        """
        Call this once when a new video is loaded, so the ruler knows length & fps.
        """
        self.total_frames = total_frames
        self.fps = fps
        self.current_frame = 0
        self.update()

    def setCurrentFrame(self, frame_idx: int):
        """
        Call this whenever you display a new frame—moves the yellow indicator.
        """
        self.current_frame = frame_idx
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w = self.width()
        h = self.height()

        # Fill background
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

        # Draw tick marks and labels
        for i in range(num_ticks + 1):
            sec = i * interval_sec
            x = int((sec / total_seconds) * w)
            painter.drawLine(x, 0, x, int(h * 0.4))

            mm = int(sec // 60)
            ss = int(sec % 60)
            label = f"{mm:02}:{ss:02}"
            painter.drawText(x + 2, int(h * 0.8), label)

        # Draw vertical indicator for current frame
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


class EditorPage(QWidget):
    def __init__(self):
        super().__init__()

        self.core = EditorCore()
        self.cap = None
        self.video_path = None
        self.rotation_angle = 0
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame_idx = 0

        # ─── Styling ─────────────────────────────────────────────────────────
        self.setStyleSheet("""
            QWidget { background-color: #2F3E4E; color: #ECECEC; font-family: Arial; }
            QLabel#video_display { background-color: transparent; border: none; }
            QSlider::groove:horizontal { background: #3C4A5A; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #A1BFCF; border: 1px solid #789BAF; width: 14px; margin: -4px 0; border-radius: 7px; }
            QPushButton { background-color: #507B9C; border: none; padding: 6px 12px; border-radius: 4px; color: #FFF; }
            QPushButton:hover { background-color: #4A738D; }
            QPushButton:pressed { background-color: #3A5A76; }
            QPushButton:disabled { background-color: #415262; color: #888; }
            QListWidget { background-color: #324154; border: 1px solid #22303D; border-radius: 4px; }
            QListWidget::item { padding: 4px; }
            QListWidget::item:selected { background-color: #507B9C; color: #FFF; }
            QLabel.thumbLabel { border: 1px solid #22303D; border-radius: 3px; }
            QLabel.thumbLabel:hover { border: 1px solid #A1BFCF; }
            QFrame#bottom_bar { background-color: #3C4A5A; padding: 4px; border-top: 1px solid #22303D; }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ─── Splitter: Video (left) & Markers (right) ─────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle { background: #22303D; }")

        # 1a) Video Display (left)
        video_container = QFrame()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)

        self.video_display = QLabel()
        self.video_display.setObjectName("video_display")
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_display.setScaledContents(True)
        video_layout.addWidget(self.video_display)

        # Slider under video
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        video_layout.addWidget(self.slider)

        # Buttons: Play, Detect, Blur
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
        self.detect_button.clicked.connect(self._on_detect_clicked)
        btn_row.addWidget(self.detect_button)

        self.blur_button = QPushButton("Blur Frame")
        self.blur_button.setEnabled(False)
        self.blur_button.clicked.connect(self._on_blur_clicked)
        btn_row.addWidget(self.blur_button)

        video_layout.addLayout(btn_row)
        splitter.addWidget(video_container)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        # 1b) Markers / Gestures pane (right)
        markers_container = QFrame()
        markers_container.setFixedWidth(260)
        markers_layout = QVBoxLayout(markers_container)
        markers_layout.setContentsMargins(10, 10, 10, 10)
        markers_layout.setSpacing(8)

        markers_title = QLabel("Detected Gestures")
        markers_title.setStyleSheet("font-weight: bold; color: #A1BFCF;")
        markers_layout.addWidget(markers_title)

        self.gesture_list = QListWidget()
        self.gesture_list.setFixedHeight(180)
        self.gesture_list.itemClicked.connect(self._on_gesture_item_clicked)
        markers_layout.addWidget(self.gesture_list)

        log_title = QLabel("Timestamps")
        log_title.setStyleSheet("font-weight: bold; color: #A1BFCF;")
        markers_layout.addWidget(log_title)

        self.log_list = QListWidget()
        self.log_list.setFixedHeight(180)
        markers_layout.addWidget(self.log_list)

        clear_btn = QPushButton("Clear Markers")
        clear_btn.clicked.connect(self._clear_markers)
        markers_layout.addWidget(clear_btn)

        markers_layout.addStretch()
        splitter.addWidget(markers_container)

        main_layout.addWidget(splitter)

        # ─── Bottom bar: TimeRuler + Thumbnails ──────────────────────────────
        bottom_bar = QFrame()
        bottom_bar.setObjectName("bottom_bar")
        bottom_layout = QVBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(10, 4, 10, 4)
        bottom_layout.setSpacing(4)

        # 2a) TimeRuler (30px height)
        self.time_ruler = TimeRuler()
        self.time_ruler.setFixedHeight(30)
        bottom_layout.addWidget(self.time_ruler)

        # 2b) Thumbnail strip
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
        main_layout.addWidget(bottom_bar)

        # ─── Playback Timer ───────────────────────────────────────────────────
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer_tick)


    def load_video(self, video_path: str):
        """
        Called by MainWindow. Loads via core, enables controls, builds thumbnails,
        shows frame 0, and tells the time ruler about total length.
        """
        meta = self.core.load_video(video_path)
        self.cap = self.core.cap
        self.video_path = video_path
        self.rotation_angle = meta["rotation_angle"]
        self.total_frames = meta["total_frames"]
        self.fps = meta["fps"]

        # Enable slider & buttons
        self.slider.setMaximum(max(0, self.total_frames - 1))
        self.slider.setEnabled(True)

        self.toggle_button.setEnabled(True)
        self.toggle_button.setText("Play")

        self.detect_button.setEnabled(True)
        self.blur_button.setEnabled(True)

        # ─── Tell the TimeRuler the total video length ─────────────────────
        self.time_ruler.setVideoInfo(self.total_frames, self.fps)

        # Clear old thumbnails
        for lbl in getattr(self, "thumbnail_labels", []):
            lbl.deleteLater()
        self.thumbnail_labels = []
        self.thumbnail_frame_indices = []

        # Generate thumbnails
        thumbs = self.core.generate_thumbnails(num_thumbs=16)
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
            thumb_label.mousePressEvent = lambda e, i=idx: self._display_frame(i)

            self.thumbnail_layout.addWidget(thumb_label)
            self.thumbnail_labels.append(thumb_label)
            self.thumbnail_frame_indices.append(idx)

        # Show first frame
        self._display_frame(0)
        self.slider.blockSignals(True)
        self.slider.setValue(0)
        self.slider.blockSignals(False)


    def _display_frame(self, frame_idx: int):
        """
        Fetch BGR (or cached blurred), convert to QPixmap, and set on QLabel.
        Also update current_frame_idx, slider, and time ruler indicator.
        """
        img_bgr = self.core.get_frame(frame_idx)
        if img_bgr is None:
            return

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        bytes_per_line = w * 3
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        self.video_display.setPixmap(pix)
        self.current_frame_idx = frame_idx

        # ─── Update TimeRuler’s caret ───────────────────────────────────
        self.time_ruler.setCurrentFrame(frame_idx)

        # ─── Update slider (without triggering valueChanged) ────────────
        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)


    def _on_slider_changed(self, value: int):
        """When slider moves, immediately display that frame."""
        self._display_frame(value)


    def _on_toggle_clicked(self):
        """Play / Pause logic via a QTimer."""
        if not self.cap:
            return

        if self.timer.isActive():
            self.timer.stop()
            self.toggle_button.setText("Play")
        else:
            interval_ms = int(1000 / self.fps) if self.fps > 0 else 33
            self.timer.start(interval_ms)
            self.toggle_button.setText("Pause")


    def _on_timer_tick(self):
        """On each timer tick, read next frame and display (or blurred)."""
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            self.timer.stop()
            self.toggle_button.setText("Play")
            return

        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self.current_frame_idx = pos

        to_show = self.core.blurred_cache.get(pos, frame)
        rgb = cv2.cvtColor(to_show, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        bytes_per_line = w * 3
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        self.video_display.setPixmap(pix)

        # Update TimeRuler & slider
        self.time_ruler.setCurrentFrame(pos)
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)


    def _on_detect_clicked(self):
        """
        Show a “Detecting gestures…” pop-up while running the heavy detection,
        then populate the gesture_list & log_list. Enable blur_button if needed.
        """
        progress = QProgressDialog(
            "Detecting gestures… Please wait…",
            None, 0, 0, self
        )
        progress.setWindowTitle("Processing")
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()

        QApplication.processEvents()

        segment_starts = self.core.detect_and_blur_hand_segments()

        progress.close()

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

        cur_idx = self.current_frame_idx
        if cur_idx in self.core.blurred_cache:
            blurred = self.core.blurred_cache[cur_idx]
            rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            bytes_per_line = w * 3
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.video_display.setPixmap(pix)

        if segment_starts:
            self.blur_button.setEnabled(True)


    def _on_blur_clicked(self):
        """
        Blur exactly the current frame (all faces), cache it, and immediately redisplay.
        """
        idx = self.current_frame_idx
        blurred = self.core.manually_blur_frame(idx)
        if blurred is None:
            return

        rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        bytes_per_line = w * 3
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.video_display.setPixmap(pix)

        # Ensure it stays blurred when scrubbing/playing
        self.blur_button.setEnabled(True)


    def _on_video_clicked(self, event):
        """(If you had added face‐click blur, implement here; otherwise pass.)"""
        pass


    def _on_gesture_item_clicked(self, item: QListWidgetItem):
        """Jump to that frame when the user clicks a gesture timestamp."""
        idx = item.data(Qt.UserRole)
        self._display_frame(idx)
        self.blur_button.setEnabled(True)


    def _clear_markers(self):
        """Clear detected gesture markers; keep any cached blurs intact."""
        self.gesture_list.clear()
        self.log_list.clear()
        self.blur_button.setEnabled(False)


    def resizeEvent(self, event):
        """
        Whenever the window is resized, re‐display the current frame so that
        setScaledContents(True) re‐stretches the pixmap correctly, and update
        the TimeRuler caret.
        """
        super().resizeEvent(event)
        if hasattr(self, "current_frame_idx"):
            self._display_frame(self.current_frame_idx)
