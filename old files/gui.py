import os
import json
import time
import cv2
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QStyle
)
from PyQt5.QtGui import QImage, QPixmap, QCursor, QIcon
from PyQt5.QtCore import QTimer, Qt, QSize

from utils import (
    get_video_rotation,
    generate_thumbnails,
    WaveDetector,
    blur_faces_in_frame
)


# ─── Helper Functions for “Recent Projects” ───

def add_to_recent_projects(video_path):
    """
    Create (or update) ~/.stopfilming/recent_projects.json with an entry
    for video_path. Also generate thumb.png in ~/.stopfilming/thumbs/.
    """
    home = os.path.expanduser("~")
    base_dir = os.path.join(home, ".stopfilming")
    recent_file = os.path.join(base_dir, "recent_projects.json")
    thumb_dir = os.path.join(base_dir, "thumbs")
    os.makedirs(thumb_dir, exist_ok=True)

    # Load existing recents
    try:
        with open(recent_file, "r", encoding="utf-8") as f:
            recents = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        recents = []

    # Remove any existing entry for this path
    recents = [r for r in recents if r.get("path") != video_path]

    # Generate thumbnail.png (first frame)
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if success:
        thumb_name = os.path.splitext(os.path.basename(video_path))[0] + ".png"
        thumb_path = os.path.join(thumb_dir, thumb_name)
        small = cv2.resize(frame, (200, 112), interpolation=cv2.INTER_AREA)
        cv2.imwrite(thumb_path, small)
    else:
        thumb_path = ""

    # Insert new record at front
    recents.insert(0, {
        "path": video_path,
        "timestamp": int(time.time()),
        "thumbnail": thumb_path
    })

    # Keep only 5 most recent
    recents = recents[:5]

    # Write back
    os.makedirs(os.path.dirname(recent_file), exist_ok=True)
    with open(recent_file, "w", encoding="utf-8") as f:
        json.dump(recents, f, indent=2)


def get_recent_projects():
    """
    Returns a list of up to 5 dicts: {"path":..., "thumbnail":...}
    """
    home = os.path.expanduser("~")
    recent_file = os.path.join(home, ".stopfilming", "recent_projects.json")
    try:
        with open(recent_file, "r", encoding="utf-8") as f:
            recents = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        recents = []
    return recents


# ─── HomePage with dynamic “Recent Projects” strip ───

class HomePage(QWidget):
    def __init__(self, import_callback, continue_callback):
        super().__init__()
        self.import_callback = import_callback
        self.continue_callback = continue_callback
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet("background-color: #1E1E1E;")
        self.setContentsMargins(0, 0, 0, 0)

        # Left icon bar
        self.left_icon_layout = QVBoxLayout()
        self.left_icon_layout.setContentsMargins(0, 0, 0, 0)
        self.left_icon_layout.setSpacing(20)

        icons = [
            ("Home",      QStyle.SP_DirHomeIcon),
            ("Import",    QStyle.SP_DialogOpenButton),
            ("OpenProj",  QStyle.SP_DirOpenIcon),
            ("Settings",  QStyle.SP_FileDialogDetailedView),
            ("Help",      QStyle.SP_DialogHelpButton),
            ("SaveProj",  QStyle.SP_DialogSaveButton),
            ("Undo",      QStyle.SP_ArrowBack),
            ("Trash",     QStyle.SP_TrashIcon),
        ]

        for name, stdicon in icons:
            btn = QPushButton()
            btn.setIcon(self.style().standardIcon(stdicon))
            btn.setIconSize(QSize(28, 28))
            btn.setFixedSize(48, 48)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.setToolTip(name)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: none;
                    color: #FFFFFF;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.1);
                }
            """)
            if name == "Import":
                btn.clicked.connect(self._on_import_clicked)
            elif name == "OpenProj":
                btn.clicked.connect(self._on_continue_clicked)
            self.left_icon_layout.addWidget(btn)

        self.left_icon_layout.addStretch()
        left_icon_widget = QWidget()
        left_icon_widget.setLayout(self.left_icon_layout)
        left_icon_widget.setFixedWidth(60)
        left_icon_widget.setStyleSheet("background-color: #232323;")

        # Center drop area + Import button
        self.drop_label = QLabel("Drag & drop your video here")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setFixedHeight(200)
        self.drop_label.setStyleSheet("""
            QLabel {
                background-color: #2A2A2A;
                color: #CCCCCC;
                font-size: 18px;
                border: 2px dashed #555555;
                border-radius: 8px;
                margin: 16px;
            }
        """)

        self.import_btn_home = QPushButton("Import Video")
        self.import_btn_home.setFixedHeight(40)
        self.import_btn_home.setCursor(QCursor(Qt.PointingHandCursor))
        self.import_btn_home.setStyleSheet("""
            QPushButton {
                background-color: #8E44AD;
                color: #FFFFFF;
                font-size: 14px;
                border: none;
                border-radius: 4px;
                padding: 8px 24px;
            }
            QPushButton:hover {
                background-color: #7D3C98;
            }
        """)
        self.import_btn_home.clicked.connect(self._on_import_clicked)

        center_layout = QVBoxLayout()
        center_layout.setContentsMargins(12, 12, 12, 12)
        center_layout.setSpacing(12)
        center_layout.addStretch()
        center_layout.addWidget(self.drop_label)
        center_layout.addWidget(self.import_btn_home, alignment=Qt.AlignCenter)
        center_layout.addStretch()

        center_widget = QWidget()
        center_widget.setLayout(center_layout)

        # Right panel
        right_widget = QWidget()
        right_widget.setFixedWidth(300)
        right_widget.setStyleSheet("background-color: #2A2A2A;")
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(16)

        self.template_preview = QLabel("Template Preview")
        self.template_preview.setAlignment(Qt.AlignCenter)
        self.template_preview.setFixedSize(260, 180)
        self.template_preview.setStyleSheet("""
            QLabel {
                background-color: #3A3A3A;
                color: #DDDDDD;
                font-size: 16px;
                border-radius: 6px;
            }
        """)
        self.continue_btn_home = QPushButton("Continue Project")
        self.continue_btn_home.setFixedHeight(36)
        self.continue_btn_home.setCursor(QCursor(Qt.PointingHandCursor))
        self.continue_btn_home.setStyleSheet("""
            QPushButton {
                background-color: #00BFA5;
                color: #FFFFFF;
                font-size: 13px;
                border: none;
                border-radius: 4px;
                padding: 6px 20px;
            }
            QPushButton:hover {
                background-color: #009477;
            }
        """)
        self.continue_btn_home.clicked.connect(self._on_continue_clicked)

        right_layout.addStretch()
        right_layout.addWidget(self.template_preview, alignment=Qt.AlignCenter)
        right_layout.addWidget(self.continue_btn_home, alignment=Qt.AlignCenter)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)

        # Top splitter
        self.top_splitter = QSplitter(Qt.Horizontal)
        self.top_splitter.addWidget(left_icon_widget)
        self.top_splitter.addWidget(center_widget)
        self.top_splitter.addWidget(right_widget)
        self.top_splitter.setSizes([60, 650, 300])
        self.top_splitter.setHandleWidth(1)

        # Bottom: Recent Projects
        self.bottom_cards_layout = QHBoxLayout()
        self.bottom_cards_layout.setContentsMargins(12, 0, 12, 12)
        self.bottom_cards_layout.setSpacing(16)

        recents = get_recent_projects()
        for entry in recents:
            video_path = entry.get("path", "")
            thumb_path = entry.get("thumbnail", "")
            fname = os.path.basename(video_path)

            card = QLabel()
            card.setFixedSize(160, 100)
            card.setAlignment(Qt.AlignCenter)
            if thumb_path and os.path.exists(thumb_path):
                pix = QPixmap(thumb_path).scaled(150, 70, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                card.setPixmap(pix)
                card.setToolTip(fname)
            else:
                card.setText(fname)
                card.setStyleSheet("""
                    QLabel {
                        background-color: #2A2A2A;
                        color: #EEEEEE;
                        font-size: 12px;
                        border: 1px solid #444444;
                        border-radius: 8px;
                        padding: 12px;
                    }
                    QLabel:hover {
                        background-color: #3A3A3A;
                        border: 1px solid #8E44AD;
                    }
                """)

            card.mousePressEvent = lambda evt, p=video_path: self._open_recent_project(p)
            self.bottom_cards_layout.addWidget(card)

        # Fill up to 5 cards if fewer
        while len(recents) < 5:
            spacer = QLabel("")
            spacer.setFixedSize(160, 100)
            self.bottom_cards_layout.addWidget(spacer)
            recents.append(None)

        bottom_cards_widget = QWidget()
        bottom_cards_widget.setLayout(self.bottom_cards_layout)

        # Final layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.top_splitter)

        hr = QFrame()
        hr.setFrameShape(QFrame.HLine)
        hr.setStyleSheet("color: #444444;")
        main_layout.addWidget(hr, stretch=0)

        main_layout.addWidget(bottom_cards_widget, stretch=0)
        main_layout.addSpacing(8)

        self.setLayout(main_layout)

    def _on_import_clicked(self):
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Select a Video File",
            "",
            "Video Files (*.mp4 *.MP4 *.avi *.AVI *.mov *.MOV);;All Files (*.*)"
        )
        if fname:
            add_to_recent_projects(fname)
            self.import_callback(fname)

    def _on_continue_clicked(self):
        # You might want to open a file dialog for .sfproj or similar
        self.continue_callback()

    def _open_recent_project(self, video_path):
        if os.path.exists(video_path):
            add_to_recent_projects(video_path)
            self.continue_callback()
            self.import_callback(video_path)


# ─── EditorPage (unchanged except for background) ───

class EditorPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #1E1E1E;")
        self.cap = None
        self.rotation_angle = 0
        self.total_frames = 0
        self.current_frame_idx = 0
        self.fps = 30.0
        self.video_path = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.thumbnail_labels = []
        self.thumbnail_frame_indices = []

        self._build_ui()

    def _build_ui(self):
        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("background-color: #000000; border: 1px solid #444444;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.gesture_list = QListWidget()
        self.gesture_list.setFixedWidth(240)
        self.gesture_list.setStyleSheet("""
            QListWidget {
                background-color: #2A2A2A;
                color: #FFFFFF;
                border: 1px solid #444444;
                font-size: 12px;
            }
            QListWidget::item:selected {
                background-color: #3A3A3A;
            }
            QListWidget::item:hover {
                background-color: #3A3A3A;
            }
        """)
        self.gesture_list.itemClicked.connect(self.jump_to_gesture_time)

        video_splitter = QSplitter(Qt.Horizontal)
        video_splitter.addWidget(self.video_label)
        video_splitter.addWidget(self.gesture_list)
        video_splitter.setSizes([800, 240])
        video_splitter.setHandleWidth(1)

        self.thumbnail_widget = QWidget()
        self.thumbnail_layout = QHBoxLayout(self.thumbnail_widget)
        self.thumbnail_layout.setSpacing(4)
        self.thumbnail_layout.setContentsMargins(0, 0, 0, 0)

        self.thumbnail_scroll = QScrollArea()
        self.thumbnail_scroll.setWidgetResizable(True)
        self.thumbnail_scroll.setFrameShape(QFrame.NoFrame)
        self.thumbnail_scroll.setWidget(self.thumbnail_widget)
        self.thumbnail_scroll.setFixedHeight(80)
        self.thumbnail_scroll.setStyleSheet("background-color: #1E1E1E;")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderPressed.connect(self.pause_video)
        self.slider.sliderMoved.connect(self.preview_slider)
        self.slider.sliderReleased.connect(self.seek_slider)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #444444;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #8E44AD;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #FFFFFF;
                border: 1px solid #8E44AD;
                width: 12px;
                height: 12px;
                margin: -3px 0;
                border-radius: 6px;
            }
        """)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("color: #DDDDDD; font-size: 12px;")
        self.time_label.setFixedHeight(20)

        btn_style = """
            QPushButton {
                background-color: #8E44AD;
                border: none;
                color: #FFFFFF;
                font-size: 12px;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #7D3C98;
            }
            QPushButton:disabled {
                background-color: #333333;
                color: #777777;
            }
        """
        self.open_button = QPushButton("Import Video")
        self.open_button.setStyleSheet(btn_style)
        self.open_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.open_button.clicked.connect(self.load_video)

        self.toggle_button = QPushButton("Play")
        self.toggle_button.setStyleSheet(btn_style)
        self.toggle_button.setEnabled(False)
        self.toggle_button.clicked.connect(self.toggle_play_pause)

        self.detect_button = QPushButton("Detect Gestures")
        self.detect_button.setStyleSheet(btn_style)
        self.detect_button.setEnabled(False)
        self.detect_button.clicked.connect(self.detect_gestures)

        self.blur_button = QPushButton("Blur Frame")
        self.blur_button.setStyleSheet(btn_style)
        self.blur_button.setEnabled(False)
        self.blur_button.clicked.connect(self.blur_current_frame)

        bottom_vbox = QVBoxLayout()
        bottom_vbox.setContentsMargins(8, 8, 8, 8)
        bottom_vbox.setSpacing(8)
        bottom_vbox.addWidget(self.thumbnail_scroll)

        tlayout = QHBoxLayout()
        tlayout.setContentsMargins(0, 0, 0, 0)
        tlayout.setSpacing(8)
        tlayout.addWidget(self.slider)
        tlayout.addWidget(self.time_label)
        bottom_vbox.addLayout(tlayout)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(12)
        btn_layout.addWidget(self.open_button)
        btn_layout.addWidget(self.toggle_button)
        btn_layout.addWidget(self.detect_button)
        btn_layout.addStretch()
        bottom_vbox.addLayout(btn_layout)

        bottom_vbox.addWidget(self.blur_button, alignment=Qt.AlignLeft)

        bottom_container = QWidget()
        bottom_container.setLayout(bottom_vbox)

        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(video_splitter)
        main_splitter.addWidget(bottom_container)
        main_splitter.setSizes([500, 300])
        main_splitter.setHandleWidth(1)

        editor_layout = QVBoxLayout()
        editor_layout.setContentsMargins(8, 8, 8, 8)
        editor_layout.setSpacing(6)
        editor_layout.addWidget(main_splitter)

        self.setLayout(editor_layout)

    def load_video(self, video_path=None):
        if video_path is None:
            video_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select a Video File",
                "",
                "Video Files (*.mp4 *.MP4 *.avi *.AVI *.mov *.MOV);;All Files (*.*)"
            )
            if not video_path:
                return

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.rotation_angle = get_video_rotation(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self.slider.setMaximum(max(0, self.total_frames - 1))
        self.slider.setEnabled(True)
        self.toggle_button.setEnabled(True)
        self.detect_button.setEnabled(True)
        self.blur_button.setEnabled(True)
        base_name = os.path.basename(video_path)
        self.setWindowTitle(f"StopFilming – Editing: {base_name}")

        for lbl in self.thumbnail_labels:
            self.thumbnail_layout.removeWidget(lbl)
            lbl.deleteLater()
        self.thumbnail_labels.clear()
        self.thumbnail_frame_indices.clear()

        thumbs = generate_thumbnails(video_path, self.total_frames, self.rotation_angle, num_thumbs=12)
        for frame_idx, thumb_img in thumbs:
            h, w, _ = thumb_img.shape
            bytes_per_line = w * 3
            qimg = QImage(thumb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            thumb_label = QLabel()
            thumb_label.setPixmap(pixmap)
            thumb_label.setFixedHeight(60)
            thumb_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
            thumb_label.setAlignment(Qt.AlignCenter)
            thumb_label.setCursor(QCursor(Qt.PointingHandCursor))
            thumb_label.setStyleSheet("""
                QLabel {
                    border: 1px solid #444444;
                    border-radius: 4px;
                }
                QLabel:hover {
                    border-color: #8E44AD;
                }
            """)
            thumb_label.mousePressEvent = self.make_thumb_click_handler(frame_idx)
            self.thumbnail_layout.addWidget(thumb_label)
            self.thumbnail_labels.append(thumb_label)
            self.thumbnail_frame_indices.append(frame_idx)

        self.show_frame(0)

    def make_thumb_click_handler(self, frame_idx):
        def handler(event):
            self.pause_video()
            self.show_frame(frame_idx)
        return handler

    def show_frame(self, frame_idx):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx = frame_idx
            self.display_frame(frame)
            self.slider.blockSignals(True)
            self.slider.setValue(frame_idx)
            self.slider.blockSignals(False)
            self.update_time_label()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.display_frame(frame)
            self.slider.setValue(self.current_frame_idx)
            self.update_time_label()
        else:
            self.timer.stop()

    def seek_slider(self):
        self.show_frame(self.slider.value())

    def preview_slider(self, value):
        self.show_frame(value)

    def update_time_label(self):
        current_sec = self.current_frame_idx / self.fps
        total_sec = self.total_frames / self.fps

        def fmt(t):
            return f"{int(t // 60):02}:{int(t % 60):02}"

        self.time_label.setText(f"{fmt(current_sec)} / {fmt(total_sec)}")

    def display_frame(self, frame):
        if self.rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def toggle_play_pause(self):
        if not self.cap:
            return
        if self.timer.isActive():
            self.pause_video()
            self.toggle_button.setText("Play")
        else:
            self.play_video()
            self.toggle_button.setText("Pause")

    def play_video(self):
        if self.cap:
            self.timer.start(int(1000 / self.fps))

    def pause_video(self):
        self.timer.stop()

    def detect_gestures(self):
        if not self.video_path:
            return
        detector = WaveDetector(self.video_path, self.fps)
        detected_timestamps = detector.detect_wave_timestamps()
        self.gesture_list.clear()
        for t in detected_timestamps:
            time_str = f"{int(t // 60):02}:{int(t % 60):02}.{int((t % 1) * 1000):03}"
            item = QListWidgetItem(f"👋 {time_str}")
            item.setData(Qt.UserRole, t)
            self.gesture_list.addItem(item)

    def jump_to_gesture_time(self, item):
        timestamp = item.data(Qt.UserRole)
        frame_idx = int(timestamp * self.fps)
        self.show_frame(frame_idx)
        self.blur_button.setEnabled(True)

    def blur_current_frame(self):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        blurred = blur_faces_in_frame(frame)
        self.display_frame(blurred)
