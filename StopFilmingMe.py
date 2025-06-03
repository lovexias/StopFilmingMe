import sys
import cv2
import subprocess
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QWidget, QHBoxLayout,
    QSlider, QScrollArea, QFrame, QSizePolicy, QListWidget, QListWidgetItem
)
from PyQt5.QtGui import QImage, QPixmap, QCursor
from PyQt5.QtCore import QTimer, Qt, QSize

import numpy as np

def get_video_rotation(path):
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream_tags=rotate",
            "-of", "json", path
        ]
        output = subprocess.check_output(cmd).decode("utf-8")
        ffprobe_data = json.loads(output)
        tags = ffprobe_data.get("streams", [{}])[0].get("tags", {})
        return int(tags.get("rotate", 0))
    except:
        return 0


class OpenCVVideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StopFilming - OpenCV Editor")
        self.setGeometry(100, 100, 960, 720)

        self.cap = None
        self.rotation_angle = 0
        self.total_frames = 0
        self.current_frame_idx = 0
        self.fps = 30
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.thumbnail_labels = []
        self.thumbnail_frame_indices = []

        self.video_path = None
        self.blurred_frames = {}  # key: frame_idx, value: True if blurred
        self.frame_cache = {}  # key: frame_idx, value: QImage of blurred or original frame

        # Video display
        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("background-color: black")
        self.video_label.setAlignment(Qt.AlignCenter)

        # Gesture timestamps list
        self.gesture_list = QListWidget()
        self.gesture_list.setFixedWidth(150)
        self.gesture_list.setStyleSheet("font-size: 11px; padding: 5px;")

        # Blur Button
        self.blur_button = QPushButton("Blur")
        self.blur_button.setVisible(False)
        self.blur_button.clicked.connect(self.blur_current_frame)

        # Time label
        self.time_label = QLabel("00:00 / 00:00")

        # Timeline slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderPressed.connect(self.pause_video)
        self.slider.sliderMoved.connect(self.preview_slider)
        self.slider.sliderReleased.connect(self.seek_slider)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: #eee;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background: #66afe9;
                border: 1px solid #4a90e2;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: white;
                border: 1px solid #4a90e2;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)

        # Thumbnail strip
        self.thumbnail_widget = QWidget()
        self.thumbnail_layout = QHBoxLayout(self.thumbnail_widget)
        self.thumbnail_layout.setSpacing(5)
        self.thumbnail_layout.setContentsMargins(0, 0, 0, 0)

        self.thumbnail_scroll = QScrollArea()
        self.thumbnail_scroll.setWidgetResizable(True)
        self.thumbnail_scroll.setFrameShape(QFrame.NoFrame)
        self.thumbnail_scroll.setWidget(self.thumbnail_widget)
        self.thumbnail_scroll.setFixedHeight(70)

        # Buttons
        self.open_button = QPushButton("Import Video")
        self.toggle_button = QPushButton("Play")
        self.detect_button = QPushButton("Enable Gesture Detection")

        self.toggle_button.clicked.connect(self.toggle_play_pause)
        self.open_button.clicked.connect(self.load_video)
        self.detect_button.clicked.connect(self.detect_gestures)

        # Layouts
        layout = QVBoxLayout()
        video_row = QHBoxLayout()
        video_row.addWidget(self.video_label)
        video_row.addWidget(self.gesture_list)

        layout.addLayout(video_row)
        layout.addWidget(self.thumbnail_scroll)
        layout.addWidget(self.slider)
        layout.addWidget(self.time_label)

        controls = QHBoxLayout()
        controls.addWidget(self.open_button)
        controls.addWidget(self.toggle_button)
        controls.addWidget(self.detect_button)
        layout.addLayout(controls)
        layout.addWidget(self.blur_button)

        layout.setSpacing(5)
        layout.setContentsMargins(10, 5, 10, 5)
        self.time_label.setFixedHeight(20)
        self.time_label.setStyleSheet("margin: 0px; padding: 0px; font-size: 11px;")
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if not video_path:
            return

        self.video_path = video_path  # <-- store the path here
        self.cap = cv2.VideoCapture(video_path)
        self.rotation_angle = get_video_rotation(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.slider.setMaximum(self.total_frames - 1)
        self.slider.setEnabled(True)
        self.setWindowTitle(f"Loaded: {video_path.split('/')[-1]}")
        self.generate_thumbnails(video_path)
        self.show_frame(0)

    def generate_thumbnails(self, video_path, num_thumbs=10):
        for label in self.thumbnail_labels:
            self.thumbnail_layout.removeWidget(label)
            label.deleteLater()
        self.thumbnail_labels.clear()
        self.thumbnail_frame_indices.clear()

        thumb_size = (80, 45)
        step = max(1, self.total_frames // num_thumbs)
        cap = cv2.VideoCapture(video_path)

        for i in range(num_thumbs):
            frame_idx = i * step
            self.thumbnail_frame_indices.append(frame_idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            if self.rotation_angle == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation_angle == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotation_angle == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            thumb = cv2.resize(frame_rgb, thumb_size)
            h, w, ch = thumb.shape
            qimg = QImage(thumb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            thumb_label = QLabel()
            thumb_label.setPixmap(pixmap)
            thumb_label.setFixedHeight(45)
            thumb_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
            thumb_label.setAlignment(Qt.AlignCenter)
            thumb_label.setCursor(QCursor(Qt.PointingHandCursor))
            thumb_label.mousePressEvent = self.make_thumb_click_handler(frame_idx)

            self.thumbnail_layout.addWidget(thumb_label)
            self.thumbnail_labels.append(thumb_label)

        cap.release()

    def make_thumb_click_handler(self, frame_idx):
        def handler(event):
            self.pause_video()
            self.show_frame(frame_idx)
        return handler

    def show_frame(self, frame_idx):
        self.current_frame_idx = frame_idx
        if frame_idx in self.frame_cache:
            qt_image = self.frame_cache[frame_idx]
            self.display_qimage(qt_image)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret:
                qt_image = self.convert_frame_to_qimage(frame)
                self.frame_cache[frame_idx] = qt_image
                self.display_qimage(qt_image)

        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)
        self.update_time_label()

    def play_video(self):
        if self.cap:
            self.timer.start(int(1000 / self.fps))

    def pause_video(self):
        self.timer.stop()

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

        def fmt(t): return f"{int(t // 60):02}:{int(t % 60):02}"

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

        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def toggle_play_pause(self):
        if self.timer.isActive():
            self.pause_video()
            self.toggle_button.setText("Play")
        else:
            self.play_video()
            self.toggle_button.setText("Pause")

    def detect_gestures(self):
        if not self.video_path:
            return

        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
        self.last_x = None
        self.movement_history = []
        self.detected_timestamps = []

        cap = cv2.VideoCapture(self.video_path)  # ✅ fixed path reference

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_time = 1.0 / fps

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            timestamp = frame_count * frame_time
            frame_count += 1

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.check_wave(hand_landmarks, timestamp)

        cap.release()
        hands.close()

        self.gesture_list.clear()
        self.gesture_list.itemClicked.connect(self.jump_to_gesture_time)
        for t in self.detected_timestamps:
            time_str = f"{int(t // 60):02}:{int(t % 60):02}.{int((t % 1) * 1000):03}"
            item = QListWidgetItem(f"👋 {time_str}")
            item.setData(Qt.UserRole, t)
            self.gesture_list.addItem(item)

    def check_wave(self, hand_landmarks, timestamp):
        media_x = sum([lm.x for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
        direction = None
        if self.last_x is not None:
            if media_x < self.last_x - 0.02:
                direction = "left"
            elif media_x > self.last_x + 0.02:
                direction = "right"
            if direction and (not self.movement_history or self.movement_history[-1][0] != direction):
                self.movement_history.append((direction, timestamp))
        self.movement_history = [(d, t) for d, t in self.movement_history if timestamp - t <= 1.0]
        if len(self.movement_history) >= 4:
            self.detected_timestamps.append(timestamp)
            self.movement_history.clear()
        self.last_x = media_x

    def jump_to_gesture_time(self, item):
        timestamp = item.data(Qt.UserRole)
        frame_idx = int(timestamp * self.fps)
        self.show_frame(frame_idx)
        self.blur_button.setVisible(True)

        # Update button text depending on current blur state
        if self.blurred_frames.get(frame_idx, False):
            self.blur_button.setText("Unblur")
        else:
            self.blur_button.setText("Blur")

    def blur_current_frame(self):
        frame_idx = self.current_frame_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return

        if self.blurred_frames.get(frame_idx, False):
            # UNBLUR: store original QImage in cache and display
            qt_image = self.convert_frame_to_qimage(frame)
            self.frame_cache[frame_idx] = qt_image
            self.display_qimage(qt_image)
            self.blurred_frames[frame_idx] = False
            self.blur_button.setText("Blur")
        else:
            # BLUR using MediaPipe
            import mediapipe as mp
            mp_face = mp.solutions.face_detection
            face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                mask = np.zeros_like(frame)
                height, width, _ = frame.shape
                for detection in results.detections:
                    box = detection.location_data.relative_bounding_box
                    x = int(box.xmin * width)
                    y = int(box.ymin * height)
                    w_box = int(box.width * width)
                    h_box = int(box.height * height)
                    x, y = max(0, x), max(0, y)
                    w_box = min(w_box, width - x)
                    h_box = min(h_box, height - y)
                    mask[y:y + h_box, x:x + w_box] = 255

                blurred = cv2.GaussianBlur(frame, (55, 55), 0)
                frame = np.where(mask == 255, blurred, frame)

            qt_image = self.convert_frame_to_qimage(frame)
            self.frame_cache[frame_idx] = qt_image
            self.display_qimage(qt_image)


    def convert_frame_to_qimage(self, frame):
        if self.rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        return QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
    
    def display_qimage(self, qimg):
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OpenCVVideoPlayer()
    window.show()
    sys.exit(app.exec_())
