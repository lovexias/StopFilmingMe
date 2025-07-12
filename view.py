# ─── Enhanced Video Editor Interface ────────────────────────────────────────────

import cv2
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QSlider, QPushButton,
    QListWidget, QListWidgetItem, QScrollArea, QLabel, QFrame, QSizePolicy,
    QProgressDialog, QApplication, QMainWindow, QMenuBar, QAction, QStatusBar,
    QToolBar, QSpacerItem, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QPoint, QTimer, QSize, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import (
    QImage, QPixmap, QCursor, QPainter, QColor, QFont, QIcon, QPalette,
    QLinearGradient, QBrush, QPen, QPolygon, QFontMetrics
)
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt


class ModernTitleBar(QWidget):
    """Custom title bar with window controls and modern styling"""
    
    minimizeClicked = pyqtSignal()
    maximizeClicked = pyqtSignal()
    closeClicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4A5568, stop:1 #2D3748);
                border-bottom: 1px solid #1A202C;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 0, 5, 0)
        layout.setSpacing(0)
        
        # App title
        self.title_label = QLabel("Video Gesture Editor")
        f = QFont("Segoe UI", 14, QFont.Bold)
        self.title_label.setFont(f)
        self.title_label.setStyleSheet("""
            color: #E2E8F0;
            font-size: 14px;
            font-weight: 600;
            padding: 0 10px;
        """)
        layout.addWidget(self.title_label)
        
        layout.addStretch()
        
        # Window controls
        self.minimize_btn = self.create_window_button("−", "#4A5568", "#5A6578")
        self.maximize_btn = self.create_window_button("□", "#4A5568", "#5A6578")
        self.close_btn = self.create_window_button("×", "#E53E3E", "#C53030")
        
        self.minimize_btn.clicked.connect(self.minimizeClicked.emit)
        self.maximize_btn.clicked.connect(self.maximizeClicked.emit)
        self.close_btn.clicked.connect(self.closeClicked.emit)
        
        layout.addWidget(self.minimize_btn)
        layout.addWidget(self.maximize_btn)
        layout.addWidget(self.close_btn)
    
    def create_window_button(self, text, bg_color, hover_color):
        btn = QPushButton(text)
        btn.setFixedSize(35, 30)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                font-size: 16px;
                font-weight: bold;
                margin: 2px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {bg_color};
            }}
        """)
        return btn


class EnhancedTimeRuler(QWidget):
    """Enhanced time ruler with gradient background and smooth animations"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame = 0
        self.setMinimumHeight(35)
        self.setAttribute(Qt.WA_TranslucentBackground)

        
        # Add drop shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

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
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Gradient background
        gradient = QLinearGradient(0, 0, 0, h)
        gradient.setColorAt(0, QColor("#2D3748"))
        gradient.setColorAt(1, QColor("#1A202C"))
        painter.fillRect(0, 0, w, h, gradient)
        
        if self.total_frames <= 0 or self.fps <= 0:
            return
            
        total_seconds = self.total_frames / self.fps
        num_ticks = min(20, int(total_seconds))
        
        if num_ticks == 0:
            return
            
        interval_sec = total_seconds / num_ticks

        # Draw tick marks and labels
        painter.setPen(QPen(QColor("#A0AEC0"), 1))
        font = QFont("Arial", 8)
        painter.setFont(font)
        
        for i in range(num_ticks + 1):
            sec = i * interval_sec
            x = int((sec / total_seconds) * w)
            
            # Major tick
            painter.drawLine(x, h - 12, x, h)
            
            # Time label
            mm = int(sec // 60)
            ss = int(sec % 60)
            label = f"{mm:02}:{ss:02}"
            
            fm = QFontMetrics(font)
            text_width = fm.width(label)
            painter.setPen(QColor("#E2E8F0"))
            painter.drawText(x - text_width // 2, h - 15, label)
            painter.setPen(QPen(QColor("#A0AEC0"), 1))

        # Current position indicator
        cur_sec = self.current_frame / self.fps
        if cur_sec > total_seconds:
            cur_sec = total_seconds
        x_cur = int((cur_sec / total_seconds) * w)
        x_cur = max(0, min(x_cur, w))

        # Draw current position with glow effect
        painter.setPen(QPen(QColor("#4FD1C7"), 3))
        painter.drawLine(x_cur, 0, x_cur, h)
        
        # Draw triangle indicator
        triangle = QPolygon([
            QPoint(x_cur - 6, 0),
            QPoint(x_cur + 6, 0),
            QPoint(x_cur, 12)
        ])

        painter.setBrush(QBrush(QColor("#4FD1C7")))
        painter.setPen(QPen(QColor("#38B2AC"), 2))
        painter.drawPolygon(triangle)


class ModernButton(QPushButton):
    """Enhanced button with modern styling and hover effects"""
    
    def __init__(self, text, button_type="primary", parent=None):
        super().__init__(text, parent)
        self.button_type = button_type
        self.setFixedHeight(36)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.apply_style()
        
        # Add drop shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
    
    def apply_style(self):
        if self.button_type == "primary":
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4299E1, stop:1 #3182CE);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 8px 16px;
                    font-size: 13px;
                    font-weight: 600;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4FD1C7, stop:1 #38B2AC);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #2C5282, stop:1 #2A4A6B);
                }
                QPushButton:disabled {
                    background: #4A5568;
                    color: #A0AEC0;
                }
            """)
        elif self.button_type == "secondary":
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4A5568, stop:1 #2D3748);
                    color: #E2E8F0;
                    border: 1px solid #718096;
                    border-radius: 8px;
                    padding: 8px 16px;
                    font-size: 13px;
                    font-weight: 600;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #718096, stop:1 #4A5568);
                    border-color: #A0AEC0;
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #2D3748, stop:1 #1A202C);
                }
                QPushButton:disabled {
                    background: #2D3748;
                    color: #718096;
                    border-color: #4A5568;
                }
            """)
        elif self.button_type == "danger":
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #F56565, stop:1 #E53E3E);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 8px 16px;
                    font-size: 13px;
                    font-weight: 600;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #FC8181, stop:1 #F56565);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #C53030, stop:1 #9C1A1A);
                }
                QPushButton:disabled {
                    background: #4A5568;
                    color: #A0AEC0;
                }
            """)


class ModernSlider(QSlider):
    """Enhanced slider with modern styling"""
    
    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2D3748, stop:1 #1A202C);
                height: 8px;
                border-radius: 4px;
                border: 1px solid #4A5568;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4FD1C7, stop:1 #38B2AC);
                border: 2px solid #2D3748;
                width: 20px;
                height: 20px;
                margin: -7px 0;
                border-radius: 12px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #68D391, stop:1 #48BB78);
            }
            QSlider::handle:horizontal:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #38B2AC, stop:1 #319795);
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4FD1C7, stop:1 #38B2AC);
                border-radius: 4px;
            }
        """)


class EnhancedVideoEditor(QMainWindow):
    """Main window with custom title bar and modern interface"""
    
    # All the same signals as EditorPanel
    importRequested = pyqtSignal()
    playToggled = pyqtSignal(bool)
    frameChanged = pyqtSignal(int)
    detectRequested = pyqtSignal()
    blurRequested = pyqtSignal(int)
    thumbnailClicked = pyqtSignal(int)
    gestureItemClicked = pyqtSignal(int)
    exportRequested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setMinimumSize(1200, 800)
        self.resize(1600, 1000)
        
        # Initialize state
        self.core_has_video = False
        self.cap = None
        self.video_path = None
        self.rotation_angle = 0
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame_idx = 0
        self.is_playing = False
        self.is_maximized = False
        self.old_geometry = None
        
        # For window dragging
        self.mouse_press_pos = None
        self.mouse_move_pos = None
        
        self.setup_ui()
        self.apply_global_styles()
        
        # Setup timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer_tick)
    
    def setup_ui(self):
        """Setup the main interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)    # ← allow rounded corners
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")     # ← give it an ID
        self.setCentralWidget(central_widget)
        
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Custom title bar
        self.title_bar = ModernTitleBar()
        self.title_bar.minimizeClicked.connect(self.showMinimized)
        self.title_bar.maximizeClicked.connect(self.toggle_maximize)
        self.title_bar.closeClicked.connect(self.close)
        main_layout.addWidget(self.title_bar)
        
        # Main content area
        content_frame = QFrame()
        content_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2D3748, stop:1 #1A202C);
            }
        """)
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(10)
        
        # Video and controls area
        self.setup_video_area(content_layout)
        
        # Bottom timeline area
        self.setup_timeline_area(content_layout)
        
        main_layout.addWidget(content_frame)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background: #1A202C;
                color: #A0AEC0;
                border-top: 1px solid #2D3748;
                padding: 5px 10px;
            }
        """)
        self.status_bar.showMessage("Ready")
        self.setStatusBar(self.status_bar)
    
    def setup_ui(self):
        central = QWidget()
        central.setObjectName("centralWidget")     # <-- important
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0,0,0,0)
        # … add your ModernTitleBar, menuBar(), contentFrame, statusBar, etc.

    
    def setup_video_area(self, parent_layout):
        """Setup the main video editing area"""
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4A5568, stop:1 #2D3748);
                border-radius: 1px;
            }
            QSplitter::handle:hover {
                background: #4FD1C7;
            }
        """)
        
        # Left side - Video player
        self.setup_video_player(splitter)
        
        # Right side - Controls panel
        self.setup_controls_panel(splitter)
        
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        
        parent_layout.addWidget(splitter, stretch=1)
    
    def setup_video_player(self, parent_splitter):
        """Setup the video player area"""
        video_frame = QFrame()
        video_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4A5568, stop:1 #2D3748);
                border: 1px solid #718096;
                border-radius: 12px;
            }
        """)
        
        # Add drop shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 5)
        video_frame.setGraphicsEffect(shadow)
        
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(15, 15, 15, 15)
        video_layout.setSpacing(10)
        
        # Video display
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_display.setStyleSheet("""
            QLabel {
                background: #1A202C;
                border: 2px solid #4A5568;
                border-radius: 8px;
                min-height: 400px;
            }
        """)
        video_layout.addWidget(self.video_display)
        
        # Import overlay
        self.create_import_overlay(video_frame)
        
        # Video controls
        self.setup_video_controls(video_layout)
        
        parent_splitter.addWidget(video_frame)
    
    def create_import_overlay(self, parent):
        """Create the import video overlay"""
        self.import_popup = QFrame(parent)
        self.import_popup.setStyleSheet("""
            QFrame {
                background: rgba(26, 32, 44, 200);
                border: 2px dashed #4FD1C7;
                border-radius: 12px;
            }
        """)
        
        overlay_layout = QVBoxLayout(self.import_popup)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.addStretch()
        
        # Import icon (using text for now)
        icon_label = QLabel("📹")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("""
            QLabel {
                color: #4FD1C7;
                font-size: 48px;
                background: none;
                border: none;
                padding: 20px;
            }
        """)
        overlay_layout.addWidget(icon_label)
        
        # Import button
        self.import_btn = ModernButton("Import Video", "primary")
        self.import_btn.setFixedSize(200, 50)
        self.import_btn.clicked.connect(self.importRequested.emit)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.import_btn)
        btn_layout.addStretch()
        overlay_layout.addLayout(btn_layout)
        
        # Help text
        help_label = QLabel("Click to import a video file or drag & drop here")
        help_label.setAlignment(Qt.AlignCenter)
        help_label.setStyleSheet("""
            QLabel {
                color: #A0AEC0;
                font-size: 14px;
                background: none;
                border: none;
                padding: 10px;
            }
        """)
        overlay_layout.addWidget(help_label)
        overlay_layout.addStretch()
        
        self.import_popup.show()
    
    def setup_video_controls(self, parent_layout):
        """Setup video playback controls"""
        # Progress slider
        self.slider = ModernSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(lambda v: self.frameChanged.emit(v))
        parent_layout.addWidget(self.slider)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        
        controls_layout.addStretch()
        
        self.play_btn = ModernButton("▶ Play", "primary")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._on_toggle_clicked)
        controls_layout.addWidget(self.play_btn)
        
        controls_layout.addStretch()
        
        parent_layout.addLayout(controls_layout)
    
    def setup_controls_panel(self, parent_splitter):
        """Setup the right-side controls panel"""
        panel_frame = QFrame()
        panel_frame.setFixedWidth(300)
        panel_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4A5568, stop:1 #2D3748);
                border: 1px solid #718096;
                border-radius: 12px;
            }
        """)
        
        # Add drop shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 5)
        panel_frame.setGraphicsEffect(shadow)
        
        panel_layout = QVBoxLayout(panel_frame)
        panel_layout.setContentsMargins(20, 20, 20, 20)
        panel_layout.setSpacing(15)
        
        # Actions section
        actions_label = QLabel("Actions")
        actions_label.setStyleSheet("""
            QLabel {
                color: #E2E8F0;
                font-size: 16px;
                font-weight: 700;
                padding: 10px 0;
                border-bottom: 2px solid #4FD1C7;
            }
        """)
        panel_layout.addWidget(actions_label)
        
        # Action buttons
        self.detect_btn = ModernButton("🔍 Detect Gestures", "secondary")
        self.detect_btn.setEnabled(False)
        self.detect_btn.clicked.connect(self.detectRequested.emit)
        panel_layout.addWidget(self.detect_btn)
        
        self.blur_btn = ModernButton("🔒 Blur Person", "secondary")
        self.blur_btn.setEnabled(False)
        self.blur_btn.clicked.connect(lambda: self.blurRequested.emit(self.current_frame_idx))
        panel_layout.addWidget(self.blur_btn)
        
        self.export_btn = ModernButton("📤 Export Video", "primary")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.exportRequested.emit)
        panel_layout.addWidget(self.export_btn)
        
        panel_layout.addSpacing(20)
        
        # Gestures section
        gestures_label = QLabel("Detected Gestures")
        gestures_label.setStyleSheet("""
            QLabel {
                color: #E2E8F0;
                font-size: 16px;
                font-weight: 700;
                padding: 10px 0;
                border-bottom: 2px solid #4FD1C7;
            }
        """)
        panel_layout.addWidget(gestures_label)
        
        self.gesture_list = QListWidget()
        self.gesture_list.setStyleSheet("""
            QListWidget {
                background: #1A202C;
                border: 1px solid #4A5568;
                border-radius: 8px;
                padding: 8px;
                color: #E2E8F0;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #2D3748;
                border-radius: 4px;
                margin: 2px 0;
            }
            QListWidget::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4FD1C7, stop:1 #38B2AC);
                color: white;
            }
            QListWidget::item:hover {
                background: #2D3748;
            }
        """)
        self.gesture_list.itemClicked.connect(
            lambda item: self.gestureItemClicked.emit(item.data(Qt.UserRole)[0])
        )
        panel_layout.addWidget(self.gesture_list)
        
        panel_layout.addStretch()
        
        parent_splitter.addWidget(panel_frame)
    
    def setup_timeline_area(self, parent_layout):
        """Setup the bottom timeline area"""
        timeline_frame = QFrame()
        timeline_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4A5568, stop:1 #2D3748);
                border: 1px solid #718096;
                border-radius: 12px;
            }
        """)
        
        # Add drop shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 3)
        timeline_frame.setGraphicsEffect(shadow)
        
        timeline_layout = QVBoxLayout(timeline_frame)
        timeline_layout.setContentsMargins(15, 10, 15, 10)
        timeline_layout.setSpacing(8)
        
        # Time ruler
        self.time_ruler = EnhancedTimeRuler()
        timeline_layout.addWidget(self.time_ruler)
        
        # Thumbnails
        self.thumbnail_scroll = QScrollArea()
        # give it at least 150px, but allow it to grow
        self.thumbnail_scroll.setMinimumHeight(150)
        self.thumbnail_scroll.setMaximumHeight(300)      # optional cap
        self.thumbnail_scroll.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        # when you add it, allow it to take “some” of the extra space
        parent_layout.addWidget(timeline_frame, stretch=1)


        self.thumbnail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.thumbnail_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.thumbnail_scroll.setWidgetResizable(True)
        self.thumbnail_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #4A5568;
                border-radius: 8px;
                background: #1A202C;
            }
            QScrollBar:horizontal {
                border: none;
                background: #2D3748;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: #4FD1C7;
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #68D391;
            }
        """)
        
        thumb_container = QWidget()
        self.thumbnail_layout = QHBoxLayout(thumb_container)
        self.thumbnail_layout.setContentsMargins(10, 10, 10, 10)
        self.thumbnail_layout.setSpacing(8)
        self.thumbnail_scroll.setWidget(thumb_container)
        
        timeline_layout.addWidget(self.thumbnail_scroll)
        
        

        # when you add it, allow it to take “some” of the extra space
        parent_layout.addWidget(timeline_frame, stretch=1)

    
    def apply_global_styles(self):
        """Apply global application styles"""
        self.setStyleSheet("""
            * {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            }
        """)
        self.setStyleSheet("""
            QWidget#centralWidget {
                background-color: #1A202C;
                border-radius: 8px;
            }
        """)
        

    
    # Window management methods
    def toggle_maximize(self):
        """Toggle between maximized and normal window state"""
        if self.is_maximized:
            self.setGeometry(self.old_geometry)
            self.is_maximized = False
            self.title_bar.maximize_btn.setText("□")
        else:
            self.old_geometry = self.geometry()
            screen = QApplication.desktop().screenGeometry()
            self.setGeometry(screen)
            self.is_maximized = True
            self.title_bar.maximize_btn.setText("❐")
    
    def mousePressEvent(self, event):
        """Handle mouse press for window dragging"""
        if event.button() == Qt.LeftButton and event.y() < 40:  # Only in title bar area
            self.mouse_press_pos = event.globalPos()
            self.mouse_move_pos = event.globalPos()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for window dragging"""
        if event.buttons() == Qt.LeftButton and self.mouse_press_pos:
            current_pos = event.globalPos()
            diff = current_pos - self.mouse_move_pos
            new_pos = self.pos() + diff
            self.move(new_pos)
            self.mouse_move_pos = current_pos
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.LeftButton:
            self.mouse_press_pos = None
            self.mouse_move_pos = None
    
    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        if hasattr(self, 'import_popup'):
            # Update import overlay size and position
            parent_rect = self.video_display.geometry()
            margin = 20
            self.import_popup.setGeometry(
                parent_rect.x() + margin,
                parent_rect.y() + margin,
                parent_rect.width() - 2 * margin,
                parent_rect.height() - 2 * margin
            )
    
    # Video editor interface methods (same as original EditorPanel)
    def set_video_info(self, rotation_angle, total_frames, fps):
        """Called by Controller after core.load_video()"""
        self.rotation_angle = rotation_angle
        self.total_frames = total_frames
        self.fps = fps
        self.current_frame_idx = 0

        # Enable controls
        self.slider.setMaximum(max(0, total_frames - 1))
        self.slider.setEnabled(True)
        
        self.play_btn.setEnabled(True)
        self.detect_btn.setEnabled(True)
        self.blur_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        
        self.time_ruler.setVideoInfo(total_frames, fps)
        
        # Hide import overlay
        self.core_has_video = True
        self.import_popup.hide()
        
        # Update status
        duration = total_frames / fps
        mm = int(duration // 60)
        ss = int(duration % 60)
        self.status_bar.showMessage(f"Video loaded: {total_frames} frames, {fps:.1f} FPS, Duration: {mm:02d}:{ss:02d}")
    
    def display_frame(self, img_bgr, frame_idx: int):
        """Display a video frame"""
        if img_bgr is None:
            # Clear display
            self.video_display.clear()
            self.current_frame_idx = -1
            self.time_ruler.setCurrentFrame(-1)
            return
        
        # Convert BGR to RGB and display
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        bytes_per_line = w * 3
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit display while maintaining aspect ratio
        display_size = self.video_display.size()
        pix = QPixmap.fromImage(qimg).scaled(
            display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        self.video_display.setPixmap(pix)
        self.current_frame_idx = frame_idx
        
        # Update timeline
        self.time_ruler.setCurrentFrame(frame_idx)
        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)
        
        # Update status
        time_sec = frame_idx / self.fps if self.fps > 0 else 0
        mm = int(time_sec // 60)
        ss = int(time_sec % 60)
        self.status_bar.showMessage(f"Frame {frame_idx + 1}/{self.total_frames} - {mm:02d}:{ss:02d}")
    
    def clear_thumbnails(self):
        """Remove all thumbnail widgets"""
        if hasattr(self, "thumbnail_labels"):
            for thumb in self.thumbnail_labels:
                self.thumbnail_layout.removeWidget(thumb)
                thumb.deleteLater()
            self.thumbnail_labels = []
            self.thumbnail_frame_indices = []
    
    def add_thumbnails(self, thumbs):
        """Add thumbnail images to the timeline"""
        if not hasattr(self, "thumbnail_labels"):
            self.thumbnail_labels = []
            self.thumbnail_frame_indices = []
        
        for idx, thumb_rgb in thumbs:
            h, w, _ = thumb_rgb.shape
            bytes_per_line = w * 3
            qimg = QImage(thumb_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaledToHeight(80, Qt.SmoothTransformation)
            
            thumb_label = QLabel()
            thumb_label.setPixmap(pix)
            thumb_label.setFixedSize(QSize(pix.width(), pix.height()))
            thumb_label.setCursor(QCursor(Qt.PointingHandCursor))
            thumb_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #4A5568;
                    border-radius: 6px;
                    padding: 2px;
                    background: #2D3748;
                }
                QLabel:hover {
                    border-color: #4FD1C7;
                    background: #4A5568;
                }
            """)
            
            # Add click handler
            thumb_label.mousePressEvent = lambda e, i=idx: self.thumbnailClicked.emit(i)
            
            # Add drop shadow
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(5)
            shadow.setColor(QColor(0, 0, 0, 100))
            shadow.setOffset(0, 2)
            thumb_label.setGraphicsEffect(shadow)
            
            self.thumbnail_layout.addWidget(thumb_label)
            self.thumbnail_labels.append(thumb_label)
            self.thumbnail_frame_indices.append(idx)
    
    def add_gesture_items(self, segment_starts):
        """Add detected gestures to the list"""
        self.gesture_list.clear()
        
        for i, idx in enumerate(segment_starts):
            t = idx / self.fps if self.fps > 0 else 0
            mm = int(t // 60)
            ss = int(t % 60)
            msec = int((t - int(t)) * 1000)
            time_str = f"{mm:02d}:{ss:02d}.{msec:03d}"
            
            item = QListWidgetItem(f"✋ Gesture {i+1} - {time_str}")
            item.setData(Qt.UserRole, idx)
            self.gesture_list.addItem(item)
        
        # Update status
        self.status_bar.showMessage(f"Detected {len(segment_starts)} gestures")
    
    def clear_markers(self):
        """Clear all detected gestures"""
        self.gesture_list.clear()
        self.blur_btn.setEnabled(False)
        self.status_bar.showMessage("Markers cleared")
    
    def show_progress_dialog(self, title="Processing", message="Please wait..."):
        """Show progress dialog"""
        self._progress = QProgressDialog(message, None, 0, 0, self)
        self._progress.setWindowTitle(title)
        self._progress.setWindowModality(Qt.WindowModal)
        self._progress.setCancelButton(None)
        self._progress.setMinimumDuration(0)
        self._progress.setAutoClose(False)
        self._progress.setAutoReset(False)
        self._progress.setStyleSheet("""
            QProgressDialog {
                background: #2D3748;
                color: #E2E8F0;
                border: 1px solid #4A5568;
                border-radius: 8px;
            }
            QProgressBar {
                border: 1px solid #4A5568;
                border-radius: 4px;
                text-align: center;
                background: #1A202C;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4FD1C7, stop:1 #38B2AC);
                border-radius: 3px;
            }
        """)
        self._progress.show()
        QApplication.processEvents()
    
    def hide_progress_dialog(self):
        """Hide progress dialog"""
        if hasattr(self, "_progress"):
            self._progress.close()
            del self._progress
    
    # Internal event handlers
    def _on_toggle_clicked(self):
        """Handle play/pause button click"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.setText("⏸ Pause")
        else:
            self.play_btn.setText("▶ Play")
        self.playToggled.emit(self.is_playing)
    
    def _on_timer_tick(self):
        """Timer tick for playback"""
        pass  # Controller manages this
    
    def closeEvent(self, event):
        """Handle window close event"""
        if hasattr(self, 'timer'):
            self.timer.stop()
        event.accept()


# Alternative: Enhanced version of original EditorPanel class
class EnhancedEditorPanel(QWidget):
    """Enhanced version of the original EditorPanel with modern styling"""
    
    # Same signals as original
    importRequested = pyqtSignal()
    playToggled = pyqtSignal(bool)
    frameChanged = pyqtSignal(int)
    detectRequested = pyqtSignal()
    blurRequested = pyqtSignal(int)
    thumbnailClicked = pyqtSignal(int)
    gestureItemClicked = pyqtSignal(int)
    exportRequested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.core_has_video = False
        
        # Internal state
        self.cap = None
        self.video_path = None
        self.rotation_angle = 0
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame_idx = 0
        self.is_playing = False
        
        self.apply_modern_styles()
        self._build_enhanced_ui()
        
        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer_tick)
    
    def apply_modern_styles(self):
        """Apply modern dark theme styles"""
        self.setStyleSheet("""
            QWidget {
                background-color: #1A202C;
                color: #E2E8F0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            }
            QLabel#video_display {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2D3748, stop:1 #1A202C);
                border: 2px solid #4A5568;
                border-radius: 12px;
            }
            QSlider::groove:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2D3748, stop:1 #1A202C);
                height: 8px;
                border-radius: 4px;
                border: 1px solid #4A5568;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4FD1C7, stop:1 #38B2AC);
                border: 2px solid #2D3748;
                width: 20px;
                height: 20px;
                margin: -7px 0;
                border-radius: 12px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #68D391, stop:1 #48BB78);
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4FD1C7, stop:1 #38B2AC);
                border-radius: 4px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4299E1, stop:1 #3182CE);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
                min-width: 100px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4FD1C7, stop:1 #38B2AC);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2C5282, stop:1 #2A4A6B);
            }
            QPushButton:disabled {
                background: #4A5568;
                color: #A0AEC0;
            }
            QPushButton#importPopupBtn {
                font-size: 18px;
                padding: 15px 30px;
                min-width: 200px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4FD1C7, stop:1 #38B2AC);
            }
            QPushButton#importPopupBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #68D391, stop:1 #48BB78);
            }
            QListWidget {
                background: #2D3748;
                border: 1px solid #4A5568;
                border-radius: 8px;
                padding: 8px;
            }
            QListWidget::item {
                padding: 12px;
                border-bottom: 1px solid #4A5568;
                border-radius: 6px;
                margin: 2px 0;
            }
            QListWidget::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4FD1C7, stop:1 #38B2AC);
                color: white;
            }
            QListWidget::item:hover {
                background: #4A5568;
            }
            QFrame#bottom_bar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4A5568, stop:1 #2D3748);
                border: 1px solid #718096;
                border-radius: 12px;
                padding: 10px;
            }
            QScrollArea {
                border: 1px solid #4A5568;
                border-radius: 8px;
                background: #2D3748;
            }
            QScrollBar:horizontal {
                border: none;
                background: #2D3748;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: #4FD1C7;
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #68D391;
            }
        """)
    
    def _build_enhanced_ui(self):
        """Build the enhanced UI with modern styling"""
        # Same structure as original but with enhanced styling
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Middle: Video / Markers split
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4A5568, stop:1 #2D3748);
                border-radius: 2px;
            }
            QSplitter::handle:hover {
                background: #4FD1C7;
            }
        """)
        
        # Left side: video container
        video_container = QFrame()
        video_container.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4A5568, stop:1 #2D3748);
                border: 1px solid #718096;
                border-radius: 15px;
                padding: 15px;
            }
        """)
        
        # Add drop shadow to video container
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 120))
        shadow.setOffset(0, 8)
        video_container.setGraphicsEffect(shadow)
        
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(15)
        
        # Video display
        self.video_display = QLabel()
        self.video_display.setObjectName("video_display")
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_display.setScaledContents(True)
        self.video_display.setMinimumHeight(400)
        video_layout.addWidget(self.video_display)
        
        # Import popup overlay
        self.create_enhanced_import_popup(video_container)
        
        # Enhanced slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(lambda v: self.frameChanged.emit(v))
        video_layout.addWidget(self.slider)
        
        # Enhanced control buttons
        self.create_enhanced_controls(video_layout)
        
        splitter.addWidget(video_container)
        
        # Right side: Enhanced markers pane
        self.create_enhanced_markers_panel(splitter)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        
        main_layout.addWidget(splitter, stretch=1)
        
        # Enhanced bottom bar
        self.create_enhanced_bottom_bar(main_layout)
        
        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer_tick)
    
    def create_enhanced_import_popup(self, parent):
        """Create enhanced import popup"""
        self.import_popup = QFrame(parent)
        self.import_popup.setStyleSheet("""
            QFrame {
                background: rgba(26, 32, 44, 240);
                border: 3px dashed #4FD1C7;
                border-radius: 15px;
            }
        """)
        
        popup_layout = QVBoxLayout(self.import_popup)
        popup_layout.setContentsMargins(0, 0, 0, 0)
        popup_layout.setSpacing(0)
        popup_layout.addStretch()
        
        # Icon
        icon_label = QLabel("🎬")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("""
            QLabel {
                font-size: 64px;
                background: none;
                border: none;
                color: #4FD1C7;
                padding: 20px;
            }
        """)
        popup_layout.addWidget(icon_label)
        
        # Button
        hbox = QHBoxLayout()
        hbox.addStretch()
        
        self.importPopupBtn = QPushButton("Import Video", self.import_popup)
        self.importPopupBtn.setObjectName("importPopupBtn")
        self.importPopupBtn.clicked.connect(self.importRequested.emit)
        hbox.addWidget(self.importPopupBtn)
        hbox.addStretch()
        
        popup_layout.addLayout(hbox)
        
        # Help text
        help_text = QLabel("Drag & drop a video file or click to browse")
        help_text.setAlignment(Qt.AlignCenter)
        help_text.setStyleSheet("""
            QLabel {
                color: #A0AEC0;
                font-size: 16px;
                background: none;
                border: none;
                padding: 20px;
            }
        """)
        popup_layout.addWidget(help_text)
        popup_layout.addStretch()
        
        self.import_popup.setGeometry(20, 20, 800, 500)
        self.import_popup.show()
    
    def create_enhanced_controls(self, parent_layout):
        """Create enhanced control buttons"""
        btn_row = QHBoxLayout()
        btn_row.setSpacing(15)
        
        btn_row.addStretch()
        
        self.toggle_button = QPushButton("▶ Play")
        self.toggle_button.setEnabled(False)
        self.toggle_button.clicked.connect(self._on_toggle_clicked)
        btn_row.addWidget(self.toggle_button)
        
        self.detect_button = QPushButton("🔍 Detect Gestures")
        self.detect_button.setEnabled(False)
        self.detect_button.clicked.connect(lambda: self.detectRequested.emit())
        btn_row.addWidget(self.detect_button)
        
        self.blur_button = QPushButton("🔒 Blur Person")
        self.blur_button.setEnabled(False)
        self.blur_button.clicked.connect(lambda: self.blurRequested.emit(self.current_frame_idx))
        btn_row.addWidget(self.blur_button)
        
        self.export_button = QPushButton("📤 Export Video")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(lambda: self.exportRequested.emit())
        btn_row.addWidget(self.export_button)
        
        btn_row.addStretch()
        
        parent_layout.addLayout(btn_row)
    
    def create_enhanced_markers_panel(self, parent_splitter):
        """Create enhanced markers panel"""
        markers_container = QFrame()
        markers_container.setFixedWidth(280)
        markers_container.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4A5568, stop:1 #2D3748);
                border: 1px solid #718096;
                border-radius: 15px;
                padding: 20px;
            }
        """)
        
        # Add drop shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 120))
        shadow.setOffset(0, 8)
        markers_container.setGraphicsEffect(shadow)
        
        markers_layout = QVBoxLayout(markers_container)
        markers_layout.setContentsMargins(0, 0, 0, 0)
        markers_layout.setSpacing(20)
        
        # Enhanced title
        markers_title = QLabel("🎯 Detected Gestures")
        markers_title.setStyleSheet("""
            QLabel {
                font-weight: 700;
                color: #E2E8F0;
                font-size: 16px;
                padding: 15px 0;
                border-bottom: 3px solid #4FD1C7;
            }
        """)
        markers_layout.addWidget(markers_title)
        
        # Enhanced gesture list
        self.gesture_list = QListWidget()
        self.gesture_list.setFixedHeight(200)
        self.gesture_list.itemClicked.connect(
            lambda item: self.gestureItemClicked.emit(item.data(Qt.UserRole)[0])
        )
        markers_layout.addWidget(self.gesture_list)
        
        markers_layout.addStretch()
        parent_splitter.addWidget(markers_container)
    
    def create_enhanced_bottom_bar(self, parent_layout):
        """Create enhanced bottom timeline bar"""
        bottom_bar = QFrame()
        bottom_bar.setObjectName("bottom_bar")
        bottom_layout = QVBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(15, 10, 15, 10)
        bottom_layout.setSpacing(10)
        
        # Enhanced time ruler
        self.time_ruler = EnhancedTimeRuler()
        bottom_layout.addWidget(self.time_ruler)
        
        # Enhanced thumbnails
        self.thumbnail_scroll = QScrollArea()
        self.thumbnail_scroll.setFixedHeight(100)
        self.thumbnail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.thumbnail_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.thumbnail_scroll.setWidgetResizable(True)
        
        thumb_container = QWidget()
        self.thumbnail_layout = QHBoxLayout(thumb_container)
        self.thumbnail_layout.setContentsMargins(10, 10, 10, 10)
        self.thumbnail_layout.setSpacing(10)
        self.thumbnail_scroll.setWidget(thumb_container)
        
        bottom_layout.addWidget(self.thumbnail_scroll)
        parent_layout.addWidget(bottom_bar, stretch=0)
    
    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        if hasattr(self, 'import_popup'):
            # Update import popup size
            parent_rect = self.video_display.geometry()
            margin = 30
            self.import_popup.setGeometry(
                parent_rect.x() + margin,
                parent_rect.y() + margin,
                parent_rect.width() - 2 * margin,
                parent_rect.height() - 2 * margin
            )
    
    # All other methods remain the same as original EditorPanel
    def set_video_info(self, rotation_angle, total_frames, fps):
        """Set video information and enable controls"""
        self.rotation_angle = rotation_angle
        self.total_frames = total_frames
        self.fps = fps
        self.current_frame_idx = 0

        self.slider.setMaximum(max(0, total_frames - 1))
        self.slider.setEnabled(True)
        self.toggle_button.setEnabled(True)
        self.detect_button.setEnabled(True)
        self.blur_button.setEnabled(True)
        self.export_button.setEnabled(True)

        self.time_ruler.setVideoInfo(total_frames, fps)
        self.core_has_video = True
        self.import_popup.hide()
    
    def display_frame(self, img_bgr, frame_idx: int):
        """Display video frame"""
        if img_bgr is None:
            w = self.video_display.width()
            h = self.video_display.height()
            blank = QPixmap(w, h)
            blank.fill(Qt.black)
            self.video_display.setPixmap(blank)
            self.current_frame_idx = -1
            self.time_ruler.setCurrentFrame(-1)
            return

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        bytes_per_line = w * 3
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        self.video_display.setPixmap(pix)
        self.current_frame_idx = frame_idx
        self.time_ruler.setCurrentFrame(frame_idx)
        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)
    
    def clear_thumbnails(self):
        """Clear all thumbnails"""
        if hasattr(self, "thumbnail_labels"):
            for thumb in self.thumbnail_labels:
                self.thumbnail_layout.removeWidget(thumb)
                thumb.deleteLater()
            self.thumbnail_labels = []
            self.thumbnail_frame_indices = []
    
    def add_thumbnails(self, thumbs):
        """Add thumbnail images to the timeline"""
        if not hasattr(self, "thumbnail_labels"):
            self.thumbnail_labels = []
            self.thumbnail_frame_indices = []

        for idx, thumb_rgb in thumbs:
            # Convert numpy → QImage → QPixmap
            h, w, _ = thumb_rgb.shape
            bytes_per_line = w * 3
            qimg = QImage(thumb_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaledToHeight(80, Qt.SmoothTransformation)

            thumb_label = QLabel()
            thumb_label.setPixmap(pix)
            thumb_label.setFixedSize(QSize(pix.width(), pix.height()))
            thumb_label.setCursor(QCursor(Qt.PointingHandCursor))
            thumb_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #4A5568;
                    border-radius: 6px;
                    background: #2D3748;
                }
                QLabel:hover {
                    border-color: #4FD1C7;
                }
            """)
            # Emit frame index when clicked
            thumb_label.mousePressEvent = lambda e, i=idx: self.thumbnailClicked.emit(i)

            # Drop shadow for depth
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(5)
            shadow.setColor(QColor(0, 0, 0, 150))
            shadow.setOffset(0, 2)
            thumb_label.setGraphicsEffect(shadow)

            self.thumbnail_layout.addWidget(thumb_label)
            self.thumbnail_labels.append(thumb_label)
            self.thumbnail_frame_indices.append(idx)

    def add_gesture_items(self, segment_starts):
        """Populate the gesture list with detected frame indices"""
        self.gesture_list.clear()
        for idx in segment_starts:
            t = idx / self.fps if self.fps > 0 else 0
            mm = int(t // 60)
            ss = int(t % 60)
            msec = int((t - int(t)) * 1000)
            time_str = f"{mm:02}:{ss:02}.{msec:03}"
            item = QListWidgetItem(f"✋  {time_str}")
            item.setData(Qt.UserRole, idx)
            self.gesture_list.addItem(item)
        # Enable the blur button only if we have gestures
        self.blur_button.setEnabled(bool(segment_starts))

    def clear_markers(self):
        """Clear all detected gestures"""
        self.gesture_list.clear()
        self.blur_button.setEnabled(False)

    def show_progress_dialog(self, title="Processing", message="Please wait..."):
        """Show a modal, indeterminate QProgressDialog"""
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
        """Hide the progress dialog"""
        if hasattr(self, "_progress"):
            self._progress.close()
            del self._progress

    def _on_toggle_clicked(self):
        """Handle Play/Pause toggle"""
        self.is_playing = not self.is_playing
        self.playToggled.emit(self.is_playing)
        self.toggle_button.setText("⏸ Pause" if self.is_playing else "▶ Play")

    # (Optionally) you can leave this empty if the controller handles playback timing
    def _on_timer_tick(self):
        """Internal timer slot (controller usually drives playback)"""
        pass



    

class KeyboardShortcutsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setModal(True)
        self.setFixedSize(350, 240)
        self.setWindowFlags(
            Qt.Dialog
            | Qt.WindowCloseButtonHint
            | Qt.MSWindowsFixedSizeDialogHint
        )

        self.setStyleSheet("""
            QDialog {
                background: #1A202C; 
                border-radius: 8px;
            }
            QLabel#header {
                font-size: 16px; 
                font-weight: bold; 
                color: #4FD1C7;
            }
            QLabel {
                font-size: 14px; 
                color: #E2E8F0;
            }
            QPushButton {
                background: #4FD1C7;
                color: #1A202C;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 13px;
                font-weight: 600;
                min-width: 80px;
            }
            QPushButton:hover {
                background: #38B2AC;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        header = QLabel("Keyboard Shortcuts", self)
        header.setObjectName("header")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        shortcuts = [
            ("Ctrl+O", "Open Video"),
            ("Ctrl+S", "Save Project"),
            ("Ctrl+Q", "Quit"),
            ("F11",    "Toggle Fullscreen"),
        ]
        for key, desc in shortcuts:
            lbl = QLabel(f"<tt>{key}</tt> &nbsp;&nbsp;–&nbsp;&nbsp; {desc}", self)
            layout.addWidget(lbl)

        layout.addStretch()

        close_btn = QPushButton("Close", self)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)




#do not delete
EditorPanel = EnhancedEditorPanel   
