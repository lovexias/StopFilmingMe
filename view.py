# ──── Enhanced Video Editor Interface ──────────────────────────────────────────────────────

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QSlider, QPushButton,
    QListWidget, QListWidgetItem, QScrollArea, QLabel, QFrame, QSizePolicy,
    QProgressDialog, QProgressBar, QApplication, QMainWindow, QMenuBar, QAction,
    QStatusBar, QToolBar, QSpacerItem, QGraphicsDropShadowEffect, QDialog
)
from PyQt5.QtCore import Qt, QPoint, QTimer, QSize, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import QImage, QPixmap, QCursor, QPainter, QColor, QFont, QIcon, QPalette, QLinearGradient, QBrush, QPen, QPolygon, QFontMetrics
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThread


#NEW
# --- ProcessingDialog: polished, theme-consistent progress UI ---
from PyQt5.QtCore import Qt, QTimer, QElapsedTimer
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QHBoxLayout, QPushButton
from PyQt5.QtWidgets import QRadioButton, QButtonGroup
from PyQt5.QtWidgets import QSpinBox
# --- In-app Export dialog -----------------------------------------------------
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QComboBox, QPushButton, QFileDialog, QSlider, QSpinBox)
from PyQt5.QtCore import Qt
import os
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent


APP_BG = "#0f172a"      # slate-900
CARD_BG = "#111827"     # near-black card
TEXT    = "#e5e7eb"     # slate-200
SUBTEXT = "#9ca3af"     # slate-400
ACCENT  = "#14b8a6"     # teal-500
ACCENT_DARK = "#0d9488" # teal-600
BORDER  = "#1f2937"     # slate-800


class ExportDialog(QDialog):
    def __init__(self, parent=None, suggest_name="export", suggest_dir=None):
        super().__init__(parent)
        self.setWindowTitle("Export Video")
        self.setModal(True)
        self.setMinimumWidth(520)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self.setStyleSheet("""
            QDialog { background:#111827; color:#e5e7eb; border:1px solid #1f2937; border-radius:10px; }
            QLabel  { color:#e5e7eb; }
            QLineEdit, QComboBox {
                background:#0f172a; color:#e5e7eb; border:1px solid #1f2937; border-radius:8px; padding:6px 8px;
            }
            QSlider::groove:horizontal { height:8px; border-radius:4px; background:#1f2937; }
            QSlider::sub-page:horizontal { background:#14b8a6; border-radius:4px; }
            QSlider::handle:horizontal { background:#14b8a6; width:18px; height:18px; margin:-6px 0; border-radius:9px; }
            QPushButton { background:#0d9488; color:white; border:none; border-radius:8px; padding:8px 14px; }
            QPushButton:hover { background:#14b8a6; }
        """)

        v = QVBoxLayout(self); v.setContentsMargins(16,16,16,16); v.setSpacing(10)

        # Destination folder + filename
        row_path = QHBoxLayout()
        self.dir_edit = QLineEdit(suggest_dir or os.path.expanduser("~"))
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._pick_folder)
        row_path.addWidget(QLabel("Folder"))
        row_path.addWidget(self.dir_edit, 1)
        row_path.addWidget(btn_browse)
        v.addLayout(row_path)

        row_name = QHBoxLayout()
        self.name_edit = QLineEdit(suggest_name)
        row_name.addWidget(QLabel("File name"))
        row_name.addWidget(self.name_edit, 1)
        v.addLayout(row_name)

        # Container / format
        row_fmt = QHBoxLayout()
        self.format_box = QComboBox()
        self.format_box.addItems(["mp4", "mov", "avi"])
        row_fmt.addWidget(QLabel("Format"))
        row_fmt.addWidget(self.format_box, 1)
        v.addLayout(row_fmt)

        # Codec (basic choices; adapt to what your core supports)
        row_codec = QHBoxLayout()
        self.codec_box = QComboBox()
        # sensible defaults per container
        self._codec_map = {
            "mp4": ["h264", "hevc"],
            "mov": ["h264", "prores"],
            "avi": ["mjpeg", "h264"]
        }
        self._refresh_codecs()
        self.format_box.currentTextChanged.connect(self._refresh_codecs)
        row_codec.addWidget(QLabel("Codec"))
        row_codec.addWidget(self.codec_box, 1)
        v.addLayout(row_codec)

        # Quality (target bitrate Mbps)
        row_q = QHBoxLayout()
        self.quality_mbps = QSlider(Qt.Horizontal)
        self.quality_mbps.setRange(2, 50)  # 2–50 Mbps
        self.quality_mbps.setValue(12)
        self.quality_label = QLabel("12 Mbps")
        self.quality_mbps.valueChanged.connect(lambda v_: self.quality_label.setText(f"{v_} Mbps"))
        row_q.addWidget(QLabel("Bitrate"))
        row_q.addWidget(self.quality_mbps, 1)
        row_q.addWidget(self.quality_label)
        v.addLayout(row_q)

        # Resolution + FPS downscale shortcuts
        row_res = QHBoxLayout()
        self.res_box = QComboBox()
        self.res_box.addItems(["Original", "1080p", "720p", "480p"])
        row_res.addWidget(QLabel("Resolution"))
        row_res.addWidget(self.res_box, 1)

        self.fps_box = QComboBox()
        self.fps_box.addItems(["Original", "60", "30", "24"])
        row_res.addWidget(QLabel("FPS"))
        row_res.addWidget(self.fps_box, 1)
        v.addLayout(row_res)

        # Buttons
        row_btns = QHBoxLayout()
        row_btns.addStretch(1)
        btn_cancel = QPushButton("Cancel"); btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("Export"); btn_ok.clicked.connect(self.accept)
        row_btns.addWidget(btn_cancel); row_btns.addWidget(btn_ok)
        v.addLayout(row_btns)

    def _refresh_codecs(self):
        self.codec_box.clear()
        self.codec_box.addItems(self._codec_map.get(self.format_box.currentText(), ["h264"]))

    def _pick_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose export folder", self.dir_edit.text() or os.path.expanduser("~"))
        if path:
            self.dir_edit.setText(path)

    def result_values(self):
        fmt = self.format_box.currentText()
        name = self.name_edit.text().strip() or "export"
        folder = self.dir_edit.text().strip() or os.path.expanduser("~")
        base = os.path.join(folder, name)
        fullpath = f"{base}.{fmt}"
        return {
            "path": fullpath,
            "container": fmt,                       # "mp4"|"mov"|...
            "codec": self.codec_box.currentText(), # e.g. "h264"
            "bitrate_mbps": int(self.quality_mbps.value()),
            "resolution": self.res_box.currentText(),
            "fps": self.fps_box.currentText(),
        }


class ProcessingDialog(QDialog):
    """
    Consistent, modern progress dialog.
    - indeterminate: pass total_steps=None (shows busy bar)
    - determinate: pass total_steps=int and call .set_progress(current)
    """
    def __init__(self, parent, title="Processing", message="Working…", total_steps=None, allow_cancel=False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(420)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        # Stylesheet (rounded bar, accent chunk, dark bg)
        self.setStyleSheet(f"""
            QDialog {{
                background: {CARD_BG};
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 12px;
            }}
            QLabel {{
                color: {TEXT};
            }}
            QLabel[subtle="true"] {{
                color: {SUBTEXT};
                font-size: 12px;
            }}
            QProgressBar {{
                background: {APP_BG};
                border: 1px solid {BORDER};
                border-radius: 10px;
                text-align: center;
                padding: 3px;
                height: 20px;
                color: {TEXT};
            }}
            QProgressBar::chunk {{
                background: {ACCENT};
                border-radius: 8px;
            }}
            QPushButton {{
                background: {ACCENT_DARK};
                color: white;
                border: none;
                padding: 6px 10px;
                border-radius: 8px;
            }}
            QPushButton:hover {{ background: {ACCENT}; }}
            QPushButton:disabled {{ background: {BORDER}; color: {SUBTEXT}; }}
        """)

        self.total_steps = total_steps
        self.elapsed = QElapsedTimer()
        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._tick)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        self.msg = QLabel(message)
        lay.addWidget(self.msg)

        self.bar = QProgressBar()
        if total_steps is None:
            # indeterminate: pulse bar
            self.bar.setRange(0, 0)
            self.bar.setTextVisible(False)
        else:
            self.bar.setRange(0, total_steps)
            self.bar.setValue(0)
            self.bar.setFormat("%p%")
            self.bar.setTextVisible(True)
        lay.addWidget(self.bar)

        info_row = QHBoxLayout()
        self.left_info = QLabel("Starting…")
        self.left_info.setProperty("subtle", True)
        self.right_info = QLabel("00:00")
        self.right_info.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.right_info.setProperty("subtle", True)
        info_row.addWidget(self.left_info)
        info_row.addWidget(self.right_info)
        lay.addLayout(info_row)

        if allow_cancel:
            btn_row = QHBoxLayout()
            btn_row.addStretch(1)
            self.cancel_btn = QPushButton("Cancel")
            self.cancel_btn.clicked.connect(self.reject)
            btn_row.addWidget(self.cancel_btn)
            lay.addLayout(btn_row)
        else:
            self.cancel_btn = None

        self.elapsed.start()
        self.timer.start()

        

    # Public API
    def set_message(self, text: str):
        self.msg.setText(text)

    def set_progress(self, current: int, label: str = None):
        """For determinate mode: update bar and optional left-side label."""
        if self.total_steps is None:
            return
        self.bar.setValue(max(0, min(current, self.total_steps)))
        if label is not None:
            self.left_info.setText(label)

    def finish(self, final_text: str = "Done"):
        self.left_info.setText(final_text)
        self.timer.stop()
        self.accept()

    # internal
    def _tick(self):
        ms = self.elapsed.elapsed()
        mm = int(ms // 1000 // 60)
        ss = int(ms // 1000 % 60)
        elapsed_txt = f"{mm:02d}:{ss:02d}"

        if self.total_steps is None:
            # busy: bounce elapsed only
            self.right_info.setText(elapsed_txt)
            if not self.msg.text():
                self.msg.setText("Working… Please wait.")
        else:
            # ETA estimate
            cur = max(1, self.bar.value())
            eta = ""
            rate = cur / max(1, ms / 1000.0)  # steps per sec
            remain = max(0, self.total_steps - cur)
            if rate > 0.01:
                eta_s = int(remain / rate)
                eta_m = eta_s // 60
                eta_ss = eta_s % 60
                eta = f" • ETA {eta_m:02d}:{eta_ss:02d}"
            self.right_info.setText(elapsed_txt + eta)


class VideoViewport(QWidget):
    """
    A paint-on-demand widget that always renders the current frame with
    the correct aspect ratio and letterboxing/pillarboxing as needed.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._qimg = None          # QImage of the current frame (RGB)
        self._ar = None            # aspect ratio (w / h)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        # Start with painting disabled until we have valid data
        self.setUpdatesEnabled(False)
        # Ensure we use a stable minimum size
        self.setMinimumSize(100, 100)

    def set_aspect_from_size(self, w: int, h: int):
        if w > 0 and h > 0:
            self._ar = float(w) / float(h)
            self.setUpdatesEnabled(True)  # Enable updates when we have valid aspect ratio
            self.updateGeometry()
            self.update()

    def set_frame_qimage(self, qimg: QImage):
        # Store a deep copy to be safe against temporary buffers
        if qimg and not qimg.isNull():
            # Create a deep copy that owns its own data
            self._qimg = qimg.copy()
        else:
            self._qimg = None
            
        if self._ar is None and self._qimg and not self._qimg.isNull():
            self.set_aspect_from_size(self._qimg.width(), self._qimg.height())
        # Enable updates now that we have something to show
        if self._qimg:
            self.setUpdatesEnabled(True)
        self.update()

    def sizeHint(self):
        # Prefer a 16:9-ish starting box if we have no aspect yet
        if self._ar:
            return QSize(960, int(960 / self._ar))
        return QSize(960, 540)

    def paintEvent(self, ev):
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)

            # Fill background (letterbox areas)
            p.fillRect(self.rect(), QColor("#1A202C"))

            # Safety checks to prevent access violation
            if not self._qimg or self._qimg is None:
                p.end()
                return
                
            if self._qimg.isNull():
                p.end()
                return
                
            if not self._ar or self._ar <= 0:
                p.end()
                return

            r = self.rect()
            rw, rh = r.width(), r.height()
            
            if rw <= 0 or rh <= 0:
                p.end()
                return
                
            current_ar = float(rw) / float(rh)

            if current_ar > self._ar:
                # too wide → fit height
                th = rh
                tw = int(th * self._ar)
            else:
                # too tall → fit width
                tw = rw
                th = int(tw / self._ar)

            # center the target rectangle
            tx = r.x() + (rw - tw) // 2
            ty = r.y() + (rh - th) // 2
            target = QRect(tx, ty, tw, th)

            # Draw the image
            p.drawImage(target, self._qimg)
            p.end()
            
        except Exception as e:
            print(f"Error in VideoViewport.paintEvent: {e}")
            import traceback
            traceback.print_exc()
            try:
                p.end()
            except:
                pass
class CardSection(QFrame):
    """
    Reusable inspector card with teal header and a sunken inner panel.
    Matches the look of your right-side cards (Video / Detection / Blur / Selection).
    """
    def __init__(self, title_text: str, parent=None):
        super().__init__(parent)
        self.setObjectName("sfCard")
        self.setStyleSheet("""
            QFrame#sfCard {
                background: #0f1216;               /* card surface */
                border: 1px solid #2b3037;         /* stroke */
                border-radius: 12px;
            }
            QFrame#sfInner {
                background: #11151b;               /* sunken inner */
                border: 1px solid #242a32;
                border-radius: 10px;
            }
            QLabel#sfTitle {
                color: #2dd4bf;                    /* teal title */
                font-weight: 700;
                font-size: 14px;
            }
            QLabel#sfField {
                color: #e6eaf0;
                font-weight: 600;
                font-size: 12px;
            }
            QLabel#sfPill {
                background: #171a1f;
                color: #aab3c2;
                border: 1px solid #2b3037;
                border-radius: 8px;
                padding: 6px 10px;
                min-width: 48px;
                qproperty-alignment: AlignCenter;
            }
            QRadioButton { color: #e6eaf0; }
            QSlider::groove:horizontal {
                background: #171a1f; height: 6px; border-radius: 3px;
                border: 1px solid #2b3037;
            }
            QSlider::sub-page:horizontal { background: #2dd4bf; border-radius: 3px; }
            QSlider::handle:horizontal {
                background: #2dd4bf; width: 18px; height: 18px;
                border: 2px solid #0f1216; border-radius: 9px; margin: -7px 0;
            }
            QListWidget {
                background: #0f1216;
                border: 1px solid #2b3037;
                border-radius: 10px;
                padding: 8px;
                color: #e6eaf0;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #242a32;
                border-radius: 6px;
                margin: 2px 0;
            }
            QListWidget::item:selected {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #2dd4bf, stop:1 #14b8a6);
                color: #0b0d11;
            }
            QListWidget::item:hover { background: #111b24; }
        """)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        self.title = QLabel(title_text); self.title.setObjectName("sfTitle")
        outer.addWidget(self.title)

        self.inner = QFrame(self); self.inner.setObjectName("sfInner")
        self.inner_lay = QVBoxLayout(self.inner)
        self.inner_lay.setContentsMargins(14, 14, 14, 14)
        self.inner_lay.setSpacing(10)
        outer.addWidget(self.inner)

    def add_row(self, left_widget: QWidget, right_widget: QWidget = None):
        row = QHBoxLayout(); row.setSpacing(10)
        row.addWidget(left_widget)
        if right_widget is not None:
            row.addWidget(right_widget, 1)
        self.inner_lay.addLayout(row)
        return row

    def add_label_value(self, label_text: str, value_text: str = "—"):
        lab = QLabel(label_text); lab.setObjectName("sfField")
        pill = QLabel(value_text); pill.setObjectName("sfPill")
        row = QHBoxLayout(); row.setSpacing(10)
        row.addWidget(lab); row.addWidget(pill, 1, Qt.AlignRight)
        self.inner_lay.addLayout(row)
        return pill




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
        self.minimize_btn = self.create_window_button("─", "#4A5568", "#5A6578")
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
    gestureItemClicked = pyqtSignal(object)   # <— accepts dict/int
    exportRequested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.core_has_video = False

        # Internal state
        self.cap = None
        self.rotation_angle = 0
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame_idx = 0
        self.is_playing = False

        # --- Playback timer (create BEFORE using it) ---
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._on_timer_tick)

        # --- Audio setup (consolidated and fixed) ---
        self._player = QMediaPlayer(self)
        self._player.setVolume(90)
        self._has_media = False
        self._audio_fallback_path = None
        
        # Connect error handling AFTER creating the player
        try:
            if hasattr(self._player, "errorOccurred"):
                self._player.errorOccurred.connect(self._on_media_error)
            else:
                self._player.error.connect(self._on_media_error)
            self._player.mediaStatusChanged.connect(self._on_media_status)
        except Exception as e:
            print(f"Audio setup warning: {e}")

        # Apply styles and build UI
        self.apply_modern_styles()
        self._build_enhanced_ui()   

    
    
    #AUDIO NEW
      

    def prepare_audio_for(self, path: str):
        """
        Try to load audio directly from the video. If DirectShow rejects it,
        extract a WAV with ffmpeg and load that instead.
        """
        from PyQt5.QtCore import QUrl
        import os, subprocess, tempfile, shutil

        # 1) first try direct
        self.set_media_source(path)

        # Sometimes the status goes InvalidMedia asynchronously; we also
        # try a quick probe via ffprobe to decide early whether to fall back.
        def _needs_fallback(p):
            try:
                # quick probe for audio stream presence
                # this will succeed even if ffprobe isn't there; we handle errors
                cp = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "a:0",
                    "-show_entries", "stream=codec_name", "-of", "default=nk=1:nw=1", p],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                return cp.returncode != 0 or not cp.stdout.strip()
            except Exception:
                # if ffprobe missing, we can’t tell — let QMediaPlayer try first
                return False

        # If it looks like no audio stream, bail out early (nothing to play)
        if _needs_fallback(path):
            print("No audio stream detected; skipping audio setup.")
            return

        # 2) Also set up a tiny delayed fallback if QMediaPlayer reports InvalidMedia
        #    (We keep a reference to the original path so _on_media_status can use it)
        self._orig_media_path_for_fallback = path

    def _on_media_status(self, status):
        """Handle media status changes, with automatic WAV fallback for DirectShow issues"""
        status_names = {
            0: "UnknownMediaStatus", 1: "NoMedia", 2: "Loading", 3: "Loaded",
            4: "Stalled", 5: "Buffering", 6: "Buffered", 7: "EndOfMedia", 8: "InvalidMedia"
        }
        
        print(f"Media status changed to: {status_names.get(int(status), status)}")
        
        if int(status) == 8:  # InvalidMedia - DirectShow codec issue
            print("DirectShow codec issue detected, falling back to WAV extraction...")
            self._try_wav_fallback()
        elif int(status) == 3:  # Loaded successfully
            self._has_media = True
            print(f"Audio loaded successfully - Duration: {self._player.duration()}ms")
            
            # NEW: If we're supposed to be playing, retry the audio playback
            if getattr(self, 'is_playing', False):
                print("Retrying audio playback after successful load...")
                self.audio_play_from_frame(self.current_frame_idx, self.fps)
                
    def _retry_audio_after_fallback(self):
        """Called after WAV fallback completes to retry audio playback"""
        # Wait a moment for the media to be fully loaded
        QTimer.singleShot(200, self._check_and_retry_audio)

    def _check_and_retry_audio(self):
        """Check if audio is ready and retry playback if needed"""
        try:
            if self._has_media and self._player.duration() > 0:
                if getattr(self, 'is_playing', False):
                    print(f"Retrying audio playback - Duration now: {self._player.duration()}ms")
                    self.audio_play_from_frame(self.current_frame_idx, self.fps)
            else:
                print(f"Audio still not ready: has_media={self._has_media}, duration={self._player.duration()}ms")
        except Exception as e:
            print(f"Retry audio error: {e}")


    def _on_media_error(self, error):
        """Handle media playback errors"""
        try:
            error_msg = self._player.errorString()
            print(f"Media error: {error_msg}")
            # If it's a codec error, try WAV fallback
            if "0x80040266" in error_msg or "codec" in error_msg.lower():
                self._try_wav_fallback()
        except Exception:
            print("Media playback error occurred")

    def _try_wav_fallback(self):
        """Extract audio to WAV and retry loading"""
        if not hasattr(self, '_video_path') or not self._video_path:
            return
            
        try:
            wav_path = self._extract_wav_with_ffmpeg(self._video_path)
            if wav_path and os.path.exists(wav_path):
                print(f"Retrying audio with WAV: {wav_path}")
                self._audio_fallback_path = wav_path
                self.set_media_source(wav_path)
                
                # NEW: Schedule a retry of audio playback after fallback
                self._retry_audio_after_fallback()
                
        except Exception as e:
            print(f"WAV fallback failed: {e}")    

    def _extract_wav_with_ffmpeg(self, src_path: str) -> str:
        """Extract 48kHz stereo WAV using ffmpeg"""
        import subprocess
        import tempfile
        
        base = os.path.splitext(os.path.basename(src_path))[0]
        temp_dir = tempfile.gettempdir()
        out_path = os.path.join(temp_dir, f"{base}_audio_fallback.wav")
        
        # Remove old fallback file if it exists
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except Exception:
                pass
        
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-vn",              # no video
            "-ac", "2",         # stereo
            "-ar", "48000",     # 48kHz
            "-acodec", "pcm_s16le",  # 16-bit PCM
            out_path
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode == 0 and os.path.exists(out_path):
                return out_path
            else:
                print(f"ffmpeg failed: {result.stderr[-500:]}")
                return ""
        except subprocess.TimeoutExpired:
            print("Audio extraction timed out")
            return ""
        except FileNotFoundError:
            print("ffmpeg not found - audio preview disabled")
            return ""
        except Exception as e:
            print(f"Audio extraction error: {e}")
            return ""
        
    @property
    def video_path(self):
        return getattr(self, "_video_path", None)

    @video_path.setter  
    def video_path(self, path):
        """Set video path and prepare audio with robust error handling"""
        self._video_path = path
        self._has_media = False
        
        if path and os.path.exists(path):
            try:
                # Try direct loading first
                self.set_media_source(path)
                
                # Give it a moment to load, then check if it worked
                QTimer.singleShot(500, self._check_audio_loaded)
                
            except Exception as e:
                print(f"Audio preparation failed: {e}")

    def _check_audio_loaded(self):
        """Check if audio loaded successfully, fallback to WAV if needed"""
        if not self._has_media and hasattr(self, '_video_path'):
            print("Audio didn't load, trying WAV fallback...")
            self._try_wav_fallback()

    def set_media_source(self, path: str):
        """Load media file for audio preview"""
        try:
            if not path or not os.path.exists(path):
                self._has_media = False
                return
                
            url = QUrl.fromLocalFile(path)
            content = QMediaContent(url)
            self._player.setMedia(content)
            self._player.setPosition(0)
            
        except Exception as e:
            print(f"Failed to set media source: {e}")
            self._has_media = False


    def audio_play_from_frame(self, frame_idx: int, fps: float):
        """Play audio from specific frame position with minimal overhead"""
        try:
            if not self._has_media or fps <= 0:
                return
                    
            # Calculate position with higher precision
            fps = max(1.0, float(fps))
            pos_ms = int(round(max(0, frame_idx) * 1000.0 / fps))
            
            # Only seek if we're significantly off (avoid micro-seeks)
            current_pos = self._player.position()
            if abs(current_pos - pos_ms) > 50:  # 50ms tolerance
                self._player.setPosition(pos_ms)
            
            # Start playback
            if self._player.state() != QMediaPlayer.PlayingState:
                self._player.play()
                
        except Exception as e:
            print(f"Audio play error: {e}")

    def audio_pause(self):
        """Pause audio playback with debugging"""
        try:
            print(f"--- AUDIO PAUSE ---")
            print(f"Has media: {self._has_media}")
            if self._has_media:
                self._player.pause()
                print(f"Audio paused, state: {self._player.state()}")
            else:
                print("No media to pause")
        except Exception as e:
            print(f"❌ Audio pause error: {e}")

    def audio_seek_to_frame(self, frame_idx: int, fps: float):
        """Seek audio to frame position without playing (for scrubbing)"""
        try:
            if self._has_media and fps > 0:
                fps = max(1.0, float(fps))
                pos_ms = int(round(max(0, frame_idx) * 1000.0 / fps))
                
                # Throttle seek operations to avoid overwhelming the audio system
                if not hasattr(self, '_last_seek_time'):
                    self._last_seek_time = 0
                
                import time
                current_time = time.time()
                if current_time - self._last_seek_time > 0.05:  # Max 20 seeks per second
                    self._player.setPosition(pos_ms)
                    self._last_seek_time = current_time
                    
        except Exception as e:
            print(f"Audio seek error: {e}")
    # ADDITIONAL: Add this method to monitor audio/video sync during playback
    def get_audio_position_ms(self):
        """Get current audio playback position in milliseconds"""
        try:
            if self._has_media:
                return self._player.position()
            return 0
        except Exception:
            return 0

    def check_av_sync(self, video_frame_idx: int, fps: float):
        """Debug method to check audio/video synchronization"""
        try:
            if self._has_media and fps > 0:
                audio_pos_ms = self.get_audio_position_ms()
                expected_audio_ms = int(video_frame_idx * 1000.0 / fps)
                diff_ms = abs(audio_pos_ms - expected_audio_ms)
                
                if diff_ms > 100:  # More than 100ms out of sync
                    print(f"A/V SYNC WARNING: Video frame {video_frame_idx} ({expected_audio_ms}ms) vs Audio {audio_pos_ms}ms (diff: {diff_ms}ms)")
                    
        except Exception as e:
            print(f"Sync check error: {e}")

    def cleanup_audio_resources(self):
        """Clean up temporary audio files"""
        try:
            if hasattr(self, '_audio_fallback_path') and self._audio_fallback_path:
                if os.path.exists(self._audio_fallback_path):
                    os.remove(self._audio_fallback_path)
                    self._audio_fallback_path = None
        except Exception as e:
            print(f"Audio cleanup error: {e}")

    def closeEvent(self, event):
        """Clean up resources when closing"""
        try:
            self.cleanup_audio_resources()
            if hasattr(self, '_player'):
                self._player.stop()
        except Exception:
            pass
        super().closeEvent(event)




    
    def _try_native_media(self, path: str) -> bool:
        """Try to load the video file directly. Returns True if backend accepts it."""
        self._has_media = False
        self._player.stop()
        self._player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
        # Give the backend a moment to change status
        QTimer.singleShot(0, lambda: None)
        # crude: we’ll consider it 'accepted' if it reaches Loaded within a short delay
        ok = []

        def _on_status(s):
            if int(s) == 3:   # Loaded
                ok.append(True)
        self._player.mediaStatusChanged.connect(_on_status)
        QApplication.processEvents()
        # tiny pump
        for _ in range(8):
            QApplication.processEvents()
            QTimer.singleShot(10, lambda: None)
        try:
            self._player.mediaStatusChanged.disconnect(_on_status)
        except Exception:
            pass

        if ok:
            self._has_media = True
            self._player.setPosition(0)
            return True
        return False

    def _extract_audio_to_wav(self, path: str) -> str:
        """Extract audio to a temporary WAV using ffmpeg CLI."""
        import tempfile, subprocess, shlex, os
        tmp_dir = tempfile.gettempdir()
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(tmp_dir, f"{base}_sfm_audio.wav")
        # -vn: no video; -ac 2: stereo; -ar 44100: 44.1k; 16-bit PCM by default
        cmd = f'ffmpeg -y -i "{path}" -vn -ac 2 -ar 44100 "{out_path}"'
        try:
            # hide console noise; ffmpeg must be in PATH
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return out_path
        except Exception as e:
            print("ffmpeg extract failed:", e)
            raise


    def debug_audio_status(self):
        """Debug method to check audio player status"""
        try:
            print(f"Audio debug:")
            print(f"  _has_media: {getattr(self, '_has_media', 'NOT SET')}")
            print(f"  Player state: {self._player.state()}")  # 0=Stopped, 1=Playing, 2=Paused
            print(f"  Media status: {self._player.mediaStatus()}")  # 3=Loaded, 8=InvalidMedia
            print(f"  Position: {self._player.position()}ms")
            print(f"  Duration: {self._player.duration()}ms")
            print(f"  Volume: {self._player.volume()}%")
            print(f"  Is muted: {self._player.isMuted()}")
            print(f"  Error: {self._player.errorString()}")
        except Exception as e:
            print(f"Audio debug error: {e}")

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
        self.video_display = VideoViewport()
        self.video_display.setObjectName("video_display")
        self.video_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_display.setMinimumHeight(400)
        video_layout.addWidget(self.video_display)

        # BELOW self.video_display setup NEW
        self.selection_badge = QLabel(self.video_display)
        self.selection_badge.setText("")
        self.selection_badge.setStyleSheet("""
            QLabel {
                background: transparent;
                color: white;
                font-weight: 700;
                padding: 6px 10px;
                border-radius: 8px;
            }
        """)
        self.selection_badge.hide()
        self.selection_badge.move(12, 12)  # top-left over the video

      
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
        """
        Left sidebar with FOUR cards:
        1) Video Properties
        2) Detection Settings
        3) Blur Settings
        4) Detected Gestures# --- 2) Detection Settings ---
card_detect = CardSection("Detection Settings", container)
# default values (adjust from your core later if you like)
self.lbl_conf = card_detect.add_label_value("Confidence:", "80%")
self.lbl_skip = card_detect.add_label_value("Frame Skip:", "2")
root.addWidget(card_detect)

        Exactly like your reference (3 boxes before the list).
        """
        container = QFrame()
       
        # NEW
        container.setMinimumWidth(330)                         # keep a sensible floor
        container.setMaximumWidth(700)                         # optional cap; remove if you want unlimited
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        container.setStyleSheet("QFrame { background:#1a202c; border:0; }")

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(18); shadow.setColor(QColor(0,0,0,110)); shadow.setOffset(0,6)
        container.setGraphicsEffect(shadow)

        root = QVBoxLayout(container)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(14)

        # --- 1) Video Properties ---
        card_video = CardSection("Video Properties", container)
        self.lbl_fps = card_video.add_label_value("FPS:", "--")
        self.lbl_res = card_video.add_label_value("Resolution:", "--")
        self.lbl_dur = card_video.add_label_value("Duration:", "--")
        root.addWidget(card_video)

               # --- 2) Detection Settings (editable) ---
        card_detect = CardSection("Detection Settings", container)

        # Confidence row: slider + % pill
        conf_row = QHBoxLayout(); conf_row.setSpacing(10)
        conf_label = QLabel("Confidence:"); conf_label.setObjectName("sfField")

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(30)    # 30% min (raise/lower if you prefer)
        self.conf_slider.setMaximum(95)    # 95% max (avoid 99/100 making no detections)
        self.conf_slider.setSingleStep(1)
        self.conf_slider.setValue(80)      # default 80%
        self.conf_val_pill = QLabel("80%"); self.conf_val_pill.setObjectName("sfPill")

        def _on_conf_change(v):
            if hasattr(self, "conf_val_pill"):
                self.conf_val_pill.setText(f"{v}%")
        self.conf_slider.valueChanged.connect(_on_conf_change)

        conf_row.addWidget(conf_label)
        conf_row.addWidget(self.conf_slider, 1)
        conf_row.addWidget(self.conf_val_pill)
        card_detect.inner_lay.addLayout(conf_row)

        # Frame Skip row: spinbox (how many frames to skip between detections)
        skip_row = QHBoxLayout(); skip_row.setSpacing(10)
        skip_label = QLabel("Frame Skip:"); skip_label.setObjectName("sfField")

        self.skip_spin = QSpinBox()
        self.skip_spin.setMinimum(1)
        self.skip_spin.setMaximum(10)
        self.skip_spin.setValue(2)
        # styling to match pills
        self.skip_spin.setStyleSheet("""
            QSpinBox {
                background:#171a1f; color:#e6eaf0; border:1px solid #2b3037;
                border-radius:8px; padding:4px 8px; min-width:64px;
            }
            QSpinBox::down-button, QSpinBox::up-button { width:16px; }
        """)

        skip_row.addWidget(skip_label)
        skip_row.addWidget(self.skip_spin, 0, Qt.AlignRight)
        card_detect.inner_lay.addLayout(skip_row)

        root.addWidget(card_detect)

        

        # --- 3) Blur Settings ---
        card_blur = CardSection("Blur Settings", container)

        blur_type_label = QLabel("Blur Type:"); blur_type_label.setObjectName("sfField")
        rb_row = QHBoxLayout(); rb_row.setSpacing(12)
        self.rb_gauss = QRadioButton("Gaussian")
        self.rb_pixel = QRadioButton("Pixelate")
        self.rb_solid = QRadioButton("Solid")
        self.rb_gauss.setChecked(True)
        self.blur_type_group = QButtonGroup(card_blur)
        self.blur_type_group.addButton(self.rb_gauss, 0)
        self.blur_type_group.addButton(self.rb_pixel, 1)
        self.blur_type_group.addButton(self.rb_solid, 2)
        rb_row.addWidget(self.rb_gauss); rb_row.addWidget(self.rb_pixel); rb_row.addWidget(self.rb_solid); rb_row.addStretch(1)
        row_bt = QHBoxLayout(); row_bt.setSpacing(10)
        row_bt.addWidget(blur_type_label); row_bt.addLayout(rb_row, 1)
        card_blur.inner_lay.addLayout(row_bt)

        self.blur_strength = QSlider(Qt.Horizontal)
        self.blur_strength.setMinimum(0); self.blur_strength.setMaximum(100)
        self.blur_strength_value = getattr(self, "blur_strength_value", 50)
        self.blur_strength.setValue(self.blur_strength_value)
        self.blur_strength.valueChanged.connect(self._on_strength_changed)
        self.lbl_strength_pct = QLabel(f"{self.blur_strength_value}%"); self.lbl_strength_pct.setObjectName("sfPill")

        row_str = QHBoxLayout(); row_str.setSpacing(10)
        lab_str = QLabel("Strength:"); lab_str.setObjectName("sfField")
        row_str.addWidget(lab_str); row_str.addWidget(self.blur_strength, 1); row_str.addWidget(self.lbl_strength_pct)
        card_blur.inner_lay.addLayout(row_str)

        root.addWidget(card_blur)

        # --- 4) Detected Gestures ---
        card_g = CardSection("Detected Gestures", container)
        self.gesture_list = QListWidget()
        # cap height so it’s ~50% shorter
        self.gesture_list.setMinimumHeight(140)
        self.gesture_list.setMaximumHeight(280)   # tweak as you like

        # Replace your existing lambda with this:
        self.gesture_list.itemClicked.connect(
            lambda it: self.gestureItemClicked.emit(
                ({"person_id": int(d[0]), "gesture": str(d[1]), "frame": int(d[2]), "bbox": d[3]}
                if isinstance((d := it.data(Qt.UserRole)), (tuple, list)) and len(d) >= 4 else
                d if isinstance(d, dict) else
                (int(d) if isinstance(d, (int, float)) else self.gesture_list.row(it)))
            )
        )
        
        card_g.inner_lay.addWidget(self.gesture_list)
        root.addWidget(card_g)          # no stretch on the gestures card
        root.addStretch(1)              # put remaining space below it

        parent_splitter.addWidget(container)
        # let the sidebar expand instead of staying fixed
        # If the video pane was added first, it's index 0 and the sidebar is index 1.
        try:
            parent_splitter.setStretchFactor(0, 3)  # video
            parent_splitter.setStretchFactor(1, 2)  # sidebar (allow growth)
            # give the splitter an initial layout (adjust numbers if you like)
            parent_splitter.setSizes([900, 480])    # [video width, sidebar width]
        except Exception:
            pass
        

    
    def _emit_gesture_payload(self, item):
        """Emit the stored payload (dict) for a clicked gesture row."""
        data = item.data(Qt.UserRole) if item is not None else None
        if data is None:
            # fallback to row index if someone forgot to set UserRole
            data = self.gesture_list.row(item)
        self.gestureItemClicked.emit(data)


    
    def _on_strength_changed(self, v: int):
        self.blur_strength_value = v
        if hasattr(self, "lbl_strength_pct"):
            self.lbl_strength_pct.setText(f"{v}%")

    
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
        super().resizeEvent(event)

        # keep the import overlay fitting the video area
        if hasattr(self, 'import_popup'):
            parent_rect = self.video_display.geometry()
            margin = 30
            self.import_popup.setGeometry(
                parent_rect.x() + margin,
                parent_rect.y() + margin,
                parent_rect.width() - 2 * margin,
                parent_rect.height() - 2 * margin
            )

        # rescale the currently shown frame to the new label size
        if hasattr(self, "_last_frame_bgr") and self._last_frame_bgr is not None:
            rgb = cv2.cvtColor(self._last_frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                self.video_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        


    
    # All other methods remain the same as original EditorPanel

    def get_detection_params(self):
        """Return current detection UI knobs in controller-friendly units."""
        conf_pct = self.conf_slider.value() if hasattr(self, "conf_slider") else 80
        frame_skip = self.skip_spin.value() if hasattr(self, "skip_spin") else 2
        return {
            "confidence": conf_pct / 100.0,   # 0.80 for 80%
            "frame_skip": int(frame_skip),
        }  


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

        # 🔊 Hook up audio if we already know the file path
        try:
            if getattr(self, "video_path", None):
                self.set_media_source(self.video_path)
        except Exception:
            pass


        # --- Update the Video Properties pills (FPS / Resolution / Duration) ---
        try:
            if hasattr(self, "lbl_fps"):
                self.lbl_fps.setText(f"{fps:.0f}" if fps else "--")

            if hasattr(self, "lbl_res"):
                # If you already know width/height here, set them:
                #   self.lbl_res.setText(f"{self.core.width}×{self.core.height}")
                # Otherwise show placeholder; we'll fill it on first frame paint.
                self.lbl_res.setText("--")

            if hasattr(self, "lbl_dur"):
                duration = (total_frames / fps) if (fps and total_frames) else 0
                mm = int(duration // 60)
                ss = int(duration % 60)
                self.lbl_dur.setText(f"{mm:02d}:{ss:02d}" if duration else "--")
        except Exception:
            pass

    #NEW TO HIGHLIGHTER THE PERSON
    def show_selection_badge(self, text: str):
        self.selection_badge.setText(text)
        self.selection_badge.adjustSize()
        self.selection_badge.show()

    def hide_selection_badge(self):
        self.selection_badge.hide()

    
    def display_frame(self, img_bgr, frame_idx: int):
        """Display video frame (optimized for smooth playback)"""
        
        if img_bgr is None:
            self.video_display.set_frame_qimage(QImage())
            self.current_frame_idx = -1
            self.time_ruler.setCurrentFrame(-1)
            return

        try:
            # Apply rotation if needed
            if hasattr(self, 'rotation_angle'):
                if self.rotation_angle == 90:
                    img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
                elif self.rotation_angle == 180:
                    img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_180)
                elif self.rotation_angle == 270:
                    img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            
            # Ensure contiguous array for performance
            rgb = np.ascontiguousarray(rgb)
            
            # Create QImage with copied data
            img_bytes = rgb.tobytes()
            qimg = QImage(img_bytes, w, h, bytes_per_line, QImage.Format_RGB888)
            qimg = qimg.copy()  # Force deep copy

            # Set aspect ratio only once
            if not hasattr(self.video_display, '_aspect_set'):
                self.video_display.set_aspect_from_size(w, h)
                self.video_display._aspect_set = True

            # Update display
            self.video_display.set_frame_qimage(qimg)
            self.current_frame_idx = frame_idx
            self.time_ruler.setCurrentFrame(frame_idx)
            
            # Update slider without triggering signals
            self.slider.blockSignals(True)
            self.slider.setValue(frame_idx)
            self.slider.blockSignals(False)

            # Sync audio only when NOT playing (avoid double-sync during playback)
            if not getattr(self, 'is_playing', False):
                self.audio_seek_to_frame(frame_idx, self.fps)

        except Exception as e:
            print(f"Error displaying frame: {e}")
            self.video_display.set_frame_qimage(QImage())
            self.current_frame_idx = -1
            self.time_ruler.setCurrentFrame(-1)
    

    
    def clear_thumbnails(self):
        """Clear all thumbnails"""
        if hasattr(self, "thumbnail_labels"):
            for thumb in self.thumbnail_labels:
                self.thumbnail_layout.removeWidget(thumb)
                thumb.deleteLater()
            self.thumbnail_labels = []
            self.thumbnail_frame_indices = []
    
    def add_thumbnails(self, thumbs):
        if not hasattr(self, "thumbnail_labels"):
            self.thumbnail_labels = []
            self.thumbnail_frame_indices = []

        for idx, thumb_rgb in thumbs:
            h, w, _ = thumb_rgb.shape
            bytes_per_line = w * 3
            qimg = QImage(thumb_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaledToHeight(80, Qt.SmoothTransformation)

            thumb_label = QLabel()  # ← QLabel, not VideoViewport
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
            thumb_label.mousePressEvent = lambda e, i=idx: self.thumbnailClicked.emit(i)

            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(5)
            shadow.setColor(QColor(0, 0, 0, 100))
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

 

    def _on_toggle_clicked(self):
        """Handle Play/Pause toggle with audio debugging"""
        self.is_playing = not self.is_playing
        self.playToggled.emit(self.is_playing)
        self.toggle_button.setText("⏸ Pause" if self.is_playing else "▶ Play")

        print(f"\n=== PLAY/PAUSE DEBUG ===")
        print(f"Playing: {self.is_playing}")
        print(f"Current frame: {self.current_frame_idx}")
        print(f"FPS: {self.fps}")
        
        # Debug audio status before attempting to play
        self.debug_audio_status()

        # 🔊 keep audio in lockstep with video transport
        if self.is_playing:
            print(f"Attempting to play audio from frame {self.current_frame_idx}")
            self.audio_play_from_frame(self.current_frame_idx, self.fps)
            
            # Check status after attempting to play
            print("After play attempt:")
            self.debug_audio_status()
        else:
            print("Pausing audio")
            self.audio_pause()



  
    def _on_timer_tick(self):
        """Internal timer slot (controller usually drives playback)"""
        pass
    # inside EnhancedEditorPanel (or EnhancedVideoEditor), add:

    # ---- Detect (indeterminate)
    def start_detect_progress(self):
        self._dlg_detect = PrettyProgress("Processing – StopFilmingMe", "Detecting gestures… Please wait…", self, determinate=False)
        self._dlg_detect.show(); QApplication.processEvents()

    def finish_detect_progress(self):
        if hasattr(self, "_dlg_detect"):
            self._dlg_detect.close(); del self._dlg_detect

    # ---- Export (determinate)
    def start_export_progress(self):
        self._dlg_export = PrettyProgress("Exporting – StopFilmingMe", "Writing video file…", self, determinate=True)
        self._dlg_export.show(); QApplication.processEvents()

    def set_export_progress(self, pct: int):
        if hasattr(self, "_dlg_export"): self._dlg_export.set_progress(pct); QApplication.processEvents()

    def finish_export_progress(self):
        if hasattr(self, "_dlg_export"):
            self._dlg_export.set_progress(100)
            self._dlg_export.close(); del self._dlg_export

    # ---- Blur (determinate; reuse same pattern)
    def start_blur_progress(self):
        self._dlg_blur = PrettyProgress("Processing – StopFilmingMe", "Blurring person in video…", self, determinate=True)
        self._dlg_blur.show(); QApplication.processEvents()

    def set_blur_progress(self, pct:int):
        if hasattr(self, "_dlg_blur"): self._dlg_blur.set_progress(pct); QApplication.processEvents()

    def finish_blur_progress(self):
        if hasattr(self, "_dlg_blur"):
            self._dlg_blur.set_progress(100)
            self._dlg_blur.close(); del self._dlg_blur



class PrettyProgress(QDialog):
    """Unified themed progress dialog with header, subtext, bar, and elapsed/ETA."""
    def __init__(self, title="Processing", text="Please wait…", parent=None, determinate=False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedWidth(460)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        # ---- Layout
        v = QVBoxLayout(self); v.setContentsMargins(16,16,16,16); v.setSpacing(10)

        self.header = QLabel(title); self.header.setStyleSheet("color:#E2E8F0; font-weight:700; font-size:14px;")
        v.addWidget(self.header)

        self.sub = QLabel(text); self.sub.setStyleSheet("color:#A0AEC0;")
        v.addWidget(self.sub)

        self.bar = QProgressBar(); v.addWidget(self.bar)
        self.bar.setTextVisible(True)
        self.bar.setFormat("%p%")

        # elapsed / eta line
        self.time_lbl = QLabel("00:00"); self.time_lbl.setAlignment(Qt.AlignRight)
        self.time_lbl.setStyleSheet("color:#A0AEC0;")
        v.addWidget(self.time_lbl)

        # ---- Style
        self.setStyleSheet("""
            QDialog { background:#2D3748; border:1px solid #4A5568; border-radius:8px; }
            QProgressBar {
                height:18px; border:1px solid #4A5568; border-radius:6px;
                background:#1A202C; text-align:center; color:#E2E8F0;
            }
            QProgressBar::chunk {
                border-radius:6px;
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #4FD1C7, stop:1 #38B2AC);
            }
        """)

        # ---- Mode
        self.determinate = determinate
        if determinate:
            self.bar.setRange(0, 100)
            self.bar.setTextVisible(True)
        else:
            # manual marquee so it always moves (works with custom styles)
            self.bar.setRange(0, 100)
            self.bar.setTextVisible(False)   # hide “0%”
            self._pct = 0
            self._spin = QTimer(self)
            self._spin.timeout.connect(self._tick)
            self._spin.start(30)  

        # timers for elapsed/eta
        self._start_ms = int(QApplication.instance().arguments() is not None)  # dummy to keep type checkers happy
        self._start_ms = QTimer().remainingTime()  # not used; we’ll just grab time()
        self._t = QTimer(self); self._t.timeout.connect(self._update_time); self._t.start(250)
        import time as _t; self._since = _t.time()
        self._pct = 0

    def _tick(self):
        self._pct = (self._pct + 2) % 101
        self.bar.setValue(self._pct)

    def closeEvent(self, e):
        if hasattr(self, "_spin"): self._spin.stop()
        super().closeEvent(e)


    def _update_time(self):
        import time
        elapsed = int(time.time() - self._since)
        mm, ss = divmod(elapsed, 60)
        self.time_lbl.setText(f"{mm:02d}:{ss:02d}")

    
    # Public API
    def set_text(self, text): self.sub.setText(text)
    def set_title(self, title): self.header.setText(title); self.setWindowTitle(title)
    def set_progress(self, pct: int):
        if self.determinate:
            self.bar.setValue(max(0, min(100, int(pct))))

    def on_export(self,panel, core, out_path):
        self.panel.start_export_progress()
        self._exp_thread = QThread(self.panel)
        self._exp_worker = ExportWorker(self.core, out_path=self._pick_path())
        self._exp_worker.moveToThread(self._exp_thread)

        self._exp_thread.started.connect(self._exp_worker.run)
        self._exp_worker.progress.connect(self.panel.set_export_progress)
        self._exp_worker.finished.connect(lambda ok, p: (self.panel.finish_export_progress(), self._on_export_done(ok, p)))
        self._exp_worker.error.connect(lambda msg: (self.panel.finish_export_progress(), self._toast(msg)))

        self._exp_worker.finished.connect(self._exp_thread.quit)
        self._exp_worker.finished.connect(self._exp_worker.deleteLater)
        self._exp_thread.finished.connect(self._exp_thread.deleteLater)

        self._exp_thread.start()


class ExportWorker(QObject):
    progress = pyqtSignal(int)     # 0..100
    finished = pyqtSignal(bool, str)  # ok, path
    error = pyqtSignal(str)

    def __init__(self, core, out_path):
        super().__init__()
        self.core = core
        self.out_path = out_path

    @pyqtSlot()
    def run(self):
        try:
            total = max(1, self.core.total_frames)  # adapt to your core
            writer = self.core.start_export(self.out_path)  # your API
            for i in range(total):
                frame = self.core.render_frame_for_export(i)  # your API
                writer.write(frame)
                if (i % 5) == 0:  # throttle UI updates
                    self.progress.emit(int((i + 1) * 100 / total))
            self.core.finish_export(writer)
            self.finished.emit(True, self.out_path)
        except Exception as e:
            self.error.emit(str(e))
class DetectWorker(QObject):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    @pyqtSlot()
    def run(self):
        try:
            segments = self.core.detect_gestures()   # your heavy call
            self.finished.emit(segments)
        except Exception as e:
            self.error.emit(str(e))

def on_detect(self):
    self.panel.start_detect_progress()
    self._det_thread = QThread(self.panel)
    self._det_worker = DetectWorker()
    self._det_worker.core = self.core
    self._det_worker.moveToThread(self._det_thread)

    self._det_thread.started.connect(self._det_worker.run)
    self._det_worker.finished.connect(lambda segs: (self.panel.finish_detect_progress(), self.panel.add_gesture_items(segs)))
    self._det_worker.error.connect(lambda msg: (self.panel.finish_detect_progress(), self._toast(msg)))

    self._det_worker.finished.connect(self._det_thread.quit)
    self._det_worker.finished.connect(self._det_worker.deleteLater)
    self._det_thread.finished.connect(self._det_thread.deleteLater)

    self._det_thread.start()

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
            lbl = QLabel(f"<tt>{key}</tt> &nbsp;&nbsp;—&nbsp;&nbsp; {desc}", self)
            layout.addWidget(lbl)

        layout.addStretch()

        close_btn = QPushButton("Close", self)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)




# ============================================================================
# IMPORTANT: main.py uses EditorPanel, which is aliased to EnhancedEditorPanel
# The EnhancedVideoEditor class above is NOT used - it's an alternative 
# implementation with a frameless window and custom title bar.
# ============================================================================
#do not delete
EditorPanel = EnhancedEditorPanel