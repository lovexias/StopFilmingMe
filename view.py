# ──── Enhanced Video Editor Interface ─────────────────────────────────────────
import os
import sys
import time
import shutil
import subprocess
import webbrowser


import cv2
import numpy as np

from PyQt5.QtCore import (
    Qt, QUrl, QTimer, QElapsedTimer, QPoint, QSize, QRect,
    QObject, QThread, pyqtSignal, pyqtSlot, QPropertyAnimation, QEasingCurve, QProcess
)
from PyQt5.QtGui import (
    QImage, QPixmap, QCursor, QPainter, QColor, QFont, QIcon, QPalette,
    QLinearGradient, QBrush, QPen, QPolygon, QFontMetrics, QDesktopServices
)
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QDialog, QLabel, QPushButton, QSlider,
    QListWidget, QListWidgetItem, QScrollArea, QFrame, QSizePolicy, QProgressBar,
    QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter, QMenuBar, QAction, QStatusBar,
    QToolBar, QSpacerItem, QGraphicsDropShadowEffect, QAbstractItemView, QTextBrowser,
    QLineEdit, QComboBox, QFileDialog, QRadioButton, QButtonGroup, QSpinBox
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QMessageBox


# Premiere Pro Style Colors - Professional Dark Theme
APP_BG = "#1e1e1e"         # Main background (darker)
PANEL_BG = "#232323"       # Panel backgrounds
CARD_BG = "#2a2a2a"        # Card/section backgrounds  
DARKER_BG = "#1a1a1a"      # Darker elements
TEXT = "#d4d4d4"           # Primary text (lighter)
SUBTEXT = "#969696"        # Secondary text
ACCENT = "#0078d4"         # Blue accent (Premiere style)
ACCENT_HOVER = "#106ebe"   # Hover state
ACCENT_LIGHT = "#40a6ff"   # Light accent
BORDER = "#404040"         # Borders
BORDER_LIGHT = "#505050"   # Lighter borders
SUCCESS = "#00d084"        # Success/positive
WARNING = "#ffb900"        # Warning
DANGER = "#d13438"         # Error/danger



def _app_root_dir() -> str:
    if getattr(sys, "_MEIPASS", None):      # PyInstaller bundle
        return sys._MEIPASS
    # dev run: folder of the launcher (main.py)
    return os.path.dirname(os.path.abspath(sys.argv[0]))

def _find_docs_html() -> str:
    base = _app_root_dir()
    candidates = [
        os.path.join(base, "docs", "documentation.html"),
        os.path.join(base, "docs", "index.html"),
        os.path.join(base, "documentation.html"),
        os.path.join(base, "docs.html"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


def open_in_edge(target: str) -> None:
    # Convert local path to file:// URL if it exists
    url = QUrl.fromLocalFile(target).toString() if os.path.exists(target) else target

    # 1) Prefer launching Edge directly with the URL/file (most reliable for local files)
    for exe in (
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ):
        if os.path.exists(exe):
            QProcess.startDetached(exe, [url])
            return

    # 2) Try the edge protocol (works well for http/https)
    if QDesktopServices.openUrl(QUrl(f"microsoft-edge:{url}")):
        return

    # 3) Last resort: system default
    QDesktopServices.openUrl(QUrl(url))


def open_docs_html_or_web(fallback_url: str = "https://example.com/stopfilming/docs") -> None:
    local = _find_docs_html()
    open_in_edge(local if local else fallback_url)


print("APP ROOT:", _app_root_dir())
print("DOCS PATH:", _find_docs_html())



class ExportDialog(QDialog):
    def __init__(self, parent=None, suggest_name="export", suggest_dir=None):
        super().__init__(parent)
        self.setWindowTitle("Export Video")
        self.setModal(True)
        self.setMinimumWidth(540)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self.setStyleSheet(f"""
            QDialog {{ 
                background: {PANEL_BG}; 
                color: {TEXT}; 
                border: 1px solid {BORDER}; 
                border-radius: 8px;
                font-family: 'Segoe UI', Tahoma, sans-serif;
            }}
            QLabel {{ 
                color: {TEXT}; 
                font-size: 13px;
                font-weight: 500;
            }}
            QLineEdit, QComboBox {{
                background: {DARKER_BG}; 
                color: {TEXT}; 
                border: 1px solid {BORDER}; 
                border-radius: 4px; 
                padding: 8px 10px;
                font-size: 13px;
                selection-background-color: {ACCENT};
            }}
            QLineEdit:focus, QComboBox:focus {{
                border-color: {ACCENT};
                background: {APP_BG};
            }}
            QSlider::groove:horizontal {{ 
                height: 4px; 
                border-radius: 2px; 
                background: {BORDER}; 
            }}
            QSlider::sub-page:horizontal {{ 
                background: {ACCENT}; 
                border-radius: 2px; 
            }}
            QSlider::handle:horizontal {{ 
                background: {ACCENT}; 
                width: 16px; 
                height: 16px; 
                margin: -6px 0; 
                border-radius: 8px;
                border: 2px solid {PANEL_BG};
            }}
            QSlider::handle:horizontal:hover {{ 
                background: {ACCENT_HOVER}; 
            }}
            QPushButton {{ 
                background: {ACCENT}; 
                color: white; 
                border: none; 
                border-radius: 4px; 
                padding: 10px 18px;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton:hover {{ 
                background: {ACCENT_HOVER}; 
            }}
            QPushButton:pressed {{ 
                background: {ACCENT}; 
                transform: translateY(1px);
            }}
        """)

        v = QVBoxLayout(self); v.setContentsMargins(24,20,24,20); v.setSpacing(16)

        # Destination folder + filename
        row_path = QHBoxLayout()
        self.dir_edit = QLineEdit(suggest_dir or os.path.expanduser("~"))
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._pick_folder)
        row_path.addWidget(QLabel("Output Folder"))
        row_path.addWidget(self.dir_edit, 1)
        row_path.addWidget(btn_browse)
        v.addLayout(row_path)

        row_name = QHBoxLayout()
        self.name_edit = QLineEdit(suggest_name)
        row_name.addWidget(QLabel("File Name"))
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
        self.quality_label.setStyleSheet(f"color: {ACCENT}; font-weight: 600;")
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
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet(f"""
            QPushButton {{
                background: {BORDER};
                color: {TEXT};
            }}
            QPushButton:hover {{
                background: {BORDER_LIGHT};
            }}
        """)
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("Export")
        row_btns.addWidget(btn_cancel)
        row_btns.addWidget(btn_ok)
        v.addLayout(row_btns)

        btn_ok.clicked.connect(self.accept)

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


# --- Quick Help dialog --------------------------------------------------------
class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quick Help")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self.setStyleSheet(f"""
            QDialog {{
                background: {PANEL_BG};
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 8px;
                font-family: 'Segoe UI', Tahoma, sans-serif;
            }}
            QLabel#title {{
                color: {ACCENT_LIGHT};
                font-size: 18px;
                font-weight: 700;
                padding-bottom: 4px;
            }}
            QLabel#section {{
                color: {TEXT};
                font-size: 13px;
                font-weight: 600;
                margin-top: 10px;
            }}
            QLabel#bullet {{
                color: {TEXT};
                font-size: 12px;
            }}
            QFrame#line {{
                background: {BORDER};
                height: 1px;
            }}
            QPushButton {{
                background: {ACCENT};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background: {ACCENT_HOVER}; }}
            QPushButton#secondary {{
                background: {PANEL_BG};
                color: {TEXT};
                border: 1px solid {BORDER};
            }}
            QPushButton#secondary:hover {{
                background: {BORDER};
            }}
        """)

        v = QVBoxLayout(self)
        v.setContentsMargins(20, 18, 20, 18)
        v.setSpacing(10)

        title = QLabel("StopFilming — Quick Help")
        title.setObjectName("title")
        v.addWidget(title)

        v.addWidget(self._line())

        # Sections
        v.addWidget(self._section("1) Import"))
        v.addWidget(self._bullets([
            "Click **File → Open Video…** or use the big **Import Video** button on the canvas.",
            "Supported: MP4, MOV, AVI (others may work depending on codecs)."
        ]))

        v.addWidget(self._section("2) Navigate & Play"))
        v.addWidget(self._bullets([
            "Use the **timeline slider** under the video to scrub.",
            "Press **Play/Pause** or hit **Space** to toggle playback.",
            "The **time ruler** shows running time; the blue line is the playhead."
        ]))

        v.addWidget(self._section("3) Detect Gestures"))
        v.addWidget(self._bullets([
            "Choose **Tools → Detect Gestures** (or the **Detect** button).",
            "We scan frames using an optimized single-pass detector for people & poses.",
            "Results appear in the **Markers** panel on the right."
        ]))

        v.addWidget(self._section("4) Review Results"))
        v.addWidget(self._bullets([
            "Click a result in the list to jump to that frame.",
            "We’ll highlight the person’s bounding box on the video."
        ]))

        v.addWidget(self._section("5) Blur Faces"))
        v.addWidget(self._bullets([
            "Select one or more people in the list and click **Blur**.",
            "Blurring persists across playback; you can add more people later.",
            "Use **Tools → Clear All Blurs** to reset."
        ]))

        v.addWidget(self._section("6) Export"))
        v.addWidget(self._bullets([
            "Choose **File → Export…** to create a new blurred video.",
            "Pick format/codec/bitrate; defaults aim for quality + compatibility."
        ]))

        v.addWidget(self._section("Tips"))
        v.addWidget(self._bullets([
            "For faster detection, avoid ultra-high-res sources when possible.",
            "If memory climbs, the app auto-trims frame cache to stay smooth.",
            "Use **View** menu to toggle thumbnails or the markers panel."
        ]))

        v.addStretch(1)
        v.addWidget(self._line())

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_docs = QPushButton("Open Docs")
        self.btn_docs.clicked.connect(self._open_docs)
        self.btn_close = QPushButton("Close")
        self.btn_close.setObjectName("secondary")
        self.btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_docs)
        btn_row.addWidget(self.btn_close)
        v.addLayout(btn_row)

    def _line(self):
        line = QFrame(); line.setObjectName("line")
        line.setFixedHeight(1)
        return line

    def _section(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("section")
        return lbl

    def _bullets(self, items):
        wrap = QVBoxLayout(); wrap.setContentsMargins(0,0,0,0); wrap.setSpacing(4)
        for t in items:
            lbl = QLabel(f"• {t}")
            lbl.setWordWrap(True)
            lbl.setObjectName("bullet")
            wrap.addWidget(lbl)
        c = QWidget(); c.setLayout(wrap)
        return c

    def _open_docs(self):
        open_docs_html_or_web("https://example.com/stopfilming/docs")


class ProcessingDialog(QDialog):
    """
    Enhanced progress dialog with Premiere Pro styling
    """
    def __init__(self, parent, title="Processing", message="Working...", total_steps=None, allow_cancel=False, show_cancel_button=True):
        super().__init__(parent)
        # Fix: Clean up the title to avoid repetition
        clean_title = title.replace("StopFilming", "").replace("–", "").strip()
        if not clean_title or clean_title == "Processing":
            clean_title = "StopFilming Pro"
        
        self.setWindowTitle(clean_title)
        self.setModal(True)
        self.setMinimumWidth(420)
        self.setMaximumWidth(500)
        self.setAttribute(Qt.WA_DeleteOnClose, True)


        # x gesture - stops what
        self.allow_cancel = bool(allow_cancel)
        self.cancelled = False
        self.setAttribute(Qt.WA_DeleteOnClose, not self.allow_cancel) #cancel gesture
    

        # Premiere Pro style progress dialog
        self.setStyleSheet(f"""
            QDialog {{
                background: {PANEL_BG};
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 6px;
                font-family: 'Segoe UI', Tahoma, sans-serif;
            }}
            QLabel {{
                color: {TEXT};
                font-size: 14px;
                font-weight: 500;
            }}
            QLabel[subtle="true"] {{
                color: {SUBTEXT};
                font-size: 12px;
                font-weight: 400;
            }}
            QProgressBar {{
                background: {DARKER_BG};
                border: 1px solid {BORDER};
                border-radius: 3px;
                text-align: center;
                padding: 2px;
                height: 20px;
                color: {TEXT};
                font-weight: 500;
                font-size: 12px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {ACCENT}, stop:1 {ACCENT_LIGHT});
                border-radius: 2px;
            }}
            QPushButton {{
                background: {ACCENT};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 13px;
            }}
            QPushButton:hover {{ 
                background: {ACCENT_HOVER}; 
            }}
            QPushButton:disabled {{ 
                background: {BORDER}; 
                color: {SUBTEXT}; 
            }}
        """)

        self.total_steps = total_steps
        self.elapsed = QElapsedTimer()
        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._tick)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 20)
        lay.setSpacing(12)

        # Clean header without redundant app name
        header_text = message if message != "Working…" else "Processing"
        self.msg = QLabel(header_text)
        self.msg.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {ACCENT_LIGHT};")
        lay.addWidget(self.msg)

        # Progress bar
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

        # Info row with better spacing
        info_row = QHBoxLayout()
        info_row.setContentsMargins(0, 0, 0, 0)
        
        self.left_info = QLabel("Starting…")
        self.left_info.setProperty("subtle", True)
        
        self.right_info = QLabel("00:00")
        self.right_info.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.right_info.setProperty("subtle", True)
        
        info_row.addWidget(self.left_info)
        info_row.addStretch()  # Push time to the right
        info_row.addWidget(self.right_info)
        lay.addLayout(info_row)
        self.elapsed.start()
        self.timer.start()

        # Add speed tracking for enhanced progress
        self.start_time = time.time()
        self.last_progress = 0
        self.last_time = self.start_time

        
        #cancelling blur
        self.cancelled = False
        self.allow_cancel = bool(allow_cancel)

        if allow_cancel and show_cancel_button:
            # create the button
            btn_row = QHBoxLayout()
            btn_row.addStretch(1)
            self.cancel_btn = QPushButton("Cancel")
            self.cancel_btn.clicked.connect(self.reject)
            btn_row.addWidget(self.cancel_btn)
            lay.addLayout(btn_row)
        else:
            self.cancel_btn = None
        
        
        
        #----------------END----------------------#
    
    #BLURRING CANCELLATION---------------///
    def was_cancelled(self) -> bool:
        return self.cancelled

    def reject(self):
        # ESC or button (if any) -> behave like ✖
        if self.allow_cancel:
            self.cancelled = True
            self.hide()
            return
        return super().reject()

    def closeEvent(self, e):
        # ✖ should cancel, but don't delete while work is running
        if self.allow_cancel:
            self.cancelled = True
            e.ignore()
            self.hide()
            return
        return super().closeEvent(e)
    #-----------------------////



    # Public API methods remain the same...
    def set_message(self, text: str):
        self.msg.setText(text)

    def set_progress(self, current: int, label: str = None):
        """For determinate mode: update bar and optional left-side label."""
        if self.total_steps is None:
            return
        self.bar.setValue(max(0, min(current, self.total_steps)))
        if label is not None:
            self.left_info.setText(label)
        
        # Enhanced: Calculate processing speed
        now = time.time()
        if now > self.last_time + 1:  # Update every second
            progress_delta = current - self.last_progress
            time_delta = now - self.last_time
            
            if time_delta > 0 and progress_delta > 0:
                speed = progress_delta / time_delta
                # Update the label to show speed
                if label:
                    self.left_info.setText(f"{label} • {speed:.1f}/sec")
            
            self.last_progress = current
            self.last_time = now

    def finish(self, final_text: str = "Complete"):
        self.left_info.setText(final_text)
        self.timer.stop()
        # Brief delay before closing to show completion
        QTimer.singleShot(500, self.accept)

    def _tick(self):
        ms = self.elapsed.elapsed()
        mm = int(ms // 1000 // 60)
        ss = int(ms // 1000 % 60)
        elapsed_txt = f"{mm:02d}:{ss:02d}"

        if self.total_steps is None:
            self.right_info.setText(elapsed_txt)
        else:
            cur = max(1, self.bar.value())
            eta = ""
            rate = cur / max(1, ms / 1000.0)
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

            # Fill background (letterbox areas) - Premiere Pro dark
            p.fillRect(self.rect(), QColor(APP_BG))

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
    Professional Premiere Pro style cards with subtle borders and shadows
    """
    def __init__(self, title_text: str, parent=None):
        super().__init__(parent)
        self.setObjectName("sfCard")
        self.setStyleSheet(f"""
            QFrame#sfCard {{
                background: {CARD_BG};               
                border: 1px solid {BORDER};         
                border-radius: 6px;
                margin: 2px;
            }}
            QFrame#sfInner {{
                background: {PANEL_BG};               
                border: 1px solid {BORDER};
                border-radius: 4px;
            }}
            QLabel#sfTitle {{
                color: {ACCENT_LIGHT};                    
                font-weight: 600;
                font-size: 13px;
                font-family: 'Segoe UI', Tahoma, sans-serif;
            }}
            QLabel#sfField {{
                color: {TEXT};
                font-weight: 500;
                font-size: 12px;
            }}
            QLabel#sfPill {{
                background: {DARKER_BG};
                color: {ACCENT_LIGHT};
                border: 1px solid {BORDER};
                border-radius: 4px;
                padding: 4px 8px;
                min-width: 40px;
                font-weight: 600;
                font-size: 12px;
                qproperty-alignment: AlignCenter;
            }}
            QRadioButton {{ 
                color: {TEXT}; 
                font-size: 12px;
                spacing: 8px;
            }}
            QRadioButton::indicator {{
                width: 14px;
                height: 14px;
            }}
            QRadioButton::indicator:unchecked {{
                border: 2px solid {BORDER};
                border-radius: 7px;
                background: {DARKER_BG};
            }}
            QRadioButton::indicator:checked {{
                border: 2px solid {ACCENT};
                border-radius: 7px;
                background: {ACCENT};
            }}
            QSlider::groove:horizontal {{
                background: {BORDER}; 
                height: 4px; 
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{ 
                background: {ACCENT}; 
                border-radius: 2px; 
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT}; 
                width: 14px; 
                height: 14px;
                border: 2px solid {PANEL_BG}; 
                border-radius: 7px; 
                margin: -5px 0;
            }}
            QSlider::handle:horizontal:hover {{
                background: {ACCENT_HOVER};
            }}
            QListWidget {{
                background: {DARKER_BG};
                border: 1px solid {BORDER};
                border-radius: 4px;
                padding: 4px;
                color: {TEXT};
                selection-background-color: {ACCENT};
                outline: none;
            }}
            QListWidget::item {{
                padding: 8px 10px;
                border-bottom: 1px solid {BORDER};
                border-radius: 3px;
                margin: 1px 0;
                color: {TEXT};
            }}
            QListWidget::item:selected {{
                background: {ACCENT};
                color: white;
            }}
            QListWidget::item:hover:!selected {{ 
                background: {BORDER}; 
            }}
            QSpinBox {{
                background: {DARKER_BG}; 
                color: {TEXT}; 
                border: 1px solid {BORDER};
                border-radius: 4px; 
                padding: 4px 8px; 
                min-width: 50px;
                font-size: 12px;
            }}
            QSpinBox:focus {{
                border-color: {ACCENT};
            }}
            QSpinBox::down-button, QSpinBox::up-button {{ 
                width: 14px; 
                background: {BORDER};
                border: none;
            }}
            QSpinBox::down-button:hover, QSpinBox::up-button:hover {{ 
                background: {BORDER_LIGHT}; 
            }}
        """)
        
        # Add subtle shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 40))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(8)

        self.title = QLabel(title_text); self.title.setObjectName("sfTitle")
        outer.addWidget(self.title)

        self.inner = QFrame(self); self.inner.setObjectName("sfInner")
        self.inner_lay = QVBoxLayout(self.inner)
        self.inner_lay.setContentsMargins(12, 10, 12, 10)
        self.inner_lay.setSpacing(8)
        outer.addWidget(self.inner)

    def add_row(self, left_widget: QWidget, right_widget: QWidget = None):
        row = QHBoxLayout(); row.setSpacing(8)
        row.addWidget(left_widget)
        if right_widget is not None:
            row.addWidget(right_widget, 1)
        self.inner_lay.addLayout(row)
        return row

    def add_label_value(self, label_text: str, value_text: str = "–"):
        lab = QLabel(label_text); lab.setObjectName("sfField")
        pill = QLabel(value_text); pill.setObjectName("sfPill")
        row = QHBoxLayout(); row.setSpacing(8)
        row.addWidget(lab); row.addWidget(pill, 1, Qt.AlignRight)
        self.inner_lay.addLayout(row)
        return pill


class ModernTitleBar(QWidget):
    """Custom title bar with Premiere Pro styling"""
    
    minimizeClicked = pyqtSignal()
    maximizeClicked = pyqtSignal()
    closeClicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setStyleSheet(f"""
            QWidget {{
                background: {PANEL_BG};
                border-bottom: 1px solid {BORDER};
            }}
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 4, 0)
        layout.setSpacing(0)
        
        # App title
        self.title_label = QLabel("StopFilming Pro")
        f = QFont("Segoe UI", 13, QFont.Normal)
        self.title_label.setFont(f)
        self.title_label.setStyleSheet(f"""
            color: {TEXT};
            font-size: 13px;
            font-weight: 500;
            padding: 0 8px;
        """)
        layout.addWidget(self.title_label)
        
        layout.addStretch()
        
        # Window controls
        self.minimize_btn = self.create_window_button("−", PANEL_BG, BORDER)
        self.maximize_btn = self.create_window_button("□", PANEL_BG, BORDER)
        self.close_btn = self.create_window_button("×", PANEL_BG, DANGER)
        
        self.minimize_btn.clicked.connect(self.minimizeClicked.emit)
        self.maximize_btn.clicked.connect(self.maximizeClicked.emit)
        self.close_btn.clicked.connect(self.closeClicked.emit)
        
        layout.addWidget(self.minimize_btn)
        layout.addWidget(self.maximize_btn)
        layout.addWidget(self.close_btn)
    
    def create_window_button(self, text, bg_color, hover_color):
        btn = QPushButton(text)
        btn.setFixedSize(32, 28)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: {TEXT};
                border: none;
                font-size: 14px;
                font-weight: bold;
                margin: 2px 1px;
                border-radius: 2px;
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
    """Premiere Pro style timeline ruler"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame = 0
        self.setMinimumHeight(28)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Premiere Pro style background
        self.setStyleSheet(f"""
            QWidget {{
                background: {PANEL_BG};
                border-top: 1px solid {BORDER};
                border-bottom: 1px solid {BORDER};
            }}
        """)

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
        
        # Professional background
        painter.fillRect(0, 0, w, h, QColor(PANEL_BG))
        
        if self.total_frames <= 0 or self.fps <= 0:
            return
            
        total_seconds = self.total_frames / self.fps
        num_ticks = min(20, int(total_seconds))
        
        if num_ticks == 0:
            return
            
        interval_sec = total_seconds / num_ticks

        # Draw tick marks and labels (Premiere style)
        painter.setPen(QPen(QColor(BORDER_LIGHT), 1))
        font = QFont("Segoe UI", 10, QFont.Normal)
        painter.setFont(font)
        
        for i in range(num_ticks + 1):
            sec = i * interval_sec
            x = int((sec / total_seconds) * w)
            
            # Major tick
            painter.drawLine(x, h - 8, x, h)
            
            # Time label
            mm = int(sec // 60)
            ss = int(sec % 60)
            label = f"{mm:02}:{ss:02}"
            
            fm = QFontMetrics(font)
            text_width = fm.width(label)
            painter.setPen(QColor(TEXT))
            painter.drawText(x - text_width // 2, h - 12, label)
            painter.setPen(QPen(QColor(BORDER_LIGHT), 1))

        # Current position indicator (Premiere blue)
        cur_sec = self.current_frame / self.fps
        if cur_sec > total_seconds:
            cur_sec = total_seconds
        x_cur = int((cur_sec / total_seconds) * w)
        x_cur = max(0, min(x_cur, w))

        # Draw current position with Premiere style
        painter.setPen(QPen(QColor(ACCENT), 2))
        painter.drawLine(x_cur, 0, x_cur, h)
        
        # Draw playhead triangle
        triangle = QPolygon([
            QPoint(x_cur - 4, 0),
            QPoint(x_cur + 4, 0),
            QPoint(x_cur, 8)
        ])

        painter.setBrush(QBrush(QColor(ACCENT)))
        painter.setPen(QPen(QColor(ACCENT), 1))
        painter.drawPolygon(triangle)


class ModernButton(QPushButton):
    """Premiere Pro style button with professional theming"""
    
    def __init__(self, text, button_type="primary", parent=None):
        super().__init__(text, parent)
        self.button_type = button_type
        self.setFixedHeight(32)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.apply_style()
        
        # Subtle shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(6)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 1)
        self.setGraphicsEffect(shadow)
    
    def apply_style(self):
        if self.button_type == "primary":
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {ACCENT};
                    color: white;
                    border: 1px solid {ACCENT};
                    border-radius: 4px;
                    padding: 6px 14px;
                    font-size: 12px;
                    font-weight: 600;
                    font-family: 'Segoe UI', Tahoma, sans-serif;
                    min-width: 70px;
                }}
                QPushButton:hover {{
                    background: {ACCENT_HOVER};
                    border-color: {ACCENT_HOVER};
                }}
                QPushButton:pressed {{
                    background: {ACCENT};
                    transform: translateY(1px);
                }}
                QPushButton:disabled {{
                    background: {BORDER};
                    color: {SUBTEXT};
                    border-color: {BORDER};
                }}
            """)
        elif self.button_type == "secondary":
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {PANEL_BG};
                    color: {TEXT};
                    border: 1px solid {BORDER};
                    border-radius: 4px;
                    padding: 6px 14px;
                    font-size: 12px;
                    font-weight: 500;
                    font-family: 'Segoe UI', Tahoma, sans-serif;
                    min-width: 70px;
                }}
                QPushButton:hover {{
                    background: {BORDER};
                    border-color: {BORDER_LIGHT};
                }}
                QPushButton:pressed {{
                    background: {DARKER_BG};
                    transform: translateY(1px);
                }}
                QPushButton:disabled {{
                    background: {DARKER_BG};
                    color: {SUBTEXT};
                    border-color: {BORDER};
                }}
            """)
        elif self.button_type == "danger":
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {DANGER};
                    color: white;
                    border: 1px solid {DANGER};
                    border-radius: 4px;
                    padding: 6px 14px;
                    font-size: 12px;
                    font-weight: 600;
                    font-family: 'Segoe UI', Tahoma, sans-serif;
                    min-width: 70px;
                }}
                QPushButton:hover {{
                    background: #e63946;
                    border-color: #e63946;
                }}
                QPushButton:pressed {{
                    background: {DANGER};
                    transform: translateY(1px);
                }}
                QPushButton:disabled {{
                    background: {BORDER};
                    color: {SUBTEXT};
                }}
            """)


class ModernSlider(QSlider):
    """Premiere Pro style slider"""
    
    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {BORDER};
                height: 4px;
                border-radius: 2px;
                border: none;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT};
                border: 2px solid {PANEL_BG};
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {ACCENT_HOVER};
            }}
            QSlider::handle:horizontal:pressed {{
                background: {ACCENT};
            }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT};
                border-radius: 2px;
            }}
        """)


# Enhanced version of original EditorPanel class with Premiere Pro styling
class EnhancedEditorPanel(QWidget):
    """Enhanced version of the original EditorPanel with Premiere Pro styling"""
    
     # Same signals as original
    importRequested = pyqtSignal()
    playToggled = pyqtSignal(bool)
    frameChanged = pyqtSignal(int)
    detectRequested = pyqtSignal()
    blurRequested = pyqtSignal(int)
    thumbnailClicked = pyqtSignal(int)
    gestureItemClicked = pyqtSignal(object)
    exportRequested = pyqtSignal()

    # NEW: emitted when a video file is dropped onto the panel
    fileDropped = pyqtSignal(str)

    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.core_has_video = False
        
        # NEW: allow drag & drop onto the editor
        self.setAcceptDrops(True)

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

        # Apply modern Premiere Pro styles and build UI
        self.apply_premiere_styles()
        self._build_enhanced_ui()   

    def apply_premiere_styles(self):
        """Apply Premiere Pro inspired styling"""
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {APP_BG};
                color: {TEXT};
                font-family: 'Segoe UI', Tahoma, sans-serif;
                font-size: 12px;
            }}
            QLabel#video_display {{
                background: {DARKER_BG};
                border: 2px solid {BORDER};
                border-radius: 4px;
            }}
            QSlider::groove:horizontal {{
                background: {BORDER};
                height: 6px;
                border-radius: 3px;
                border: none;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT};
                border: 2px solid {PANEL_BG};
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {ACCENT_HOVER};
            }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT};
                border-radius: 3px;
            }}
            QPushButton {{
                background: {ACCENT};
                color: white;
                border: 1px solid {ACCENT};
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 600;
                min-width: 90px;
            }}
            QPushButton:hover {{
                background: {ACCENT_HOVER};
                border-color: {ACCENT_HOVER};
            }}
            QPushButton:pressed {{
                background: {ACCENT};
                transform: translateY(1px);
            }}
            QPushButton:disabled {{
                background: {BORDER};
                color: {SUBTEXT};
                border-color: {BORDER};
            }}
            QPushButton#importPopupBtn {{
                font-size: 16px;
                padding: 12px 24px;
                min-width: 180px;
                background: {ACCENT};
                border-radius: 6px;
            }}
            QPushButton#importPopupBtn:hover {{
                background: {ACCENT_HOVER};
            }}
            QListWidget {{
                background: {PANEL_BG};
                border: 1px solid {BORDER};
                border-radius: 4px;
                padding: 4px;
                selection-background-color: {ACCENT};
                outline: none;
            }}
            QListWidget::item {{
                padding: 8px 10px;
                border-bottom: 1px solid {BORDER};
                border-radius: 3px;
                margin: 1px 0;
            }}
            QListWidget::item:selected {{
                background: {ACCENT};
                color: white;
            }}
            QListWidget::item:hover:!selected {{ 
                background: {BORDER}; 
            }}
            QFrame#bottom_bar {{
                background: {PANEL_BG};
                border: 1px solid {BORDER};
                border-radius: 6px;
                padding: 8px;
            }}
            QScrollArea {{
                border: 1px solid {BORDER};
                border-radius: 4px;
                background: {PANEL_BG};
            }}
            QScrollBar:horizontal {{
                border: none;
                background: {PANEL_BG};
                height: 14px;
                border-radius: 7px;
            }}
            QScrollBar::handle:horizontal {{
                background: {ACCENT};
                border-radius: 7px;
                min-width: 20px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {ACCENT_HOVER};
            }}
        """)
    
    def _build_enhanced_ui(self):
        """Build the enhanced UI with Premiere Pro styling"""
        # Same structure as original but with enhanced styling
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # Middle: Video / Markers split
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background: {BORDER};
                border-radius: 1px;
            }}
            QSplitter::handle:hover {{
                background: {ACCENT};
            }}
        """)
        
        # Left side: video container
        video_container = QFrame()
        video_container.setStyleSheet(f"""
            QFrame {{
                background: {PANEL_BG};
                border: 1px solid {BORDER};
                border-radius: 6px;
                padding: 12px;
            }}
        """)
        
        # Professional shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(12)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(0, 4)
        video_container.setGraphicsEffect(shadow)
        
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(10)
        
        # Video display
        self.video_display = VideoViewport()
        self.video_display.setObjectName("video_display")
        self.video_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_display.setMinimumHeight(400)
        video_layout.addWidget(self.video_display)

        # Selection badge
        self.selection_badge = QLabel(self.video_display)
        self.selection_badge.setText("")
        self.selection_badge.setStyleSheet(f"""
            QLabel {{
                background: rgba(0, 120, 212, 200);
                color: white;
                font-weight: 600;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
            }}
        """)
        self.selection_badge.hide()
        self.selection_badge.move(8, 8)
        
        # Import popup overlay
        self.create_enhanced_import_popup(video_container)
        
        # Enhanced slider
        self.slider = ModernSlider(Qt.Horizontal)
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
        """Create Premiere Pro style import popup"""
        self.import_popup = QFrame(parent)
        self.import_popup.setStyleSheet(f"""
            QFrame {{
                background: rgba(35, 35, 35, 245);
                border: 2px dashed {ACCENT};
                border-radius: 8px;
            }}
        """)
        
        popup_layout = QVBoxLayout(self.import_popup)
        popup_layout.setContentsMargins(0, 0, 0, 0)
        popup_layout.setSpacing(0)
        popup_layout.addStretch()
        
        # Professional icon
        icon_label = QLabel("📁")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet(f"""
            QLabel {{
                font-size: 48px;
                background: none;
                border: none;
                color: {ACCENT};
                padding: 16px;
            }}
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
        
        # Professional help text
        help_text = QLabel("Drop a video file here or click to browse")
        help_text.setAlignment(Qt.AlignCenter)
        help_text.setStyleSheet(f"""
            QLabel {{
                color: {SUBTEXT};
                font-size: 13px;
                background: none;
                border: none;
                padding: 16px;
                font-weight: 400;
            }}
        """)
        popup_layout.addWidget(help_text)
        popup_layout.addStretch()
        
        self.import_popup.setGeometry(20, 20, 800, 500)
        self.import_popup.show()
    
    def create_enhanced_controls(self, parent_layout):
        """Create Premiere Pro style control buttons"""
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        
        btn_row.addStretch()
        
        self.toggle_button = ModernButton("▶ Play", "primary")
        self.toggle_button.setEnabled(False)
        self.toggle_button.clicked.connect(self._on_toggle_clicked)
        btn_row.addWidget(self.toggle_button)
        
        self.detect_button = ModernButton("🔍 Detect", "secondary")
        self.detect_button.setEnabled(False)
        self.detect_button.clicked.connect(lambda: self.detectRequested.emit())
        btn_row.addWidget(self.detect_button)
        
        self.blur_button = ModernButton("🔒 Blur", "secondary")
        self.blur_button.setEnabled(False)
        self.blur_button.clicked.connect(lambda: self.blurRequested.emit(self.current_frame_idx))
        btn_row.addWidget(self.blur_button)
        
        self.export_button = ModernButton("📤 Export", "primary")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(lambda: self.exportRequested.emit())
        btn_row.addWidget(self.export_button)
        
        btn_row.addStretch()
        
        parent_layout.addLayout(btn_row)
    #-----------------------------------------
    #to highlight the person
    def highlight_person_on_frame(self, bbox):
        """Draw a highlighted bounding box overlay on the current frame to show detected person"""
        if not bbox or len(bbox) != 4:
            return
        
        # Create a custom transparent widget for the highlight
        if not hasattr(self, 'highlight_widget'):
            class TransparentHighlight(QWidget):
                def __init__(self, parent):
                    super().__init__(parent)
                    self.setAttribute(Qt.WA_TransparentForMouseEvents)
                    self.setAttribute(Qt.WA_TranslucentBackground)
                    self.setStyleSheet("background: transparent;")
                    self.bbox_rect = None
                    self.pulse_value = 3  # Start at normal width
                    self.opacity = 255
                    
                def set_bbox(self, x, y, w, h):
                    self.bbox_rect = (x, y, w, h)
                    self.update()
                    
                def set_pulse(self, value):
                    self.pulse_value = value
                    self.opacity = int(180 + (value - 3) * 25)  # Vary opacity with pulse
                    self.update()
                    
                def paintEvent(self, event):
                    if not self.bbox_rect:
                        return
                        
                    painter = QPainter(self)
                    painter.setRenderHint(QPainter.Antialiasing)
                    
                    # Draw yellow outline with variable width from pulse
                    color = QColor(255, 215, 0, self.opacity)
                    pen = QPen(color, self.pulse_value)
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)  # No fill!
                    
                    x, y, w, h = self.bbox_rect
                    painter.drawRect(2, 2, w-4, h-4)  # Slight inset to avoid clipping
                    
                    # Draw corner accents (always bright)
                    corner_len = 20
                    accent_color = QColor(255, 215, 0, 255)  # Full opacity for accents
                    pen_thick = QPen(accent_color, self.pulse_value + 2)
                    painter.setPen(pen_thick)
                    
                    # Top-left
                    painter.drawLine(0, 0, corner_len, 0)
                    painter.drawLine(0, 0, 0, corner_len)
                    
                    # Top-right  
                    painter.drawLine(w-corner_len, 0, w, 0)
                    painter.drawLine(w-1, 0, w-1, corner_len)
                    
                    # Bottom-left
                    painter.drawLine(0, h-corner_len, 0, h)
                    painter.drawLine(0, h-1, corner_len, h-1)
                    
                    # Bottom-right
                    painter.drawLine(w-corner_len, h-1, w, h-1)
                    painter.drawLine(w-1, h-corner_len, w-1, h)
            
            self.highlight_widget = TransparentHighlight(self.video_display)
            self.highlight_widget.hide()
        
        # Calculate display coordinates from bbox
        if hasattr(self.video_display, '_ar') and self.video_display._ar:
            display_rect = self.video_display.rect()
            display_width = display_rect.width()
            display_height = display_rect.height()
            
            video_ar = self.video_display._ar
            current_ar = display_width / display_height if display_height > 0 else 1
            
            if current_ar > video_ar:
                actual_height = display_height
                actual_width = int(actual_height * video_ar)
                offset_x = (display_width - actual_width) // 2
                offset_y = 0
            else:
                actual_width = display_width
                actual_height = int(actual_width / video_ar)
                offset_x = 0
                offset_y = (display_height - actual_height) // 2
            
            if hasattr(self.video_display, '_qimg') and self.video_display._qimg:
                orig_width = self.video_display._qimg.width()
                orig_height = self.video_display._qimg.height()
            else:
                orig_width = 1920
                orig_height = int(orig_width / video_ar)
            
            x1, y1, x2, y2 = bbox
            scale_x = actual_width / orig_width
            scale_y = actual_height / orig_height
            
            display_x1 = int(x1 * scale_x) + offset_x
            display_y1 = int(y1 * scale_y) + offset_y
            display_width = int((x2 - x1) * scale_x)
            display_height = int((y2 - y1) * scale_y)
            
            # Set position and size
            self.highlight_widget.setGeometry(display_x1, display_y1, display_width, display_height)
            self.highlight_widget.set_bbox(0, 0, display_width, display_height)   # local to the overlay

            self.highlight_widget.show()
            
            # Start pulse animation
            self.start_pulse_animation()
            
            # Auto-hide after 4 seconds
            QTimer.singleShot(4000, self.hide_person_highlight)

        # --- Drag & drop support -------------------------------------------------
    def dragEnterEvent(self, event):
        """Accept video files dragged from Explorer/Finder."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if not url.isLocalFile():
                    continue
                path = url.toLocalFile()
                ext = os.path.splitext(path)[1].lower()
                if ext in (".mp4", ".mov", ".avi", ".mkv", ".m4v"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event):
        # Keep the "copy" cursor while moving over the widget
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        """Emit fileDropped(str) when a video file is dropped."""
        if not event.mimeData().hasUrls():
            event.ignore()
            return

        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            ext = os.path.splitext(path)[1].lower()
            if ext in (".mp4", ".mov", ".avi", ".mkv", ".m4v"):
                event.acceptProposedAction()
                self.fileDropped.emit(path)
                return

        event.ignore()


    def start_pulse_animation(self):
        """Create a pulsing effect by animating the border width"""
        if not hasattr(self, 'pulse_timer'):
            self.pulse_timer = QTimer()
            self.pulse_direction = 1
            self.pulse_value = 3
            
            def pulse_step():
                self.pulse_value += self.pulse_direction * 0.5
                if self.pulse_value >= 6:
                    self.pulse_value = 6
                    self.pulse_direction = -1
                elif self.pulse_value <= 3:
                    self.pulse_value = 3
                    self.pulse_direction = 1
                
                if hasattr(self, 'highlight_widget'):
                    self.highlight_widget.set_pulse(int(self.pulse_value))
            
            self.pulse_timer.timeout.connect(pulse_step)
        
        self.pulse_timer.start(50)  # Update every 50ms for smooth animation

    def hide_person_highlight(self):
        """Hide the person highlight overlay"""
        if hasattr(self, 'highlight_widget'):
            self.highlight_widget.hide()
        if hasattr(self, 'pulse_timer'):
            self.pulse_timer.stop()

    #---------------------------------
    def create_enhanced_markers_panel(self, parent_splitter):
        """
        Professional right sidebar with four cards: Video Properties, Detection Settings, Blur Settings, Detected Gestures
        """
        container = QFrame()
        container.setMinimumWidth(320)
        container.setMaximumWidth(400)
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        container.setStyleSheet(f"QFrame {{ background: {APP_BG}; border: 0; }}")

        # Professional shadow for the entire panel
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(-2, 0)
        container.setGraphicsEffect(shadow)

        root = QVBoxLayout(container)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # --- 1) Video Properties ---
        card_video = CardSection("Video Properties", container)
        self.lbl_fps = card_video.add_label_value("FPS:", "--")
        self.lbl_res = card_video.add_label_value("Resolution:", "--")
        self.lbl_dur = card_video.add_label_value("Duration:", "--")
        root.addWidget(card_video)

 
        # --- 3) Blur Settings ---
        card_blur = CardSection("Blur Settings", container)

        # Blur type radio buttons
        blur_type_label = QLabel("Blur Type:")
        blur_type_label.setObjectName("sfField")
        rb_row = QHBoxLayout()
        rb_row.setSpacing(10)
        self.rb_gauss = QRadioButton("Gaussian")
        self.rb_pixel = QRadioButton("Pixelate")
        self.rb_solid = QRadioButton("Solid")
        self.rb_gauss.setChecked(True)
        self.blur_type_group = QButtonGroup(card_blur)
        self.blur_type_group.addButton(self.rb_gauss, 0)
        self.blur_type_group.addButton(self.rb_pixel, 1)
        self.blur_type_group.addButton(self.rb_solid, 2)
        rb_row.addWidget(self.rb_gauss)
        rb_row.addWidget(self.rb_pixel)
        rb_row.addWidget(self.rb_solid)
        rb_row.addStretch(1)
        
        row_bt = QHBoxLayout()
        row_bt.setSpacing(8)
        row_bt.addWidget(blur_type_label)
        row_bt.addLayout(rb_row, 1)
        card_blur.inner_lay.addLayout(row_bt)

        # Blur strength slider
        self.blur_strength = QSlider(Qt.Horizontal)
        self.blur_strength.setMinimum(0)
        self.blur_strength.setMaximum(100)
        self.blur_strength_value = getattr(self, "blur_strength_value", 50)
        self.blur_strength.setValue(self.blur_strength_value)
        self.blur_strength.valueChanged.connect(self._on_strength_changed)
        self.lbl_strength_pct = QLabel(f"{self.blur_strength_value}%")
        self.lbl_strength_pct.setObjectName("sfPill")

        row_str = QHBoxLayout()
        row_str.setSpacing(8)
        lab_str = QLabel("Strength:")
        lab_str.setObjectName("sfField")
        row_str.addWidget(lab_str)
        row_str.addWidget(self.blur_strength, 1)
        row_str.addWidget(self.lbl_strength_pct)
        card_blur.inner_lay.addLayout(row_str)

        root.addWidget(card_blur)

        # --- 4) Detected Gestures ---
        card_g = CardSection("Detected Gestures", container)
        self.gesture_list = QListWidget()
        self.gesture_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.gesture_list.setMinimumHeight(120)
        self.gesture_list.setMaximumHeight(250)

        # Connect gesture list clicks
        self.gesture_list.itemClicked.connect(
            lambda it: self.gestureItemClicked.emit(
                ({"person_id": int(d[0]), "gesture": str(d[1]), "frame": int(d[2]), "bbox": d[3]}
                if isinstance((d := it.data(Qt.UserRole)), (tuple, list)) and len(d) >= 4 else
                d if isinstance(d, dict) else
                (int(d) if isinstance(d, (int, float)) else self.gesture_list.row(it)))
            )
        )
        
        card_g.inner_lay.addWidget(self.gesture_list)
        root.addWidget(card_g)
        root.addStretch(1)

        parent_splitter.addWidget(container)
        
        # Set splitter proportions
        try:
            parent_splitter.setStretchFactor(0, 3)  # video
            parent_splitter.setStretchFactor(1, 1)  # sidebar
            parent_splitter.setSizes([900, 400])
        except Exception:
            pass

    def _on_strength_changed(self, v: int):
        self.blur_strength_value = v
        if hasattr(self, "lbl_strength_pct"):
            self.lbl_strength_pct.setText(f"{v}%")

    def create_enhanced_bottom_bar(self, parent_layout):
        """Create Premiere Pro style timeline bar"""
        bottom_bar = QFrame()
        bottom_bar.setObjectName("bottom_bar")
        bottom_layout = QVBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(12, 8, 12, 8)
        bottom_layout.setSpacing(6)
        
        # Timeline ruler
        self.time_ruler = EnhancedTimeRuler()
        bottom_layout.addWidget(self.time_ruler)
        
        # Thumbnail timeline
        self.thumbnail_scroll = QScrollArea()
        self.thumbnail_scroll.setFixedHeight(80)
        self.thumbnail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.thumbnail_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.thumbnail_scroll.setWidgetResizable(True)
        
        thumb_container = QWidget()
        self.thumbnail_layout = QHBoxLayout(thumb_container)
        self.thumbnail_layout.setContentsMargins(6, 6, 6, 6)
        self.thumbnail_layout.setSpacing(4)    
        self.thumbnail_scroll.setWidget(thumb_container)
        
        bottom_layout.addWidget(self.thumbnail_scroll)
        parent_layout.addWidget(bottom_bar, stretch=0)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)

        # Keep the import overlay fitting the video area
        if hasattr(self, 'import_popup'):
            parent_rect = self.video_display.geometry()
            margin = 24
            self.import_popup.setGeometry(
                parent_rect.x() + margin,
                parent_rect.y() + margin,
                parent_rect.width() - 2 * margin,
                parent_rect.height() - 2 * margin
            )

        # Rescale the currently shown frame to the new label size
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

        # Audio setup if we already know the file path
        try:
            if getattr(self, "video_path", None):
                self.set_media_source(self.video_path)
        except Exception:
            pass

        # Update the Video Properties pills (FPS / Resolution / Duration)
        try:
            if hasattr(self, "lbl_fps"):
                self.lbl_fps.setText(f"{fps:.0f}" if fps else "--")

            if hasattr(self, "lbl_res"):
                self.lbl_res.setText("--")

            if hasattr(self, "lbl_dur"):
                duration = (total_frames / fps) if (fps and total_frames) else 0
                mm = int(duration // 60)
                ss = int(duration % 60)
                self.lbl_dur.setText(f"{mm:02d}:{ss:02d}" if duration else "--")
        except Exception:
            pass

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
            pix = QPixmap.fromImage(qimg).scaledToHeight(64, Qt.SmoothTransformation)

            thumb_label = QLabel()
            thumb_label.setPixmap(pix)
            thumb_label.setFixedSize(QSize(pix.width(), pix.height()))
            thumb_label.setCursor(QCursor(Qt.PointingHandCursor))
            thumb_label.setStyleSheet(f"""
                QLabel {{
                    border: 2px solid {BORDER};
                    border-radius: 3px;
                    padding: 1px;
                    background: {PANEL_BG};
                }}
                QLabel:hover {{
                    border-color: {ACCENT};
                    background: {BORDER};
                }}
            """)
            thumb_label.mousePressEvent = lambda e, i=idx: self.thumbnailClicked.emit(i)

            # Subtle shadow for thumbnails
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(4)
            shadow.setColor(QColor(0, 0, 0, 60))
            shadow.setOffset(0, 1)
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

        # Keep audio in lockstep with video transport
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

    #DETECTING GESTURE
    def start_detect_progress(self, total_steps: int = 100):
        # If a previous detect dialog exists (hidden/cancelled), delete it
        old = getattr(self, "_dlg_detect", None)
        if old is not None:
            try:
                old.hide()
                old.close()
                QTimer.singleShot(0, old.deleteLater)
            except Exception:
                pass
            self._dlg_detect = None

        # Fresh dialog for a fresh run
        self._dlg_detect = ProcessingDialog(
            self,
            title="Detecting Gestures",
            message="Analyzing video...",
            total_steps=total_steps,
            allow_cancel=True,          # ✖ sets cancelled=True and hides (non-destructive)
        )
        self._dlg_detect.show()
        QApplication.processEvents()


    def set_detect_progress(self, pct: int, label: str = None) -> bool:
        """Safe: returns False if dialog missing/cancelled/destroyed."""
        dlg = getattr(self, "_dlg_detect", None)
        if not dlg or dlg.was_cancelled():
            return False
        try:
            if label is not None:
                dlg.set_progress(int(max(0, min(100, pct))), label=label)
            else:
                dlg.set_progress(int(max(0, min(100, pct))))
            QApplication.processEvents()
            return True
        except RuntimeError:
            # e.g., user closed window between checks
            self._dlg_detect = None
            return False

    def finish_detect_progress(self, success: bool = True):
        dlg = getattr(self, "_dlg_detect", None)
        if not dlg:
            return
        try:
            if not dlg.was_cancelled():
                dlg.finish("Complete" if success else "Cancelled")
            dlg.close()
            QTimer.singleShot(0, dlg.deleteLater)
        except Exception:
            pass
        self._dlg_detect = None


    #EXPORTING
    def start_export_progress(self):
        self._dlg_export = ProcessingDialog(self, "Exporting", "Writing video file…", 100, False)
        self._dlg_export.show()
        QApplication.processEvents()

    def set_export_progress(self, pct: int):
        if hasattr(self, "_dlg_export"):
            self._dlg_export.set_progress(pct)
            QApplication.processEvents()

    def finish_export_progress(self, success=True):
        """Finish export progress dialog"""
        if hasattr(self, "_dlg_export"):
            self._dlg_export.set_progress(100)
            if success:
                self._dlg_export.set_message("Export complete")
            else:
                self._dlg_export.set_message("Export failed")
                
            QTimer.singleShot(300, lambda: (
                self._dlg_export.close() if hasattr(self, "_dlg_export") else None,
                delattr(self, "_dlg_export") if hasattr(self, "_dlg_export") else None
            ))

    def start_blur_progress(self):
        # allow_cancel=True so ✖ becomes "cancel"
        self._dlg_blur = ProcessingDialog(
            parent=self, title="Processing", message="Blurring person in video…",
            total_steps=100, allow_cancel=True
        )
        self._dlg_blur.show()
        QApplication.processEvents()

    def set_blur_progress(self, pct: int):
        dlg = getattr(self, "_dlg_blur", None)
        if not dlg or dlg.was_cancelled():
            return
        try:
            dlg.set_progress(int(max(0, min(100, pct))))
            QApplication.processEvents()
        except RuntimeError:
            # dialog got destroyed between checks
            self._dlg_blur = None

    def finish_blur_progress(self, success: bool = True):
        dlg = getattr(self, "_dlg_blur", None)
        if not dlg:
            return
        try:
            # If user didn’t cancel, show a short “Complete/Failed” then close
            if not dlg.was_cancelled():
                dlg.finish("Complete" if success else "Failed")
            # Close immediately; the dialog hides itself on cancel.
            dlg.close()
            # Ensure the widget is actually cleaned up soon, without blocking
            QTimer.singleShot(0, dlg.deleteLater)
        except Exception:
            pass
        self._dlg_blur = None

    # ---- Audio Methods (keeping existing functionality) ----
    def prepare_audio_for(self, path: str):
        """Try to load audio directly from the video. If DirectShow rejects it, extract a WAV with ffmpeg and load that instead."""
        

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
                # if ffprobe missing, we can't tell — let QMediaPlayer try first
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
            
            # If we're supposed to be playing, retry the audio playback
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
                
                # Schedule a retry of audio playback after fallback
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
            print(f"⚠ Audio pause error: {e}")

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


# Enhanced Progress Dialog for unified use
class PrettyProgress(QDialog):
    """Unified themed progress dialog with Premiere Pro styling."""
    def __init__(self, title="Processing", text="Please wait…", parent=None, determinate=False):
        super().__init__(parent)
        
        # Clean up title
        clean_title = title.replace("StopFilming", "").replace("–", "").strip()
        if not clean_title:
            clean_title = "StopFilming Pro"
            
        self.setWindowTitle(clean_title)
        self.setModal(True)
        self.setFixedWidth(480)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        # Layout
        v = QVBoxLayout(self)
        v.setContentsMargins(24, 20, 24, 20)
        v.setSpacing(12)

        # Clean header
        self.header = QLabel(text)
        self.header.setStyleSheet(f"""
            color: {ACCENT_LIGHT}; 
            font-weight: 600; 
            font-size: 16px;
            margin-bottom: 4px;
            font-family: 'Segoe UI', Tahoma, sans-serif;
        """)
        v.addWidget(self.header)

        # Progress bar
        self.bar = QProgressBar()
        self.bar.setMinimumHeight(24)
        v.addWidget(self.bar)

        # Time display
        self.time_lbl = QLabel("00:00")
        self.time_lbl.setAlignment(Qt.AlignRight)
        self.time_lbl.setStyleSheet(f"color: {SUBTEXT}; font-size: 12px;")
        v.addWidget(self.time_lbl)

        # Premiere Pro styling
        self.setStyleSheet(f"""
            QDialog {{ 
                background: {PANEL_BG}; 
                border: 1px solid {BORDER}; 
                border-radius: 8px;
                font-family: 'Segoe UI', Tahoma, sans-serif;
            }}
            QProgressBar {{
                height: 24px; 
                border: 1px solid {BORDER}; 
                border-radius: 4px;
                background: {DARKER_BG}; 
                text-align: center; 
                color: {TEXT};
                font-weight: 500;
                font-size: 12px;
            }}
            QProgressBar::chunk {{
                border-radius: 3px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, 
                    stop:0 {ACCENT}, stop:1 {ACCENT_LIGHT});
            }}
        """)

        # Mode setup
        self.determinate = determinate
        if determinate:
            self.bar.setRange(0, 100)
            self.bar.setTextVisible(True)
            self.bar.setFormat("%p%")
        else:
            self.bar.setRange(0, 100)
            self.bar.setTextVisible(False)
            self._pct = 0
            self._spin = QTimer(self)
            self._spin.timeout.connect(self._tick)
            self._spin.start(30)

        # Timer for elapsed time
        self._t = QTimer(self)
        self._t.timeout.connect(self._update_time)
        self._t.start(250)
        import time as _t
        self._since = _t.time()

    def _tick(self):
        self._pct = (self._pct + 2) % 101
        self.bar.setValue(self._pct)

    def _update_time(self):
        import time
        elapsed = int(time.time() - self._since)
        mm, ss = divmod(elapsed, 60)
        self.time_lbl.setText(f"{mm:02d}:{ss:02d}")

    def set_text(self, text):
        self.header.setText(text)

    def set_title(self, title):
        clean_title = title.replace("StopFilming", "").replace("–", "").strip()
        if not clean_title:
            clean_title = "StopFilming Pro"
        self.header.setText(clean_title)
        self.setWindowTitle(clean_title)

    def set_progress(self, pct: int):
        if self.determinate:
            self.bar.setValue(max(0, min(100, int(pct))))

    def closeEvent(self, e):
        if hasattr(self, "_spin"):
            self._spin.stop()
        if hasattr(self, "_t"):
            self._t.stop()
        super().closeEvent(e)


# Worker classes for threaded operations
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

    def __init__(self, core):
        super().__init__()
        self.core = core

    @pyqtSlot()
    def run(self):
        try:
            segments = self.core.detect_gestures()   # your heavy call
            self.finished.emit(segments)
        except Exception as e:
            self.error.emit(str(e))


class KeyboardShortcutsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts – StopFilming")
        self.setModal(True)
        self.setMinimumSize(520, 380)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        # Fallbacks in case these constants aren't imported above
        palette = globals()
        APP_BG       = palette.get("APP_BG", "#1E1E1E")
        PANEL_BG     = palette.get("PANEL_BG", "#252A32")
        TEXT         = palette.get("TEXT", "#E2E8F0")
        SUBTLE_TEXT  = palette.get("SUBTLE_TEXT", "#A0AEC0")
        BORDER       = palette.get("BORDER", "#3A4453")
        ACCENT       = palette.get("ACCENT", "#3B82F6")
        ACCENT_HOVER = palette.get("ACCENT_HOVER", "#2563EB")

        self.setStyleSheet(f"""
            QDialog {{
                background: {PANEL_BG};
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
            QLabel#title {{
                color: {ACCENT};
                font-size: 18px;
                font-weight: 700;
            }}
            QLabel#subtitle {{
                color: {SUBTLE_TEXT};
                font-size: 12px;
                padding-bottom: 4px;
            }}
            QLabel.shortcut {{
                color: {TEXT};
                font-family: Consolas, "SF Mono", Menlo, monospace;
                font-size: 13px;
                padding: 2px 8px;
                background: rgba(255,255,255,0.04);
                border: 1px solid {BORDER};
                border-radius: 6px;
            }}
            QLabel.action {{
                color: {TEXT};
                font-size: 13px;
                padding-left: 6px;
            }}
            QFrame#line {{
                background: {BORDER};
                height: 1px;
            }}
            QPushButton.primary {{
                background: {ACCENT};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                min-width: 96px;
            }}
            QPushButton.primary:hover {{ background: {ACCENT_HOVER}; }}
        """)

        # ---- Layout ---------------------------------------------------------
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        title = QLabel("Keyboard Shortcuts")
        title.setObjectName("title")
        root.addWidget(title)

        subtitle = QLabel("Handy keys to navigate, detect, blur and export quickly.")
        subtitle.setObjectName("subtitle")
        root.addWidget(subtitle)

        line = QFrame(); line.setObjectName("line"); line.setFixedHeight(1)
        root.addWidget(line)

        grid = QGridLayout()
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        root.addLayout(grid)

        def add_row(r, key, action):
            k = QLabel(key);    k.setObjectName("shortcut"); k.setProperty("class", "shortcut")
            a = QLabel(action); a.setObjectName("action");   a.setProperty("class", "action")
            # ensure pure text (no rich text/links/underline)
            k.setTextFormat(Qt.PlainText); a.setTextFormat(Qt.PlainText)
            grid.addWidget(k, r, 0, alignment=Qt.AlignLeft)
            grid.addWidget(a, r, 1, alignment=Qt.AlignLeft)

        rows = [
            ("Ctrl+O", "Open Video"),
            ("Ctrl+S", "Save Project"),
            ("Ctrl+E", "Export Video"),
            ("Space",  "Play / Pause"),
            ("Ctrl+D", "Detect Gestures"),
            ("Ctrl+B", "Blur Person"),
            ("F11",    "Toggle Fullscreen"),
            ("Ctrl+Q", "Quit"),
        ]
        for i, (k, a) in enumerate(rows):
            add_row(i, k, a)

        root.addStretch(1)

        line2 = QFrame(); line2.setObjectName("line"); line2.setFixedHeight(1)
        root.addWidget(line2)

        btns = QHBoxLayout()
        btns.addStretch(1)
        close_btn = QPushButton("Close"); close_btn.setObjectName("close")
        close_btn.setProperty("class", "primary")
        close_btn.clicked.connect(self.accept)
        btns.addWidget(close_btn)
        root.addLayout(btns)

# --- About dialog -------------------------------------------------------------
class AboutDialog(QDialog):
    def __init__(self, parent=None, version="v1.0", year="2025"):
        super().__init__(parent)
        self.setWindowTitle("About StopFilming")
        self.setModal(True)
        self.setMinimumWidth(480)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        

        # Fallbacks if tokens aren't in scope
        g = globals()
        APP_BG       = g.get("APP_BG", "#1E1E1E")
        PANEL_BG     = g.get("PANEL_BG", "#252A32")
        TEXT         = g.get("TEXT", "#E2E8F0")
        SUBTLE_TEXT  = g.get("SUBTLE_TEXT", "#A0AEC0")
        BORDER       = g.get("BORDER", "#3A4453")
        ACCENT       = g.get("ACCENT", "#3B82F6")
        ACCENT_HOVER = g.get("ACCENT_HOVER", "#2563EB")

        self.setStyleSheet(f"""
            QDialog {{
                background: {PANEL_BG};
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
            QLabel#title {{
                color: {ACCENT};
                font-size: 20px;
                font-weight: 700;
            }}
            QLabel#subtitle {{
                color: {SUBTLE_TEXT};
                font-size: 12px;
            }}
            QLabel#meta {{
                color: {TEXT};
                font-size: 13px;
            }}
            QLabel#link {{
                color: {ACCENT};
                font-size: 13px;
            }}
            QFrame#line {{ background: {BORDER}; height: 1px; }}
            QPushButton.primary {{
                background: {ACCENT};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                min-width: 96px;
            }}
            QPushButton.primary:hover {{ background: {ACCENT_HOVER}; }}
            QPushButton.ghost {{
                background: transparent;
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
            }}
            QPushButton.ghost:hover {{ background: rgba(255,255,255,0.04); }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        # Header row with icon + title
        header = QHBoxLayout()
        icon_lbl = QLabel()
        # Use the app/window icon if available
        pix = (parent.windowIcon().pixmap(48, 48) if parent and parent.windowIcon()
               else QIcon().pixmap(48, 48))
        icon_lbl.setPixmap(pix)
        icon_lbl.setFixedSize(48, 48)

        title_box = QVBoxLayout()
        title = QLabel("StopFilming")
        title.setObjectName("title")
        subtitle = QLabel("Privacy Protection Video Editor")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        header.addWidget(icon_lbl, 0, Qt.AlignTop)
        header.addLayout(title_box)
        header.addStretch(1)
        root.addLayout(header)

        line = QFrame(); line.setObjectName("line"); line.setFixedHeight(1)
        root.addWidget(line)

        meta = QLabel(f"Version {version}  •  © {year}")
        meta.setObjectName("meta")
        root.addWidget(meta)

        # Links / info
        # "Documentation" label link
        link = QLabel('<a href="#">Documentation</a>')
        link.setObjectName("link")
        link.setOpenExternalLinks(False)
        link.linkActivated.connect(lambda _: open_docs_html_or_web("https://example.com/stopfilming/docs"))

        root.addStretch(1)
        line2 = QFrame(); line2.setObjectName("line"); line2.setFixedHeight(1)
        root.addWidget(line2)

        btns = QHBoxLayout()
        btns.addStretch(1)
        more_btn = QPushButton("Website"); more_btn.setProperty("class", "ghost")
        more_btn.clicked.connect(lambda: open_docs_html_or_web())

        ok_btn = QPushButton("OK"); ok_btn.setProperty("class", "primary")
        ok_btn.clicked.connect(self.accept)

        btns.addWidget(more_btn)
        btns.addWidget(ok_btn)
        root.addLayout(btns)

        
    def _open_readme(self):
        try:
            dlg = ReadmeDialog(self)
            dlg.exec_()
        except Exception as e:
            QMessageBox.information(self, "Documentation", f"Could not open README:\n{e}")


    def _open_url(self, url):
        open_docs_html_or_web("https://example.com/stopfilming/docs")


class ReadmeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("README – StopFilming")
        self.setModal(True)
        self.resize(760, 560)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        g = globals()
        PANEL_BG     = g.get("PANEL_BG", "#252A32")
        TEXT         = g.get("TEXT", "#E2E8F0")
        SUBTLE_TEXT  = g.get("SUBTLE_TEXT", "#A0AEC0")
        BORDER       = g.get("BORDER", "#3A4453")
        ACCENT       = g.get("ACCENT", "#3B82F6")
        ACCENT_HOVER = g.get("ACCENT_HOVER", "#2563EB")

        self.setStyleSheet(f"""
            QDialog {{ background:{PANEL_BG}; color:{TEXT}; border:1px solid {BORDER}; border-radius:10px; }}
            QLabel#title {{ color:{ACCENT}; font-size:18px; font-weight:700; }}
            QFrame#line {{ background:{BORDER}; height:1px; }}
            QTextBrowser {{ background:transparent; color:{TEXT}; border:none; font-size:14px; }}
            QPushButton.primary {{ background:{ACCENT}; color:white; border:none; border-radius:6px; padding:8px 16px; font-weight:600; }}
            QPushButton.primary:hover {{ background:{ACCENT_HOVER}; }}
        """)

        v = QVBoxLayout(self); v.setContentsMargins(16,14,16,14); v.setSpacing(10)
        title = QLabel("README"); title.setObjectName("title")
        v.addWidget(title)
        line = QFrame(); line.setObjectName("line"); line.setFixedHeight(1)
        v.addWidget(line)

        self.viewer = QTextBrowser()
        self.viewer.setOpenExternalLinks(True)
        v.addWidget(self.viewer, 1)

        btns = QHBoxLayout(); btns.addStretch(1)
        close = QPushButton("Close"); close.setProperty("class","primary"); close.clicked.connect(self.accept)
        btns.addWidget(close); v.addLayout(btns)

        self._load_readme()

    def _readme_path(self):
        base = os.path.dirname(os.path.abspath(sys.argv[0]))
        for name in ("README.md", "Readme.md", "readme.md"):
            p = os.path.join(base, name)
            if os.path.exists(p): return p
        return None

    def _load_readme(self):
        p = self._readme_path()
        if not p:
            self.viewer.setPlainText("README.md not found next to the app.")
            return
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # Qt ≥ 5.14 supports Markdown; earlier falls back to plain text
        if hasattr(self.viewer, "setMarkdown"):
            self.viewer.setMarkdown(text)
        else:
            self.viewer.setPlainText(text)




# ============================================================================
# IMPORTANT: main.py uses EditorPanel, which is aliased to EnhancedEditorPanel
# The EnhancedVideoEditor class above is NOT used - it's an alternative 
# implementation with a frameless window and custom title bar.
# ============================================================================
# Alias for compatibility with main.py
EditorPanel = EnhancedEditorPanel