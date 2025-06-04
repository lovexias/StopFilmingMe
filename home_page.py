# home_page.py

import os
import json
import time
import cv2

from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QSizePolicy,
    QSplitter,
    QStyle,
    QMessageBox
)
from PyQt5.QtGui import QPixmap, QCursor
from PyQt5.QtCore import Qt, QSize

#
# ─── Helper Functions for Recent Projects ───
#

def add_to_recent_projects(video_path):
    home = os.path.expanduser("~")
    base_dir = os.path.join(home, ".stopfilming")
    recent_file = os.path.join(base_dir, "recent_projects.json")
    thumb_dir = os.path.join(base_dir, "thumbs")
    os.makedirs(thumb_dir, exist_ok=True)

    try:
        with open(recent_file, "r", encoding="utf-8") as f:
            recents = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        recents = []

    recents = [r for r in recents if r.get("path") != video_path]

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

    recents.insert(0, {
        "path": video_path,
        "timestamp": int(time.time()),
        "thumbnail": thumb_path
    })
    recents = recents[:5]

    os.makedirs(os.path.dirname(recent_file), exist_ok=True)
    with open(recent_file, "w", encoding="utf-8") as f:
        json.dump(recents, f, indent=2)


def get_recent_projects():
    home = os.path.expanduser("~")
    recent_file = os.path.join(home, ".stopfilming", "recent_projects.json")
    try:
        with open(recent_file, "r", encoding="utf-8") as f:
            recents = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        recents = []
    return recents


#
# ─── HomePage Definition ───
#

class HomePage(QWidget):
    def __init__(self, import_callback, continue_callback):
        super().__init__()
        self.import_callback = import_callback
        self.continue_callback = continue_callback
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet("background-color: #1E1E1E;")
        self.setContentsMargins(0, 0, 0, 0)

        # ─── Left Icon Bar ───
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

        # ─── Center Drop Area + Import Button ───
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

        # ─── Right Panel ───
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

        # ─── Top Splitter ───
        self.top_splitter = QSplitter(Qt.Horizontal)
        self.top_splitter.addWidget(left_icon_widget)
        self.top_splitter.addWidget(center_widget)
        self.top_splitter.addWidget(right_widget)
        self.top_splitter.setSizes([60, 650, 300])
        self.top_splitter.setHandleWidth(1)

        # ─── Bottom: Recent Projects ───
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
                pix = QPixmap(thumb_path).scaled(
                    150, 70, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
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

        while len(recents) < 5:
            spacer = QLabel("")
            spacer.setFixedSize(160, 100)
            self.bottom_cards_layout.addWidget(spacer)
            recents.append(None)

        bottom_cards_widget = QWidget()
        bottom_cards_widget.setLayout(self.bottom_cards_layout)

        # ─── Final Layout ───
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
        self.continue_callback()

    def _open_recent_project(self, video_path):
        if os.path.exists(video_path):
            add_to_recent_projects(video_path)
            self.continue_callback()
            self.import_callback(video_path)
