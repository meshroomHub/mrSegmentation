#!/usr/bin/env python3
"""
Visualiseur de séquences EXR avec overlay de masques
Nécessite: PySide6, OpenImageIO, numpy
"""

import sys
import os
import re
import glob
import numpy as np
import OpenImageIO as oiio
from collections import OrderedDict
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QFileDialog, QSpinBox, QGroupBox,
    QCheckBox, QStatusBar, QStyle, QStyleOptionSlider, QMessageBox,
    QInputDialog
)
from PySide6.QtCore import Qt, Signal, QTimer, QRect
from PySide6.QtGui import QImage, QPixmap, QKeyEvent, QMouseEvent, QPainter, QColor, QPen
from mask_label_manager import (
    scan_available_labels, find_label_sequence_files,
    obj_id_from_color, color_from_obj_id,
    BBoxesManager, copy_bboxes_json, BBOXES_JSON_NAME
)

from sam_integration import SamSession
import shutil
import cv2

# ---------------------- Constantes fixées ----------------------
FIXED_OVERLAY_OPACITY = 0.5
FIXED_COLOR_TOLERANCE = 0.01
FIXED_EXPOSURE = 0.0
FIXED_GAMMA = 2.2
DEFAULT_FPS = 24


def find_sequence_files(folder):
    """Trouve les séquences .####.exr dans un dossier et retourne un dict {frame_num: filepath}"""
    pattern = os.path.join(folder, "*.exr")
    files = glob.glob(pattern)

    frame_re = re.compile(r"\.(\d+)\.exr$", re.IGNORECASE)
    frames = {}

    for f in files:
        m = frame_re.search(os.path.basename(f))
        if m:
            frame_num = int(m.group(1))
            frames[frame_num] = f

    return frames


def load_exr_raw(filepath):
    """Charge un EXR et retourne l'array float brut (H, W, C) + dimensions"""
    inp = oiio.ImageInput.open(filepath)
    if not inp:
        raise IOError(f"Impossible d'ouvrir {filepath}: {oiio.geterror()}")

    spec = inp.spec()
    width = spec.width
    height = spec.height
    nchannels = spec.nchannels

    pixels = inp.read_image()
    inp.close()

    if pixels is None:
        raise IOError(f"Impossible de lire les pixels de {filepath}")

    arr = np.array(pixels, dtype=np.float32).reshape(height, width, nchannels)
    return arr, width, height, nchannels


def raw_to_rgb8(arr, nchannels, exposure=0.0, gamma=2.2):
    """Convertit un array float brut en RGB uint8 (avec tonemapping) + retourne l'alpha float"""
    height, width = arr.shape[0], arr.shape[1]

    if nchannels >= 3:
        rgb = arr[:, :, :3].copy()
        alpha = arr[:, :, 3].copy() if nchannels >= 4 else np.ones((height, width), dtype=np.float32)
    else:
        rgb = np.repeat(arr[:, :, 0:1], 3, axis=2)
        alpha = np.ones((height, width), dtype=np.float32)

    rgb = rgb * (2.0 ** exposure)
    rgb = np.clip(rgb, 0.0, None)
    rgb = np.power(rgb, 1.0 / gamma)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    return rgb, alpha


def compute_presence_alpha(mask_raw):
    """
    Calcule un alpha binaire basé sur la présence du masque
    (pixel non-noir / non-vide) plutôt que sur son canal alpha.
    """
    rgb = mask_raw[:, :, :3]
    magnitude = np.sum(np.abs(rgb), axis=2)
    presence = (magnitude > 1e-6).astype(np.float32)
    return presence


class FrameCache:
    """Cache LRU simple pour stocker les données brutes (float) des frames chargées"""

    def __init__(self, max_size=100):
        self.max_size = max_size
        self._cache = OrderedDict()

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key, value):
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)

    def __contains__(self, key):
        return key in self._cache

class RangeSlider(QSlider):
    """
    QSlider personnalisé qui affiche visuellement un intervalle actif (in/out)
    en grisant les zones hors intervalle et en marquant les bornes.
    Le clic sur la rainure saute directement à la position cliquée (plutôt que
    par page step, qui est le comportement par défaut de QSlider).
    """

    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.range_start = 0
        self.range_end = 0

    def set_active_range(self, start, end):
        self.range_start = start
        self.range_end = end
        self.update()

    def _value_from_pos(self, pos_x):
        """Calcule la valeur du slider correspondant à une position x en pixels"""
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        groove_rect = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self
        )
        handle_rect = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self
        )

        # Largeur utile : la rainure moins la largeur du handle, car le centre du
        # handle ne peut pas atteindre tout à fait les bords de la rainure
        slider_min = groove_rect.x() + handle_rect.width() / 2
        slider_max = groove_rect.right() - handle_rect.width() / 2
        slider_length = max(1.0, slider_max - slider_min)

        ratio = (pos_x - slider_min) / slider_length
        ratio = min(max(ratio, 0.0), 1.0)

        value = self.minimum() + round(ratio * (self.maximum() - self.minimum()))
        return int(value)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            handle_rect = self.style().subControlRect(
                QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self
            )

            pos_x = event.position().x() if hasattr(event, "position") else event.x()

            # Si le clic est directement sur le handle, on laisse le comportement
            # natif de QSlider gérer le drag normalement
            if handle_rect.contains(int(pos_x), handle_rect.center().y()):
                super().mousePressEvent(event)
                return

            # Sinon, saut direct à la position cliquée
            new_value = self._value_from_pos(pos_x)
            self.setValue(new_value)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Permet de continuer à glisser après le clic initial, en suivant la souris
        if event.buttons() & Qt.LeftButton:
            pos_x = event.position().x() if hasattr(event, "position") else event.x()
            new_value = self._value_from_pos(pos_x)
            self.setValue(new_value)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        groove_rect = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self
        )

        minimum = self.minimum()
        maximum = self.maximum()
        span = max(1, maximum - minimum)

        groove_x = groove_rect.x()
        groove_w = groove_rect.width()

        def value_to_x(value):
            ratio = (value - minimum) / span
            return int(groove_x + ratio * groove_w)

        bar_y = groove_rect.y()
        bar_h = groove_rect.height()

        painter.fillRect(QRect(groove_x, bar_y, groove_w, bar_h), QColor(60, 60, 60))

        start_x = value_to_x(self.range_start)
        end_x = value_to_x(self.range_end)
        active_rect = QRect(start_x, bar_y, max(1, end_x - start_x), bar_h)
        painter.fillRect(active_rect, QColor(90, 140, 200))

        painter.setPen(QColor(30, 30, 30))
        painter.drawRect(QRect(groove_x, bar_y, groove_w - 1, bar_h - 1))

        pen = QPen(QColor(255, 200, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(start_x, bar_y - 3, start_x, bar_y + bar_h + 3)
        painter.drawLine(end_x, bar_y - 3, end_x, bar_y + bar_h + 3)

        handle_opt = QStyleOptionSlider()
        self.initStyleOption(handle_opt)
        handle_opt.subControls = QStyle.SC_SliderHandle
        self.style().drawComplexControl(QStyle.CC_Slider, handle_opt, painter, self)

        painter.end()

class ImageViewer(QWidget):
    clicked_normalized = Signal(float, float)
    right_clicked_normalized = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(100, 100)
        self.label.setStyleSheet("background-color: #222;")
        self.label.setCursor(Qt.CrossCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

        self._current_pixmap = None
        self._points_to_draw = []  # liste de dicts {x_norm, y_norm, positive}
        self.label.mousePressEvent = self._on_label_click

    def set_image(self, rgb_array):
        h, w, _ = rgb_array.shape
        rgb_array = np.ascontiguousarray(rgb_array)
        qimg = QImage(rgb_array.data, w, h, w * 3, QImage.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(qimg.copy())
        self._update_display()

    def set_points(self, points):
        """
        points: liste de dicts {'x_norm': float, 'y_norm': float, 'positive': bool}
        en coordonnées normalisées [0,1] par rapport à l'image originale.
        """
        self._points_to_draw = points
        self._update_display()

    def clear_points(self):
        self._points_to_draw = []
        self._update_display()

    def resizeEvent(self, event):
        self._update_display()
        super().resizeEvent(event)

    def _update_display(self):
        if self._current_pixmap is None:
            return

        scaled = self._current_pixmap.scaled(
            self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        if self._points_to_draw:
            scaled = self._draw_points_on_pixmap(scaled)

        self.label.setPixmap(scaled)

    def _draw_points_on_pixmap(self, pixmap):
        """Dessine les points positifs/négatifs par-dessus le pixmap scalé"""
        result = QPixmap(pixmap)  # copie pour ne pas modifier le pixmap original en cache
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing)

        w = result.width()
        h = result.height()
        radius = max(4, min(w, h) // 100)  # taille adaptative selon la résolution affichée

        for pt in self._points_to_draw:
            px = pt["x_norm"] * w
            py = pt["y_norm"] * h

            if pt["positive"]:
                fill_color = QColor(40, 220, 40)
            else:
                fill_color = QColor(220, 40, 40)

            # Contour blanc pour la visibilité sur fonds clairs/foncés
            pen = QPen(QColor(255, 255, 255))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(fill_color)
            painter.drawEllipse(int(px - radius), int(py - radius), radius * 2, radius * 2)

            # Petite croix au centre pour les points négatifs (accessibilité daltonisme)
            if not pt["positive"]:
                pen2 = QPen(QColor(255, 255, 255))
                pen2.setWidth(2)
                painter.setPen(pen2)
                cross_size = radius * 0.6
                painter.drawLine(
                    int(px - cross_size), int(py - cross_size),
                    int(px + cross_size), int(py + cross_size)
                )
                painter.drawLine(
                    int(px - cross_size), int(py + cross_size),
                    int(px + cross_size), int(py - cross_size)
                )

        painter.end()
        return result

    def _on_label_click(self, event: QMouseEvent):
        if self._current_pixmap is None or self.label.pixmap() is None:
            return

        pixmap = self.label.pixmap()
        pw, ph = pixmap.width(), pixmap.height()
        lw, lh = self.label.width(), self.label.height()

        offset_x = (lw - pw) / 2.0
        offset_y = (lh - ph) / 2.0

        click_x = event.position().x() if hasattr(event, "position") else event.x()
        click_y = event.position().y() if hasattr(event, "position") else event.y()

        rel_x = click_x - offset_x
        rel_y = click_y - offset_y

        if 0 <= rel_x < pw and 0 <= rel_y < ph:
            norm_x = rel_x / pw
            norm_y = rel_y / ph
            if event.button() == Qt.RightButton:
                self.right_clicked_normalized.emit(norm_x, norm_y)
            else:
                self.clicked_normalized.emit(norm_x, norm_y)

class MainWindow(QMainWindow):
    def __init__(self, base_frame_paths=None, mask_folder=None, bboxes_json_path=None):
        """
        base_frame_paths: liste optionnelle de chemins de fichiers EXR (frames de base),
                           déjà nommés selon la convention .####.exr
        mask_folder: chemin optionnel vers un dossier de masques (scanné comme si
                     l'utilisateur l'avait choisi via le bouton "Choisir dossier MASQUES")
        bboxes_json_path: chemin optionnel vers un fichier bboxes.json à utiliser
                           explicitement, au lieu de celui trouvé automatiquement
                           dans mask_folder (s'il existe un bboxes.json dans ce dossier,
                           bboxes_json_path a priorité)
        """
        super().__init__()
        self.setWindowTitle("EXR Sequence Viewer - Base + Masks Overlay")
        self.resize(1200, 800)

        self.launch_with_params = base_frame_paths is not None or mask_folder is not None

        self.base_folder = None
        self.mask_folder = None
        self.base_frames = {}
        self.mask_frames = {}
        self.frame_list = []

        self.show_overlay = True

        self.color_pick_mode = False
        self.selected_color = None
        self.selected_obj_id = None

        self.base_cache = FrameCache(max_size=100)
        self.mask_cache = FrameCache(max_size=100)

        self.is_playing = False
        self.loop_playback = False
        self.fps = DEFAULT_FPS
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._on_play_tick)

        self.in_index = 0
        self.out_index = 0

        self.working_mask_folder = None
        self.current_label = None
        self.current_tracking = None
        self.bboxes_manager = None

        self.sam_mode_active = False
        self.sam_session = None
        self.sam_reference_sam_idx = 0
        self.sam_points = []
        self.sam_current_mask = None
        self.sam_propagated_masks = {}

        self._build_ui()

        # --- Initialisation programmatique optionnelle ---
        if self.launch_with_params:
            self._init_from_params(base_frame_paths, mask_folder, bboxes_json_path)

    # ---------------------- Initialisation programmatique ----------------------

    def _init_from_params(self, base_frame_paths, mask_folder, bboxes_json_path):
        """
        Initialise la séquence directement depuis des paramètres fournis au constructeur,
        en réutilisant exactement la même logique que les sélections manuelles via l'UI.
        """
        if base_frame_paths is not None:
            self.base_frames = self._build_frame_dict_from_paths(base_frame_paths)
            self.base_cache.clear()
            self.status.showMessage(f"{len(self.base_frames)} images fournies en paramètre")

        if mask_folder is not None:
            self._load_mask_folder(mask_folder, explicit_bboxes_path=bboxes_json_path)

        self.try_build_sequence()

    def _build_frame_dict_from_paths(self, paths):
        """
        Construit un dict {frame_num: filepath} à partir d'une liste de chemins fournie,
        en appliquant le même regex de parsing que find_sequence_files().
        find_sequence_files() est utilisé en backup si la liste ne contient qu'un chemin
        et que c'est un dossier ou que paths est directement un folder
        """
        if isinstance(paths, list) and len(paths) == 1 and os.path.isdir(paths[0]):
            return find_sequence_files(paths[0])
        elif isinstance(paths, str) and os.path.isdir(paths):
            return find_sequence_files(paths)

        frame_re = re.compile(r"\.(\d+)\.exr$", re.IGNORECASE)
        frames = {}
        for path in paths:
            m = frame_re.search(os.path.basename(path))
            if m:
                frame_num = int(m.group(1))
                frames[frame_num] = path
        return frames

    def _load_mask_folder(self, folder, explicit_bboxes_path=None):
        """
        Factorise la logique de chargement d'un dossier de masques : détection du label
        (avec boîte de dialogue si ambiguïté), construction de la séquence filtrée,
        et initialisation du BBoxesManager. Utilisée à la fois par select_mask_folder()
        (sélection manuelle) et _init_from_params() (initialisation programmatique).

        Le dossier de masques fourni est considéré comme le dossier de travail directement
        (pas de copie supplémentaire créée), conformément au mode d'initialisation
        programmatique.
        """
        self.mask_folder = folder

        labels_info = scan_available_labels(folder)

        if not labels_info:
            QMessageBox.warning(self, "Aucun masque détecté",
                                 "Aucun fichier de masque reconnu (format colorMask_<label>_<tracking>_...) "
                                 "n'a été trouvé dans ce dossier.")
            return False

        if len(labels_info) == 1:
            chosen_label = next(iter(labels_info))
        else:
            labels_list = sorted(labels_info.keys())
            chosen_label, ok = QInputDialog.getItem(
                self, "Choix du label",
                "Plusieurs labels de masques ont été détectés, lequel charger ?",
                labels_list, 0, False
            )
            if not ok:
                return False

        self.current_label = chosen_label
        self.current_tracking = labels_info[chosen_label]["tracking"]
        rest = labels_info[chosen_label]["rest"]

        self.mask_frames = find_label_sequence_files(folder, chosen_label, self.current_tracking, rest)
        self.mask_cache.clear()

        self.status.showMessage(
            f"{len(self.mask_frames)} masques trouvés pour le label '{chosen_label}' "
            f"(tracking: {self.current_tracking}) dans {folder}"
        )

        # Le dossier de masques fourni EST le dossier de travail : pas de copie créée
        self.working_mask_folder = folder

        bboxes_path = explicit_bboxes_path if explicit_bboxes_path is not None \
            else os.path.join(folder, BBOXES_JSON_NAME)

        if os.path.exists(bboxes_path):
            self.bboxes_manager = BBoxesManager(bboxes_path, chosen_label)
        else:
            self.bboxes_manager = None
            self.status.showMessage(
                f"{len(self.mask_frames)} masques trouvés pour le label '{chosen_label}' "
                f"(tracking: {self.current_tracking}) dans {folder} — "
                f"aucun fichier bboxes.json trouvé à '{bboxes_path}'"
            )

        return True

    # ---------------------- UI ----------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # --- Boutons de sélection de dossiers et cache management---
        folder_layout = QHBoxLayout()

        if not self.launch_with_params:

            self.btn_base = QPushButton("Choisir dossier IMAGES")
            self.btn_base.clicked.connect(self.select_base_folder)
            folder_layout.addWidget(self.btn_base)

            self.btn_mask = QPushButton("Choisir dossier MASQUES")
            self.btn_mask.clicked.connect(self.select_mask_folder)
            folder_layout.addWidget(self.btn_mask)

            self.chk_use_working_folder = QCheckBox("Créer dossier de travail")
            self.chk_use_working_folder.setChecked(True)
            folder_layout.addWidget(self.chk_use_working_folder)

        self.cache_status_label = QLabel("Cache: 0 images / 0 masques")
        folder_layout.addWidget(self.cache_status_label)

        self.btn_preload = QPushButton("Précharger toute la séquence")
        self.btn_preload.clicked.connect(self.preload_sequence)
        folder_layout.addWidget(self.btn_preload)

        self.btn_clear_cache = QPushButton("Vider le cache")
        self.btn_clear_cache.clicked.connect(self.clear_caches)
        folder_layout.addWidget(self.btn_clear_cache)

        main_layout.addLayout(folder_layout)

        # --- Viewer ---
        self.viewer = ImageViewer()
        self.viewer.clicked_normalized.connect(self.on_viewer_left_click)
        self.viewer.right_clicked_normalized.connect(self.on_viewer_right_click)
        
        main_layout.addWidget(self.viewer, stretch=1)

        # --- Contrôles overlay (simplifié) ---
        overlay_group = QGroupBox("Overlay")
        overlay_layout = QHBoxLayout(overlay_group)

        self.chk_overlay = QCheckBox("Afficher overlay")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.stateChanged.connect(self.on_overlay_toggle)
        overlay_layout.addWidget(self.chk_overlay)

        self.color_status_label = QLabel("Aucune sélection de couleur")
        overlay_layout.addWidget(self.color_status_label)

        self.btn_reset_selection = QPushButton("Réinitialiser sélection")
        self.btn_reset_selection.clicked.connect(self.reset_color_selection)
        overlay_layout.addWidget(self.btn_reset_selection)

        overlay_layout.addStretch()
        main_layout.addWidget(overlay_group)

        # --- Groupe SAM2 ---
        sam_group = QGroupBox("Re-segmentation SAM2")
        sam_layout = QVBoxLayout(sam_group)

        sam_top_row = QHBoxLayout()
        self.btn_sam_start = QPushButton("Segmenter avec SAM2")
        self.btn_sam_start.clicked.connect(self.start_sam_session)
        sam_top_row.addWidget(self.btn_sam_start)

        self.sam_status_label = QLabel("")
        sam_top_row.addWidget(self.sam_status_label)
        sam_top_row.addStretch()
        sam_layout.addLayout(sam_top_row)

        # Ligne visible seulement en mode SAM2
        self.sam_controls_widget = QWidget()
        sam_controls_layout = QHBoxLayout(self.sam_controls_widget)
        sam_controls_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_sam_step_bwd = QPushButton("← 1 frame")
        self.btn_sam_step_bwd.clicked.connect(self.sam_step_backward)
        sam_controls_layout.addWidget(self.btn_sam_step_bwd)

        self.btn_sam_step10_bwd = QPushButton("← 10 frames")
        self.btn_sam_step10_bwd.clicked.connect(self.sam_step10_backward)
        sam_controls_layout.addWidget(self.btn_sam_step10_bwd)

        self.btn_sam_propagate_bwd = QPushButton("← Propager arrière")
        self.btn_sam_propagate_bwd.clicked.connect(self.sam_propagate_backward)
        sam_controls_layout.addWidget(self.btn_sam_propagate_bwd)

        self.btn_sam_propagate_fwd = QPushButton("Propager avant →")
        self.btn_sam_propagate_fwd.clicked.connect(self.sam_propagate_forward)
        sam_controls_layout.addWidget(self.btn_sam_propagate_fwd)

        self.btn_sam_step10_fwd = QPushButton("10 frames →")
        self.btn_sam_step10_fwd.clicked.connect(self.sam_step10_forward)
        sam_controls_layout.addWidget(self.btn_sam_step10_fwd)

        self.btn_sam_step_fwd = QPushButton("1 frame →")
        self.btn_sam_step_fwd.clicked.connect(self.sam_step_forward)
        sam_controls_layout.addWidget(self.btn_sam_step_fwd)

        self.btn_sam_propagate_all = QPushButton("Propager tout")
        self.btn_sam_propagate_all.clicked.connect(self.sam_propagate_all)
        sam_controls_layout.addWidget(self.btn_sam_propagate_all)

        self.btn_sam_clear_points = QPushButton("Effacer points (frame actuelle)")
        self.btn_sam_clear_points.clicked.connect(self.sam_clear_points_current_frame)
        sam_controls_layout.addWidget(self.btn_sam_clear_points)

        sam_controls_layout.addStretch()

        self.btn_sam_validate = QPushButton("✓ Valider")
        self.btn_sam_validate.setStyleSheet("font-weight: bold; color: green;")
        self.btn_sam_validate.clicked.connect(self.sam_validate_and_exit)
        sam_controls_layout.addWidget(self.btn_sam_validate)

        self.btn_sam_cancel = QPushButton("✗ Annuler")
        self.btn_sam_cancel.setStyleSheet("color: red;")
        self.btn_sam_cancel.clicked.connect(self.sam_cancel_and_exit)
        sam_controls_layout.addWidget(self.btn_sam_cancel)

        sam_layout.addWidget(self.sam_controls_widget)
        self.sam_controls_widget.setEnabled(False)  # disabled tant qu'on n'est pas en mode SAM2

        main_layout.addWidget(sam_group)

        # --- Intervalle de lecture (in/out) ---
        range_group = QGroupBox("Intervalle de lecture")
        range_layout = QHBoxLayout(range_group)

        range_layout.addWidget(QLabel("Frame début:"))
        self.in_spin = QSpinBox()
        self.in_spin.valueChanged.connect(self.on_in_changed)
        range_layout.addWidget(self.in_spin)

        range_layout.addWidget(QLabel("Frame fin:"))
        self.out_spin = QSpinBox()
        self.out_spin.valueChanged.connect(self.on_out_changed)
        range_layout.addWidget(self.out_spin)

        self.btn_set_in_here = QPushButton("Début = frame courante")
        self.btn_set_in_here.clicked.connect(self.set_in_to_current)
        range_layout.addWidget(self.btn_set_in_here)

        self.btn_set_out_here = QPushButton("Fin = frame courante")
        self.btn_set_out_here.clicked.connect(self.set_out_to_current)
        range_layout.addWidget(self.btn_set_out_here)

        self.btn_reset_range = QPushButton("Réinitialiser intervalle")
        self.btn_reset_range.clicked.connect(self.reset_range)
        range_layout.addWidget(self.btn_reset_range)

        range_layout.addStretch()
        main_layout.addWidget(range_group)

        # --- Navigation + Lecture ---
        nav_group = QGroupBox("Navigation / Lecture")
        nav_layout = QHBoxLayout(nav_group)

        self.btn_prev = QPushButton("<<")
        self.btn_prev.clicked.connect(self.prev_frame)
        nav_layout.addWidget(self.btn_prev)

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setMinimumHeight(30)
        self.btn_play.setMinimumWidth(90)
        nav_layout.addWidget(self.btn_play)

        self.btn_next = QPushButton(">>")
        self.btn_next.clicked.connect(self.next_frame)
        nav_layout.addWidget(self.btn_next)

        self.chk_loop = QCheckBox("Boucle")
        self.chk_loop.stateChanged.connect(self.on_loop_toggle)
        nav_layout.addWidget(self.chk_loop)

        nav_layout.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(DEFAULT_FPS)
        self.fps_spin.valueChanged.connect(self.on_fps_changed)
        nav_layout.addWidget(self.fps_spin)

        # --- Slider personnalisé avec zones grisées ---
        self.frame_slider = RangeSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        nav_layout.addWidget(self.frame_slider, stretch=1)

        self.frame_spin = QSpinBox()
        self.frame_spin.valueChanged.connect(self.on_spin_changed)
        nav_layout.addWidget(self.frame_spin)

        main_layout.addWidget(nav_group)

        # --- Status bar ---
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.current_index = 0

    # ---------------------- Sélection dossiers ----------------------

    def select_base_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Dossier des images de base")
        if folder:
            self.base_folder = folder
            self.base_frames = find_sequence_files(folder)
            self.base_cache.clear()
            self.status.showMessage(f"{len(self.base_frames)} images trouvées dans {folder}")
            self.try_build_sequence()

    def select_mask_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Dossier des masques")
        if not folder:
            return

        if self.chk_use_working_folder.isChecked():
            # Comportement habituel : on scanne d'abord pour connaître le label,
            # puis on copie dans un dossier de travail avant de charger depuis celui-ci
            labels_info = scan_available_labels(folder)

            if not labels_info:
                QMessageBox.warning(self, "Aucun masque détecté",
                                     "Aucun fichier de masque reconnu (format colorMask_<label>_<tracking>_...) "
                                     "n'a été trouvé dans ce dossier.")
                return

            if len(labels_info) == 1:
                chosen_label = next(iter(labels_info))
            else:
                labels_list = sorted(labels_info.keys())
                chosen_label, ok = QInputDialog.getItem(
                    self, "Choix du label",
                    "Plusieurs labels de masques ont été détectés, lequel charger ?",
                    labels_list, 0, False
                )
                if not ok:
                    return

            tracking = labels_info[chosen_label]["tracking"]
            rest = labels_info[chosen_label]["rest"]

            self.current_label = chosen_label
            self.current_tracking = tracking
            self._setup_working_mask_folder(folder, chosen_label, tracking, rest)
        else:
            # Pas de copie : le dossier sélectionné devient directement le dossier de travail
            self._load_mask_folder(folder)

        self.try_build_sequence()

    def _setup_working_mask_folder(self, original_folder, label, tracking, rest):
        """Copie les masques du label choisi (et bboxes.json) dans un dossier de travail"""
        parent = os.path.dirname(original_folder.rstrip("/\\"))
        base_name = os.path.basename(original_folder.rstrip("/\\"))
        self.working_mask_folder = os.path.join(parent, f"{base_name}_working")

        if os.path.exists(self.working_mask_folder):
            reply = QMessageBox.question(
                self, "Dossier de travail existant",
                f"Le dossier de travail '{self.working_mask_folder}' existe déjà.\n"
                f"Voulez-vous le réutiliser (Oui) ou le régénérer depuis l'original (Non) ?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                shutil.rmtree(self.working_mask_folder)
                os.makedirs(self.working_mask_folder, exist_ok=True)
                self._copy_label_files(original_folder, self.working_mask_folder, label, tracking, rest)
        else:
            os.makedirs(self.working_mask_folder, exist_ok=True)
            self._copy_label_files(original_folder, self.working_mask_folder, label, tracking, rest)

        self.mask_frames = find_label_sequence_files(self.working_mask_folder, label, tracking, rest)
        self.mask_cache.clear()

        bboxes_path = os.path.join(self.working_mask_folder, BBOXES_JSON_NAME)
        self.bboxes_manager = BBoxesManager(bboxes_path, label)

        self.status.showMessage(f"Dossier de travail prêt : {self.working_mask_folder}")

    def _copy_label_files(self, source_folder, dest_folder, label, tracking, rest):
        """Copie uniquement les fichiers du label/tracking choisi, plus bboxes.json"""
        frames = find_label_sequence_files(source_folder, label, tracking, rest)
        for frame_num, filepath in frames.items():
            dest_path = os.path.join(dest_folder, os.path.basename(filepath))
            shutil.copy2(filepath, dest_path)

        copy_bboxes_json(source_folder, dest_folder)

    def try_build_sequence(self):
        if not self.base_frames:
            return

        if self.mask_frames:
            common = sorted(set(self.base_frames.keys()) & set(self.mask_frames.keys()))
            if not common:
                self.status.showMessage("Aucune frame commune entre les deux séquences !")
                return
            self.frame_list = common
        else:
            self.frame_list = sorted(self.base_frames.keys())

        self.frame_slider.setRange(0, len(self.frame_list) - 1)
        self.frame_spin.setRange(self.frame_list[0], self.frame_list[-1])

        # Initialise l'intervalle de lecture sur toute la séquence
        self._init_range_controls()

        self.current_index = 0
        self.frame_slider.setValue(0)
        self.load_and_display()

    # ---------------------- Intervalle de lecture ----------------------

    def _init_range_controls(self):
        """Initialise les spinbox in/out avec les bornes de la séquence complète"""
        if not self.frame_list:
            return

        first_frame = self.frame_list[0]
        last_frame = self.frame_list[-1]

        self.in_index = 0
        self.out_index = len(self.frame_list) - 1

        for spin in (self.in_spin, self.out_spin):
            spin.blockSignals(True)
            spin.setRange(first_frame, last_frame)
            spin.blockSignals(False)

        self.in_spin.blockSignals(True)
        self.in_spin.setValue(first_frame)
        self.in_spin.blockSignals(False)

        self.out_spin.blockSignals(True)
        self.out_spin.setValue(last_frame)
        self.out_spin.blockSignals(False)

        self._update_range_slider_display()

    def _update_range_slider_display(self):
        """Met à jour l'affichage visuel du slider selon in_index/out_index"""
        self.frame_slider.set_active_range(self.in_index, self.out_index)

    def _nearest_index_for_frame(self, frame_num):
        """Trouve l'index dans frame_list le plus proche (>=) du numéro de frame donné"""
        for i, f in enumerate(self.frame_list):
            if f >= frame_num:
                return i
        return len(self.frame_list) - 1

    def on_in_changed(self, value):
        if not self.frame_list:
            return
        idx = self._nearest_index_for_frame(value)
        if idx > self.out_index:
            idx = self.out_index
            self.in_spin.blockSignals(True)
            self.in_spin.setValue(self.frame_list[idx])
            self.in_spin.blockSignals(False)
        self.in_index = idx
        self._update_range_slider_display()
        self.status.showMessage(f"Frame début fixée à {self.frame_list[self.in_index]}")

    def on_out_changed(self, value):
        if not self.frame_list:
            return
        idx = self._nearest_index_for_frame(value)
        if idx < self.in_index:
            idx = self.in_index
            self.out_spin.blockSignals(True)
            self.out_spin.setValue(self.frame_list[idx])
            self.out_spin.blockSignals(False)
        self.out_index = idx
        self._update_range_slider_display()
        self.status.showMessage(f"Frame fin fixée à {self.frame_list[self.out_index]}")

    def set_in_to_current(self):
        if not self.frame_list:
            return
        self.in_spin.setValue(self.frame_list[self.current_index])

    def set_out_to_current(self):
        if not self.frame_list:
            return
        self.out_spin.setValue(self.frame_list[self.current_index])

    def reset_range(self):
        self._init_range_controls()
        self.status.showMessage("Intervalle de lecture réinitialisé sur toute la séquence.")

    # ---------------------- Cache management ----------------------

    def get_base_raw(self, frame_num):
        cached = self.base_cache.get(frame_num)
        if cached is not None:
            return cached

        path = self.base_frames.get(frame_num)
        if not path:
            return None

        data = load_exr_raw(path)
        self.base_cache.put(frame_num, data)
        return data

    def get_mask_raw(self, frame_num):
        cached = self.mask_cache.get(frame_num)
        if cached is not None:
            return cached

        path = self.mask_frames.get(frame_num)
        if not path:
            return None

        data = load_exr_raw(path)
        self.mask_cache.put(frame_num, data)
        return data

    def update_cache_status(self):
        self.cache_status_label.setText(
            f"Cache: {len(self.base_cache)} images / {len(self.mask_cache)} masques"
        )

    def clear_caches(self):
        self.base_cache.clear()
        self.mask_cache.clear()
        self.update_cache_status()
        self.status.showMessage("Cache vidé.")

    def preload_sequence(self):
        if not self.frame_list:
            return

        total = len(self.frame_list)
        self.btn_preload.setEnabled(False)

        for i, frame_num in enumerate(self.frame_list):
            try:
                self.get_base_raw(frame_num)
                if self.mask_frames:
                    self.get_mask_raw(frame_num)
            except Exception as e:
                self.status.showMessage(f"Erreur préchargement frame {frame_num}: {e}")

            self.status.showMessage(f"Préchargement... {i + 1}/{total}")
            self.update_cache_status()
            QApplication.processEvents()

        self.btn_preload.setEnabled(True)
        self.status.showMessage(f"Préchargement terminé ({total} frames).")
        self.update_cache_status()

    # ---------------------- Navigation ----------------------

    def prev_frame(self):
        if self.current_index > 0:
            self.current_index -= 1
            self._sync_controls()
            self.load_and_display()

    def next_frame(self):
        """Avance d'une frame, en respectant l'intervalle in/out pendant la lecture"""
        if not self.frame_list:
            return

        if self.current_index < self.out_index:
            self.current_index += 1
            self._sync_controls()
            self.load_and_display()
        elif self.current_index >= self.out_index:
            if self.loop_playback:
                self.current_index = self.in_index
                self._sync_controls()
                self.load_and_display()
            elif self.is_playing:
                self.stop_play()

    def on_slider_changed(self, value):
        if 0 <= value < len(self.frame_list):
            self.current_index = value
            self._sync_controls(update_slider=False)
            self.load_and_display()

    def on_spin_changed(self, value):
        if value in self.frame_list:
            self.current_index = self.frame_list.index(value)
            self._sync_controls(update_spin=False)
            self.load_and_display()

    def _sync_controls(self, update_slider=True, update_spin=True):
        if not self.frame_list:
            return
        frame_num = self.frame_list[self.current_index]
        if update_slider:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_index)
            self.frame_slider.blockSignals(False)
        if update_spin:
            self.frame_spin.blockSignals(True)
            self.frame_spin.setValue(frame_num)
            self.frame_spin.blockSignals(False)

    # ---------------------- Lecture automatique ----------------------

    def toggle_play(self):
        if self.is_playing:
            self.stop_play()
        else:
            self.start_play()

    def start_play(self):
        if not self.frame_list:
            return

        if self.current_index < self.in_index or self.current_index > self.out_index:
            self.current_index = self.in_index
            self._sync_controls()
            self.load_and_display()

        self.is_playing = True
        self.btn_play.setText("⏸ Pause")
        interval_ms = int(1000 / self.fps)
        self.play_timer.start(interval_ms)

    def stop_play(self):
        self.is_playing = False
        self.btn_play.setText("▶ Play")
        self.play_timer.stop()

    def _on_play_tick(self):
        self.next_frame()

    def on_loop_toggle(self, state):
        self.loop_playback = bool(state)

    def on_fps_changed(self, value):
        self.fps = value
        if self.is_playing:
            self.play_timer.setInterval(int(1000 / self.fps))

    # ---------------------- Overlay controls ----------------------

    def on_overlay_toggle(self, state):
        self.show_overlay = bool(state)
        self.load_and_display()

    # ---------------------- Sélection de couleur ----------------------

    def on_viewer_clicked(self, norm_x, norm_y):
        if not self.frame_list or not self.mask_frames:
            self.status.showMessage("Aucun masque chargé, impossible de sélectionner une couleur.")
            return

        frame_num = self.frame_list[self.current_index]

        try:
            mask_data = self.get_mask_raw(frame_num)
            if mask_data is None:
                self.status.showMessage("Pas de masque pour cette frame.")
                return

            mask_raw, mw, mh, mc = mask_data
            height, width = mask_raw.shape[0], mask_raw.shape[1]

            px = int(np.clip(norm_x * width, 0, width - 1))
            py = int(np.clip(norm_y * height, 0, height - 1))

            color = tuple(mask_raw[py, px, :3].tolist())
            self.selected_color = color
            self.color_pick_mode = True

            # --- Résolution de l'obj_id correspondant, via la palette déterministe ---
            self.selected_obj_id = obj_id_from_color(color)

            obj_id_str = f", obj_id={self.selected_obj_id}" if self.selected_obj_id is not None else " (obj_id inconnu)"
            self.color_status_label.setText(
                f"Couleur sélectionnée: R={color[0]:.3f} G={color[1]:.3f} B={color[2]:.3f}{obj_id_str}"
            )
            self.status.showMessage(f"Masque filtré sur la couleur ({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}){obj_id_str}")

            self.load_and_display()

        except Exception as e:
            self.status.showMessage(f"Erreur lors du picking de couleur: {e}")

    def on_viewer_left_click(self, norm_x, norm_y):
        if self.sam_mode_active:
            self.sam_handle_click(norm_x, norm_y, positive=True)
        else:
            self.on_viewer_clicked(norm_x, norm_y)

    def on_viewer_right_click(self, norm_x, norm_y):
        if self.sam_mode_active:
            self.sam_handle_click(norm_x, norm_y, positive=False)

    def reset_color_selection(self):
        self.color_pick_mode = False
        self.selected_color = None
        self.selected_obj_id = None
        self.color_status_label.setText("Aucune sélection de couleur")
        self.status.showMessage("Sélection de couleur réinitialisée.")
        self.load_and_display()

    def compute_color_mask_alpha(self, mask_raw, presence_alpha):
        rgb = mask_raw[:, :, :3]

        target = np.array(self.selected_color, dtype=np.float32)
        diff = rgb - target[np.newaxis, np.newaxis, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))

        match = (dist <= FIXED_COLOR_TOLERANCE).astype(np.float32)

        return match * presence_alpha

    # ---------------------- Chargement / affichage ----------------------

    def load_and_display(self):
        if not self.frame_list:
            return

        frame_num = self.frame_list[self.current_index]

        try:
            base_data = self.get_base_raw(frame_num)
            if base_data is None:
                self.status.showMessage(f"Image introuvable pour la frame {frame_num}")
                return
            base_raw, w, h, bc = base_data
            base_rgb, _ = raw_to_rgb8(base_raw, bc, FIXED_EXPOSURE, FIXED_GAMMA)
        except Exception as e:
            self.status.showMessage(f"Erreur chargement image: {e}")
            return

        result = base_rgb
        sam_idx = None

        if self.sam_mode_active and self._is_current_frame_in_sam_interval():
            sam_idx = self._current_sam_idx()
            mask_u8 = self.sam_propagated_masks.get(sam_idx)
            if mask_u8 is not None:
                result = self._composite_sam_preview(base_rgb, mask_u8)

        elif self.show_overlay and self.mask_frames:
            try:
                mask_data = self.get_mask_raw(frame_num)
                if mask_data is not None:
                    mask_raw, mw, mh, mc = mask_data
                    if (mw, mh) == (w, h):
                        mask_rgb, _ = raw_to_rgb8(mask_raw, mc, 0.0, 1.0)
                        presence_alpha = compute_presence_alpha(mask_raw)
                        if self.color_pick_mode and self.selected_color is not None:
                            effective_alpha = self.compute_color_mask_alpha(mask_raw, presence_alpha)
                        else:
                            effective_alpha = presence_alpha
                        result = self.composite_overlay(base_rgb, mask_rgb, effective_alpha)
                    else:
                        self.status.showMessage("Résolution du masque différente de l'image de base !")
            except Exception as e:
                self.status.showMessage(f"Erreur chargement masque: {e}")

        # --- Affichage : image d'abord, puis points par-dessus si en mode SAM2 ---
        self.viewer.set_image(result)

        if sam_idx is not None:
            self._update_viewer_points(sam_idx, w, h)
        else:
            self.viewer.clear_points()

        self.update_cache_status()

        mode_str = ""
        if self.sam_mode_active:
            mode_str = " [MODE SAM2 ACTIF]"
        elif self.color_pick_mode:
            mode_str = " [FILTRE COULEUR ACTIF]"

        range_str = ""
        if self.frame_list and (self.in_index > 0 or self.out_index < len(self.frame_list) - 1):
            range_str = f" [Intervalle: {self.frame_list[self.in_index]}-{self.frame_list[self.out_index]}]"

        base_path = self.base_frames.get(frame_num, "")
        self.status.showMessage(
            f"Frame {frame_num} ({self.current_index + 1}/{len(self.frame_list)}) - "
            f"{os.path.basename(base_path)}{mode_str}{range_str}"
        )

    def _update_viewer_points(self, sam_idx, img_w, img_h):
        """Prépare et envoie au viewer les points (normalisés) de la frame SAM2 courante"""
        points_for_frame = [
            {
                "x_norm": p["x"] / img_w,
                "y_norm": p["y"] / img_h,
                "positive": p["positive"],
            }
            for p in self.sam_points
            if p["sam_idx"] == sam_idx
        ]
        self.viewer.set_points(points_for_frame)

    def _composite_sam_preview(self, base_rgb, mask_u8):
        """
        Overlay du masque SAM2 en cours de calcul, avec la couleur du masque
        actuellement sélectionné (self.selected_color), pour rester cohérent
        visuellement avec le rendu du masque final une fois validé.
        """
        base_f = base_rgb.astype(np.float32)
        alpha = (mask_u8.astype(np.float32) / 255.0) * FIXED_OVERLAY_OPACITY
        alpha = alpha[:, :, np.newaxis]

        preview_color = self._selected_color_to_display_rgb()
        preview_rgb = np.broadcast_to(preview_color, base_f.shape)

        blended = base_f * (1.0 - alpha) + preview_rgb * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _selected_color_to_display_rgb(self):
        """
        Convertit self.selected_color (valeurs linéaires EXR, typiquement 0.0-1.0+)
        en RGB 8-bit d'affichage (gamma-corrigé), cohérent avec le tonemapping
        utilisé pour afficher les masques normalement (raw_to_rgb8 avec gamma=FIXED_GAMMA).
        """
        if self.selected_color is None:
            # Fallback de sécurité (ne devrait pas arriver puisque le mode SAM2
            # nécessite une sélection de couleur au préalable)
            return np.array([255.0, 60.0, 60.0], dtype=np.float32)

        color = np.array(self.selected_color, dtype=np.float32)
        color = np.clip(color, 0.0, None)
        color = np.power(color, 1.0 / FIXED_GAMMA)
        color = np.clip(color * 255.0, 0, 255)
        return color.astype(np.float32)

    def composite_overlay(self, base_rgb, mask_rgb, presence_alpha):
        base_f = base_rgb.astype(np.float32)
        mask_f = mask_rgb.astype(np.float32)

        alpha = presence_alpha * FIXED_OVERLAY_OPACITY
        alpha = alpha[:, :, np.newaxis]

        blended = base_f * (1.0 - alpha) + mask_f * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)

    # ---------------------- SAM2 - Re-segmentation ----------------------

    def start_sam_session(self):
        """Entre en mode SAM2 : extraction des frames de l'intervalle et init du predictor"""
        if self.selected_color is None:
            QMessageBox.warning(self, "Aucune sélection",
                                 "Veuillez d'abord sélectionner un masque par couleur (clic sur l'image).")
            return

        if not self.frame_list:
            return

        if self.working_mask_folder is None:
            reply = QMessageBox.warning(
                self, "Pas de dossier de travail",
                "Aucun dossier de travail n'a été créé (case décochée).\n"
                "La validation SAM2 modifiera directement les EXR originaux du dossier de masques.\n\n"
                "Voulez-vous continuer ?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
            session_base_folder = self.mask_folder
        else:
            session_base_folder = self.working_mask_folder

        interval_frame_numbers = self.frame_list[self.in_index:self.out_index + 1]
        if len(interval_frame_numbers) < 1:
            QMessageBox.warning(self, "Intervalle invalide", "L'intervalle sélectionné est vide.")
            return

        self.status.showMessage("Extraction des frames pour SAM2, veuillez patienter...")
        QApplication.processEvents()

        work_root = os.path.join(session_base_folder, "_sam2_session")
        self.sam_session = SamSession(work_root=work_root)

        try:
            self.sam_session.prepare_frames(
                interval_frame_numbers,
                get_base_raw_fn=self.get_base_raw,
                tonemap_fn=lambda raw, nc: raw_to_rgb8(raw, nc, FIXED_EXPOSURE, FIXED_GAMMA)[0],
            )
            self.sam_session.load_predictor()
            self.sam_session.init_inference_state()
        except Exception as e:
            QMessageBox.critical(self, "Erreur SAM2", f"Impossible d'initialiser la session SAM2:\n{e}")
            self.sam_session = None
            self.status.showMessage("Erreur lors de l'initialisation de SAM2.")
            return

        self.sam_mode_active = True
        self.sam_points = []
        self.sam_current_mask = None
        self.sam_propagated_masks = {}

        # La frame de référence initiale = frame courante si elle est dans l'intervalle, sinon in_index
        if self.in_index <= self.current_index <= self.out_index:
            self.sam_reference_sam_idx = self.current_index - self.in_index
        else:
            self.current_index = self.in_index
            self.sam_reference_sam_idx = 0
            self._sync_controls()

        self.sam_controls_widget.setEnabled(True)
        self.btn_sam_start.setEnabled(False)
        self._set_normal_controls_enabled(False)

        self.sam_status_label.setText(
            f"Mode SAM2 actif — cliquez sur l'image (gauche=positif, droit=négatif) "
            f"pour segmenter la frame de référence."
        )
        self.load_and_display()

    def _set_normal_controls_enabled(self, enabled):
        """Désactive/active les contrôles normaux pendant le mode SAM2 pour éviter les conflits"""
        self.chk_overlay.setEnabled(enabled)
        self.btn_reset_selection.setEnabled(enabled)
        self.in_spin.setEnabled(enabled)
        self.out_spin.setEnabled(enabled)
        self.btn_set_in_here.setEnabled(enabled)
        self.btn_set_out_here.setEnabled(enabled)
        self.btn_reset_range.setEnabled(enabled)
        if not self.launch_with_params:
            self.btn_base.setEnabled(enabled)
            self.btn_mask.setEnabled(enabled)


    def _current_sam_idx(self):
        """Convertit l'index courant du viewer en index local SAM2 (dans l'intervalle)"""
        return self.current_index - self.in_index

    def _is_current_frame_in_sam_interval(self):
        return self.in_index <= self.current_index <= self.out_index

    def sam_handle_click(self, norm_x, norm_y, positive):
        """Appelé quand l'utilisateur clique sur le viewer en mode SAM2"""
        if not self.sam_mode_active or self.sam_session is None:
            return

        if not self._is_current_frame_in_sam_interval():
            self.status.showMessage("Naviguez dans l'intervalle sélectionné pour ajouter des points.")
            return

        sam_idx = self._current_sam_idx()
        self.sam_reference_sam_idx = sam_idx  # la frame cliquée devient la référence

        w = self.sam_session.exr_width
        h = self.sam_session.exr_height
        px = float(np.clip(norm_x * w, 0, w - 1))
        py = float(np.clip(norm_y * h, 0, h - 1))

        self.sam_points.append({
            "sam_idx": sam_idx,
            "x": px,
            "y": py,
            "positive": positive,
        })

        points_this_frame = [(p["x"], p["y"]) for p in self.sam_points if p["sam_idx"] == sam_idx]
        labels_this_frame = [1 if p["positive"] else 0 for p in self.sam_points if p["sam_idx"] == sam_idx]

        try:
            mask = self.sam_session.add_points(sam_idx, points_this_frame, labels_this_frame)
            self.sam_current_mask = mask
            self.sam_propagated_masks[sam_idx] = mask  # la frame de référence a aussi son masque à jour
        except Exception as e:
            self.status.showMessage(f"Erreur segmentation SAM2: {e}")
            return

        label_str = "positif" if positive else "négatif"
        self.status.showMessage(f"Point {label_str} ajouté sur frame locale {sam_idx}. Masque mis à jour.")
        self.load_and_display()

    def sam_clear_points_current_frame(self):
        """Efface les points de la frame de référence courante et réinitialise son masque"""
        if not self.sam_mode_active or self.sam_session is None:
            return

        sam_idx = self._current_sam_idx()
        self.sam_points = [p for p in self.sam_points if p["sam_idx"] != sam_idx]
        self.sam_propagated_masks.pop(sam_idx, None)
        self.sam_current_mask = None

        cleared = self.sam_session.clear_frame_points(sam_idx)

        if cleared:
            self.status.showMessage(f"Points et tracking effacés pour la frame locale {sam_idx}.")
        else:
            self.status.showMessage(f"Aucun point à effacer pour la frame locale {sam_idx}.")

        self.load_and_display()

    def _sam_progress_callback(self, sam_idx, total):
        self.status.showMessage(f"Propagation SAM2... frame locale {sam_idx}/{total - 1}")
        QApplication.processEvents()

    def _navigate_to_sam_idx(self, sam_idx):
        """Déplace le viewer sur la frame correspondant à cet index local SAM2"""
        self.current_index = self.in_index + sam_idx
        self._sync_controls()

    def sam_propagate_forward(self):
        if not self.sam_mode_active or self.sam_session is None:
            return
        self.status.showMessage("Propagation en avant...")
        QApplication.processEvents()
        results = self.sam_session.propagate(
            start_sam_idx=self.sam_reference_sam_idx, reverse=False,
            progress_callback=self._sam_progress_callback
        )
        self.sam_propagated_masks.update(results)

        if results:
            last_idx = max(results.keys())
            self.sam_reference_sam_idx = last_idx
            self._navigate_to_sam_idx(last_idx)
        
        self.status.showMessage(f"Propagation avant terminée ({len(results)} frames).")
        self.load_and_display()

    def sam_propagate_backward(self):
        if not self.sam_mode_active or self.sam_session is None:
            return
        self.status.showMessage("Propagation en arrière...")
        QApplication.processEvents()
        results = self.sam_session.propagate(
            start_sam_idx=self.sam_reference_sam_idx, reverse=True,
            progress_callback=self._sam_progress_callback
        )
        self.sam_propagated_masks.update(results)

        if results:
            last_idx = min(results.keys())
            self.sam_reference_sam_idx = last_idx
            self._navigate_to_sam_idx(last_idx)
        
        self.status.showMessage(f"Propagation arrière terminée ({len(results)} frames).")
        self.load_and_display()

    def sam_step_forward(self):
        """Propage une seule frame vers l'avant depuis la frame courante, puis navigue dessus"""
        if not self.sam_mode_active or self.sam_session is None:
            return

        if not self._is_current_frame_in_sam_interval():
            self.status.showMessage("Naviguez dans l'intervalle SAM2 avant de propager.")
            return

        current_sam_idx = self._current_sam_idx()
        new_sam_idx, mask = self.sam_session.propagate_one_frame_forward(current_sam_idx)

        if mask is None and new_sam_idx == current_sam_idx:
            self.status.showMessage("Déjà à la dernière frame de l'intervalle.")
            return

        self.sam_propagated_masks[new_sam_idx] = mask
        self.sam_reference_sam_idx = new_sam_idx
        self._navigate_to_sam_idx(new_sam_idx)

        self.status.showMessage(f"Propagé 1 frame en avant (frame locale {new_sam_idx}).")
        self.load_and_display()

    def sam_step_backward(self):
        """Propage une seule frame vers l'arrière depuis la frame courante, puis navigue dessus"""
        if not self.sam_mode_active or self.sam_session is None:
            return

        if not self._is_current_frame_in_sam_interval():
            self.status.showMessage("Naviguez dans l'intervalle SAM2 avant de propager.")
            return

        current_sam_idx = self._current_sam_idx()
        new_sam_idx, mask = self.sam_session.propagate_one_frame_backward(current_sam_idx)

        if mask is None and new_sam_idx == current_sam_idx:
            self.status.showMessage("Déjà à la première frame de l'intervalle.")
            return

        self.sam_propagated_masks[new_sam_idx] = mask
        self.sam_reference_sam_idx = new_sam_idx
        self._navigate_to_sam_idx(new_sam_idx)

        self.status.showMessage(f"Propagé 1 frame en arrière (frame locale {new_sam_idx}).")
        self.load_and_display()

    def sam_step10_forward(self):
        """Propage jusqu'à 10 frames vers l'avant depuis la frame courante, navigue sur la dernière traitée"""
        if not self.sam_mode_active or self.sam_session is None:
            return

        if not self._is_current_frame_in_sam_interval():
            self.status.showMessage("Naviguez dans l'intervalle SAM2 avant de propager.")
            return

        current_sam_idx = self._current_sam_idx()
        self.status.showMessage("Propagation de 10 frames vers l'avant...")
        QApplication.processEvents()

        last_sam_idx, results = self.sam_session.propagate_n_frames_forward(current_sam_idx, n=10)

        if not results:
            self.status.showMessage("Déjà à la dernière frame de l'intervalle.")
            return

        self.sam_propagated_masks.update(results)
        self.sam_reference_sam_idx = last_sam_idx
        self._navigate_to_sam_idx(last_sam_idx)

        self.status.showMessage(f"Propagé {len(results)} frame(s) en avant (arrêt frame locale {last_sam_idx}).")
        self.load_and_display()

    def sam_step10_backward(self):
        """Propage jusqu'à 10 frames vers l'arrière depuis la frame courante, navigue sur la dernière traitée"""
        if not self.sam_mode_active or self.sam_session is None:
            return

        if not self._is_current_frame_in_sam_interval():
            self.status.showMessage("Naviguez dans l'intervalle SAM2 avant de propager.")
            return

        current_sam_idx = self._current_sam_idx()
        self.status.showMessage("Propagation de 10 frames vers l'arrière...")
        QApplication.processEvents()

        last_sam_idx, results = self.sam_session.propagate_n_frames_backward(current_sam_idx, n=10)

        if not results:
            self.status.showMessage("Déjà à la première frame de l'intervalle.")
            return

        self.sam_propagated_masks.update(results)
        self.sam_reference_sam_idx = last_sam_idx
        self._navigate_to_sam_idx(last_sam_idx)

        self.status.showMessage(f"Propagé {len(results)} frame(s) en arrière (arrêt frame locale {last_sam_idx}).")
        self.load_and_display()

    def sam_propagate_all(self):
        """Propage d'abord en avant, puis en arrière, depuis la frame de référence"""
        ref_index = self.current_index
        self.sam_propagate_forward()
        self.sam_reference_sam_idx = ref_index
        self.sam_propagate_backward()

    def sam_validate_and_exit(self):
        """Écrit les masques calculés dans les EXR du dossier de travail, puis quitte le mode SAM2"""
        if not self.sam_mode_active or self.sam_session is None:
            return

        if not self.sam_propagated_masks:
            reply = QMessageBox.question(
                self, "Aucun masque calculé",
                "Aucune frame n'a de masque calculé. Voulez-vous vraiment valider (l'intervalle restera vide) ?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        self.status.showMessage("Écriture des masques dans les EXR de travail...")
        QApplication.processEvents()

        total = len(self.sam_propagated_masks)
        for i, (sam_idx, mask_u8) in enumerate(self.sam_propagated_masks.items()):
            frame_num = self.sam_session.sam_idx_to_frame_num.get(sam_idx)
            if frame_num is None:
                continue
            try:
                self._write_mask_into_exr(frame_num, mask_u8)

                # --- Mise à jour du fichier bboxes.json ---
                if self.bboxes_manager is not None:
                    self.bboxes_manager.update_from_mask(frame_num, self.selected_obj_id, mask_u8)
                    
            except Exception as e:
                self.status.showMessage(f"Erreur écriture masque frame {frame_num}: {e}")
                continue

            if i % 5 == 0:
                self.status.showMessage(f"Écriture... {i + 1}/{total}")
                QApplication.processEvents()

        if self.bboxes_manager is not None:
            self.bboxes_manager.save()

        # Invalide le cache pour les frames modifiées (pour forcer le rechargement depuis disque)
        for sam_idx in self.sam_propagated_masks.keys():
            frame_num = self.sam_session.sam_idx_to_frame_num.get(sam_idx)
            if frame_num is not None:
                self.mask_cache._cache.pop(frame_num, None)

        self.status.showMessage(f"{total} masques écrits avec succès dans le dossier de travail.")
        self._exit_sam_mode()

    def sam_cancel_and_exit(self):
        """Quitte le mode SAM2 sans écrire aucun changement"""
        self.status.showMessage("Session SAM2 annulée, aucun changement appliqué.")
        self._exit_sam_mode()

    def _exit_sam_mode(self):
        if self.sam_session is not None:
            self.sam_session.close()
            self.sam_session = None

        self.sam_mode_active = False
        self.sam_points = []
        self.sam_current_mask = None
        self.sam_propagated_masks = {}

        self.sam_controls_widget.setEnabled(False)
        self.btn_sam_start.setEnabled(True)
        self._set_normal_controls_enabled(True)
        self.sam_status_label.setText("")

        self.viewer.clear_points()
        self.reset_color_selection()
        self.reset_range()
        self.load_and_display()

    def _write_mask_into_exr(self, frame_num, mask_u8):
        """
        Remplace, dans l'EXR du dossier de travail pour cette frame, tous les pixels
        de la couleur précédemment sélectionnée par la nouvelle zone masquée (mask_u8),
        en conservant les autres couleurs/masques déjà présents.
        """
        mask_path = self.mask_frames.get(frame_num)
        if not mask_path:
            raise IOError(f"Pas de fichier masque pour la frame {frame_num}")

        mask_raw, w, h, nc = load_exr_raw(mask_path)

        if mask_u8.shape != (h, w):
            # Sécurité si jamais SAM2 a retourné une résolution différente (ne devrait pas arriver)
            mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)

        target = np.array(self.selected_color, dtype=np.float32)

        # 1. Efface l'ancienne zone de cette couleur (remise à zéro / noir)
        rgb = mask_raw[:, :, :3]
        diff = rgb - target[np.newaxis, np.newaxis, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        old_zone = dist <= FIXED_COLOR_TOLERANCE
        mask_raw[old_zone, :3] = 0.0

        # 2. Applique la couleur sélectionnée sur la nouvelle zone (mask_u8 > 127)
        new_zone = mask_u8 > 127
        mask_raw[new_zone, 0] = target[0]
        mask_raw[new_zone, 1] = target[1]
        mask_raw[new_zone, 2] = target[2]

        # 3. Sauvegarde l'EXR modifié
        self._save_exr(mask_path, mask_raw, nc)

    def _save_exr(self, filepath, arr, nchannels):
        """Sauvegarde un array float (H,W,C) en EXR via OpenImageIO"""
        h, w = arr.shape[0], arr.shape[1]
        out = oiio.ImageOutput.create(filepath)
        if not out:
            raise IOError(f"Impossible de créer le fichier de sortie {filepath}: {oiio.geterror()}")

        spec = oiio.ImageSpec(w, h, nchannels, oiio.FLOAT)
        out.open(filepath, spec)
        out.write_image(arr)
        out.close()

    # ---------------------- Raccourcis clavier ----------------------

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Left:
            self.prev_frame()
        elif event.key() == Qt.Key_Right:
            self.next_frame()
        elif event.key() == Qt.Key_Space:
            self.toggle_play()
        elif event.key() == Qt.Key_O:
            self.chk_overlay.toggle()
        elif event.key() == Qt.Key_Escape:
            self.reset_color_selection()
        elif event.key() == Qt.Key_I:
            self.set_in_to_current()
        elif event.key() == Qt.Key_U:
            self.set_out_to_current()
        else:
            super().keyPressEvent(event)


def main():

    app = QApplication(sys.argv)

    base_paths = sys.argv[1]
    mask_folder = sys.argv[2]
    bboxes_json_path = sys.argv[3]

    window = MainWindow(
        base_frame_paths=base_paths,
        mask_folder=mask_folder,
        bboxes_json_path=bboxes_json_path,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()