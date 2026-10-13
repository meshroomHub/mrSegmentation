#!/usr/bin/env python3
"""
Gestion des labels de masques, du fichier bboxes.json associé,
et de la correspondance couleur <-> obj_id via paletteGenerator.
"""

import os
import re
import json
import shutil
import numpy as np
from color_palette import paletteGenerator


# Pattern des fichiers de masque : colorMask_{label}_{tracking}_{reste}.{frame}.exr
MASK_FILENAME_RE = re.compile(
    r"^colorMask_(?P<label>[^_]+)_(?P<tracking>merged|forward|backward)_(?P<rest>.+)\.(?P<frame>\d+)\.exr$",
    re.IGNORECASE
)

BBOXES_JSON_NAME = "bboxes.json"

# Palette partagée, cohérente avec le pipeline existant (seed fixe = déterministe)
_palette = paletteGenerator(seed=42)


def scan_available_labels(folder):
    """
    Scanne un dossier de masques et retourne un dict {label: {'tracking': str, 'rest': str}}
    en ne gardant qu'un seul type de tracking détecté par label (celui qui correspond réellement
    aux fichiers présents dans CE dossier, puisque chaque type de tracking vit dans un dossier séparé).
    """
    labels_info = {}

    for filename in os.listdir(folder):
        if not filename.lower().endswith(".exr"):
            continue
        m = MASK_FILENAME_RE.match(filename)
        if not m:
            continue

        label = m.group("label")
        tracking = m.group("tracking").lower()
        rest = m.group("rest")

        if label not in labels_info:
            labels_info[label] = {"tracking": tracking, "rest": rest}

    return labels_info


def find_label_sequence_files(folder, label, tracking, rest):
    """
    Trouve les fichiers EXR correspondant à un label/tracking/rest donné dans un dossier,
    retourne un dict {frame_num: filepath}, comme find_sequence_files() mais filtré sur le label.
    """
    pattern_re = re.compile(
        rf"^colorMask_{re.escape(label)}_{re.escape(tracking)}_{re.escape(rest)}\.(\d+)\.exr$",
        re.IGNORECASE
    )

    frames = {}
    for filename in os.listdir(folder):
        m = pattern_re.match(filename)
        if m:
            frame_num = int(m.group(1))
            frames[frame_num] = os.path.join(folder, filename)

    return frames


def color_float_to_uint8(color_float):
    """Convertit une couleur linéaire EXR (float, typiquement 0.0-1.0+) en tuple RGB uint8 0-255"""
    arr = np.clip(np.array(color_float, dtype=np.float32), 0.0, 1.0)
    rgb_u8 = tuple(int(round(c * 255.0)) for c in arr)
    return rgb_u8


def color_uint8_to_float(color_u8):
    """Convertit un tuple RGB uint8 0-255 en couleur linéaire EXR float 0.0-1.0"""
    return tuple(c / 255.0 for c in color_u8)


def obj_id_from_color(color_float):
    """
    Retrouve l'obj_id correspondant à une couleur (float linéaire EXR) sélectionnée dans l'image,
    via la palette déterministe partagée. Retourne None si aucune correspondance exacte.
    """
    color_u8 = color_float_to_uint8(color_float)
    idx = _palette.index(color_u8)
    return idx


def color_from_obj_id(obj_id):
    """Retourne la couleur (float linéaire EXR 0.0-1.0) correspondant à un obj_id donné"""
    color_u8 = _palette.at(obj_id)
    if color_u8 is None:
        return None
    return color_uint8_to_float(color_u8)


class BBoxesManager:
    """
    Gère la lecture/écriture du fichier bboxes.json pour un label donné.
    """

    def __init__(self, json_path, label):
        self.json_path = json_path
        self.label = label
        self.data = {}
        self._load()

    def _load(self):
        if os.path.exists(self.json_path):
            with open(self.json_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {}

    def save(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)

    def _get_tracking_key(self):
        """Retourne 'merged' si présent pour ce label, sinon 'forward'"""
        label_data = self.data.get(self.label, {})
        if "merged" in label_data:
            return "merged"
        return "forward"

    def get_bbox(self, frame_num, obj_id):
        """Retourne [x1,y1,x2,y2] ou None si absent"""
        tracking_key = self._get_tracking_key()
        label_data = self.data.get(self.label, {})
        tracking_data = label_data.get(tracking_key, {})
        frame_data = tracking_data.get(str(frame_num), {})
        return frame_data.get(str(obj_id))

    def set_bbox(self, frame_num, obj_id, bbox):
        """
        Définit/remplace la bbox [x1,y1,x2,y2] pour une frame/obj_id donnés,
        dans la clé de tracking active (merged si présente, sinon forward).
        Crée les structures intermédiaires si nécessaire.
        """
        tracking_key = self._get_tracking_key()

        if self.label not in self.data:
            self.data[self.label] = {}
        if tracking_key not in self.data[self.label]:
            self.data[self.label][tracking_key] = {}

        frame_key = str(frame_num)
        if frame_key not in self.data[self.label][tracking_key]:
            self.data[self.label][tracking_key][frame_key] = {}

        self.data[self.label][tracking_key][frame_key][str(obj_id)] = list(bbox)

    def remove_bbox(self, frame_num, obj_id):
        """Supprime l'entrée bbox pour une frame/obj_id donnés, si elle existe"""
        tracking_key = self._get_tracking_key()
        label_data = self.data.get(self.label, {})
        tracking_data = label_data.get(tracking_key, {})
        frame_key = str(frame_num)

        frame_data = tracking_data.get(frame_key)
        if frame_data is None:
            return

        frame_data.pop(str(obj_id), None)

        # Nettoyage : si la frame n'a plus aucun obj_id, on retire l'entrée frame vide
        if not frame_data:
            tracking_data.pop(frame_key, None)

    def update_from_mask(self, frame_num, obj_id, mask_u8):
        """
        Met à jour (ou supprime) la bbox pour frame_num/obj_id à partir d'un masque binaire uint8.
        Si le masque est vide (aucun pixel actif), l'entrée bbox est supprimée.
        Sinon elle est mise à jour avec le rectangle englobant des pixels actifs.
        """
        ys, xs = np.where(mask_u8 > 127)
        if len(xs) == 0 or len(ys) == 0:
            self.remove_bbox(frame_num, obj_id)
            return

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        self.set_bbox(frame_num, obj_id, [x1, y1, x2, y2])


def copy_bboxes_json(source_folder, dest_folder):
    """Copie bboxes.json du dossier source vers le dossier de travail, si présent"""
    src = os.path.join(source_folder, BBOXES_JSON_NAME)
    dst = os.path.join(dest_folder, BBOXES_JSON_NAME)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        return dst
    return None
