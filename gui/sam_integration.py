#!/usr/bin/env python3
"""
Intégration SAM2 pour la re-segmentation de masques dans un intervalle.
Adapté de l'architecture SamManager de Sammie-Roto-2.
Version : passage des frames directement en mémoire (np.ndarray), sans extraction JPEG.
"""

import os
import shutil
import numpy as np
import cv2
import torch
from sam2.build_sam import build_sam2_video_predictor


# ---------------------- Configuration modèle ----------------------
SAM2_CHECKPOINT = "/s/apps/users/multiview/sammieRoto2/2.4.1/repo/checkpoints/sam2.1_hiera_base_plus.pt"
SAM2_MODEL_CFG = "./sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"

FIXED_OBJ_ID = 1


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SamSession:
    """
    Gère une session de re-segmentation SAM2 sur un intervalle de frames.
    Les frames sont passées directement en mémoire (liste de np.ndarray HxWx3 uint8),
    sans passer par une extraction JPEG sur disque.

    Cycle de vie : create -> prepare_frames -> init_predictor -> init_inference_state
                   -> (add points, propagate)* -> export_masks -> close
    """

    def __init__(self, work_root):
        """
        work_root: dossier racine de travail (ex: 'sam2_session') où seront stockés
        uniquement les masques PNG de sortie (plus de frames JPEG à gérer).
        """
        self.work_root = work_root
        self.mask_dir = os.path.join(work_root, "masks")

        self.predictor = None
        self.inference_state = None

        # Table de correspondance : index SAM2 (0..N-1) -> numéro de frame EXR réel
        self.sam_idx_to_frame_num = {}
        self.frame_num_to_sam_idx = {}

        self.exr_width = None
        self.exr_height = None

        # Liste des frames en mémoire (np.ndarray HxWx3 uint8 RGB), conservée tant que
        # la session est active - nécessaire car init_state peut en avoir besoin
        self._frames_in_memory = []

        self.is_initialized = False

    # ---------------------- Préparation des dossiers ----------------------

    def _reset_work_dirs(self):
        if os.path.exists(self.mask_dir):
            shutil.rmtree(self.mask_dir)
        os.makedirs(self.mask_dir, exist_ok=True)

    # ---------------------- Préparation des frames en mémoire ----------------------

    def prepare_frames(self, frame_numbers, get_base_raw_fn, tonemap_fn):
        """
        Construit la liste de frames (np.ndarray HxWx3 uint8, RGB) directement en mémoire,
        dans l'ordre de l'intervalle, et établit la correspondance sam_idx <-> frame_num.

        frame_numbers: liste ordonnée des numéros de frame EXR réels de l'intervalle
        get_base_raw_fn: fonction (frame_num) -> (raw_array, w, h, nchannels), typiquement
                         MainWindow.get_base_raw
        tonemap_fn: fonction (raw_array, nchannels) -> rgb_uint8 (H,W,3), typiquement
                    raw_to_rgb8 avec les paramètres fixes d'exposition/gamma

        Retourne le nombre de frames préparées.
        """
        self._reset_work_dirs()

        self.sam_idx_to_frame_num = {}
        self.frame_num_to_sam_idx = {}
        self._frames_in_memory = []

        for sam_idx, frame_num in enumerate(frame_numbers):
            raw_data = get_base_raw_fn(frame_num)
            if raw_data is None:
                raise IOError(f"Impossible de charger la frame {frame_num} pour la session SAM2")

            raw_arr, w, h, nchannels = raw_data
            rgb8 = tonemap_fn(raw_arr, nchannels)  # (H, W, 3) uint8, RGB

            if self.exr_width is None:
                self.exr_width = w
                self.exr_height = h

            # S'assure que le tableau est bien contigu en mémoire, requis par certaines
            # opérations internes de SAM2 (sinon risque d'erreur silencieuse ou de copie coûteuse)
            rgb8 = np.ascontiguousarray(rgb8)

            self._frames_in_memory.append(rgb8)
            self.sam_idx_to_frame_num[sam_idx] = frame_num
            self.frame_num_to_sam_idx[frame_num] = sam_idx

        return len(frame_numbers)

    # ---------------------- Chargement du predictor ----------------------

    def load_predictor(self):
        device = get_device()
        self.predictor = build_sam2_video_predictor(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=device)
        return device

    def init_inference_state(self):
        """
        Initialise l'état d'inférence en passant directement la liste de frames
        en mémoire (np.ndarray) à video_path, au lieu d'un chemin de dossier.
        """
        if self.predictor is None:
            raise RuntimeError("Le predictor SAM2 n'est pas chargé (appelez load_predictor() avant).")

        if not self._frames_in_memory:
            raise RuntimeError("Aucune frame préparée (appelez prepare_frames() avant).")

        self.inference_state = self.predictor.init_state(
            video_path=self._frames_in_memory,
            async_loading_frames=True,
            offload_video_to_cpu=True,
        )
        self.is_initialized = True

    def reset_inference_state(self):
        """Réinitialise l'état d'inférence (efface tous les points/masques de cette session)"""
        if self.predictor is not None and self.inference_state is not None:
            self.predictor.reset_state(self.inference_state)

    # ---------------------- Points et segmentation (frame de référence) ----------------------

    def add_points(self, sam_frame_idx, points, labels):
        """
        Ajoute des points positifs/négatifs sur une frame et retourne le masque résultant.

        points: liste de (x, y) en coordonnées pixel de l'image (résolution originale)
        labels: liste de 1 (positif) ou 0 (négatif), même longueur que points

        Retourne: masque binaire uint8 (H, W) avec 0/255, ou None en cas d'erreur
        """
        if not points:
            return None

        input_points = np.array(points, dtype=np.float32)
        input_labels = np.array(labels, dtype=np.int32)

        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=sam_frame_idx,
            obj_id=FIXED_OBJ_ID,
            points=input_points,
            labels=input_labels,
            clear_old_points=True,
        )

        for i, out_obj_id in enumerate(out_obj_ids):
            if out_obj_id != FIXED_OBJ_ID:
                continue
            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            mask_u8 = (mask * 255).astype(np.uint8)
            self._save_mask_png(sam_frame_idx, mask_u8)
            return mask_u8

        return None

    def _save_mask_png(self, sam_frame_idx, mask_u8):
        mask_filename = os.path.join(self.mask_dir, f"{sam_frame_idx:05d}", f"{FIXED_OBJ_ID}.png")
        os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
        cv2.imwrite(mask_filename, mask_u8)

    def clear_frame_points(self, sam_frame_idx):
        """
        Efface les points et les données de tracking pour une frame donnée,
        pour l'objet fixe géré par cette session (FIXED_OBJ_ID).
        """
        if self.predictor is None or self.inference_state is None:
            return False

        obj_idx = self.inference_state.get("obj_id_to_idx", {}).get(FIXED_OBJ_ID)
        if obj_idx is None:
            return False

        tracked = self.inference_state["frames_tracked_per_obj"][obj_idx]
        if sam_frame_idx not in tracked:
            return False

        self.predictor.clear_all_prompts_in_frame(
            self.inference_state, sam_frame_idx, FIXED_OBJ_ID, need_output=False
        )
        tracked.pop(sam_frame_idx, None)

        for d in (self.inference_state["output_dict_per_obj"][obj_idx],
                  self.inference_state["temp_output_dict_per_obj"][obj_idx]):
            d["non_cond_frame_outputs"].pop(sam_frame_idx, None)

        mask_filename = os.path.join(self.mask_dir, f"{sam_frame_idx:05d}", f"{FIXED_OBJ_ID}.png")
        if os.path.exists(mask_filename):
            os.remove(mask_filename)

        return True

    # ---------------------- Propagation ----------------------

    def propagate(self, start_sam_idx, max_frames_to_track=None, reverse=False, progress_callback=None):
        """
        Propage le masque depuis start_sam_idx sur l'intervalle SAM2 (indices locaux 0..N-1).
        """
        results = {}
        total_frames = len(self.sam_idx_to_frame_num)

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state,
                start_frame_idx=start_sam_idx,
                max_frame_num_to_track=max_frames_to_track,
                reverse=reverse):

            for i, out_obj_id in enumerate(out_obj_ids):
                if out_obj_id != FIXED_OBJ_ID:
                    continue
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                mask_u8 = (mask * 255).astype(np.uint8)
                self._save_mask_png(out_frame_idx, mask_u8)
                results[out_frame_idx] = mask_u8

            if progress_callback is not None:
                progress_callback(out_frame_idx, total_frames)

        return results

    def propagate_one_frame_forward(self, current_sam_idx):
        max_idx = len(self.sam_idx_to_frame_num) - 1
        if current_sam_idx >= max_idx:
            return current_sam_idx, None

        results = self.propagate(
            start_sam_idx=current_sam_idx,
            max_frames_to_track=1,
            reverse=False,
        )

        new_idx = current_sam_idx + 1
        mask = results.get(new_idx)
        return new_idx, mask

    def propagate_one_frame_backward(self, current_sam_idx):
        if current_sam_idx <= 0:
            return current_sam_idx, None

        results = self.propagate(
            start_sam_idx=current_sam_idx,
            max_frames_to_track=1,
            reverse=True,
        )

        new_idx = current_sam_idx - 1
        mask = results.get(new_idx)
        return new_idx, mask

    def propagate_n_frames_forward(self, current_sam_idx, n=10):
        """
        Propage le masque jusqu'à n frames vers l'avant depuis current_sam_idx.
        Retourne (last_sam_idx, results) où results est le dict {sam_idx: mask_u8}
        de toutes les frames traitées, et last_sam_idx le dernier index atteint
        (current_sam_idx si rien n'a pu être propagé).
        """
        max_idx = len(self.sam_idx_to_frame_num) - 1
        if current_sam_idx >= max_idx:
            return current_sam_idx, {}

        results = self.propagate(
            start_sam_idx=current_sam_idx,
            max_frames_to_track=n,
            reverse=False,
        )

        last_idx = max(results.keys()) if results else current_sam_idx
        return last_idx, results

    def propagate_n_frames_backward(self, current_sam_idx, n=10):
        """
        Propage le masque jusqu'à n frames vers l'arrière depuis current_sam_idx.
        Retourne (last_sam_idx, results) où results est le dict {sam_idx: mask_u8}
        de toutes les frames traitées, et last_sam_idx le dernier index atteint
        (current_sam_idx si rien n'a pu être propagé).
        """
        if current_sam_idx <= 0:
            return current_sam_idx, {}

        results = self.propagate(
            start_sam_idx=current_sam_idx,
            max_frames_to_track=n,
            reverse=True,
        )

        last_idx = min(results.keys()) if results else current_sam_idx
        return last_idx, results

    def load_mask(self, sam_frame_idx):
        """Charge un masque déjà calculé pour une frame donnée, ou None si absent"""
        mask_filename = os.path.join(self.mask_dir, f"{sam_frame_idx:05d}", f"{FIXED_OBJ_ID}.png")
        if not os.path.exists(mask_filename):
            return None
        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
        return mask

    # ---------------------- Nettoyage ----------------------

    def close(self):
        """Libère la mémoire GPU/CPU et nettoie les dossiers temporaires"""
        self.predictor = None
        self.inference_state = None
        self._frames_in_memory = []  # libère les frames en mémoire
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if os.path.exists(self.work_root):
            shutil.rmtree(self.work_root, ignore_errors=True)