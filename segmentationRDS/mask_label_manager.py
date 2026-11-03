#!/usr/bin/env python3
"""
Management of the mask bounding boxes stored in a bboxes.json file.
"""

import os
import copy
import json
import numpy as np


class BBoxesManager:
    """
    Reads and writes the bboxes.json file for a given label.
    """

    def __init__(self, json_path, label):
        self.json_path = json_path
        self.label = label
        self.data = {}
        if os.path.exists(self.json_path):
            with open(self.json_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

    def save(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def create_label_from(self, src_label):
        """
        Create (or replace) the managed label entry as an independent copy of the bboxes
        of src_label for its active tracking key. The src_label entry is left untouched.
        """
        src_label_data = self.data.get(src_label, {})
        tracking_key = self.tracking_key_of(src_label_data)
        self.data[self.label] = {tracking_key: copy.deepcopy(src_label_data.get(tracking_key, {}))}

    def keep_only_label(self):
        """Drop all the labels except the managed one"""
        self.data = {self.label: self.data.get(self.label, {})}

    def get_tracking_data(self):
        """Return the {frame: {obj_id: bbox}} dict of the managed label for the active tracking key"""
        return self.data.get(self.label, {}).get(self._get_tracking_key(), {})

    @staticmethod
    def tracking_key_of(label_data):
        """Return 'merged' if present and not empty in label_data, otherwise 'forward'"""
        if label_data.get("merged"):
            return "merged"
        return "forward"

    def _get_tracking_key(self):
        """Return 'merged' if present and not empty for the managed label, otherwise 'forward'"""
        return self.tracking_key_of(self.data.get(self.label, {}))

    def set_bbox(self, frame_num, obj_id, bbox):
        """
        Set/replace the [x1,y1,x2,y2] bbox for a given frame/obj_id in the active tracking key.
        Intermediate structures are created if needed.
        """
        tracking_key = self._get_tracking_key()
        frame_data = self.data.setdefault(self.label, {}).setdefault(tracking_key, {}).setdefault(str(frame_num), {})
        frame_data[str(obj_id)] = list(bbox)

    def remove_bbox(self, frame_num, obj_id):
        """Remove the bbox entry for a given frame/obj_id, if it exists"""
        tracking_data = self.get_tracking_data()
        frame_key = str(frame_num)

        frame_data = tracking_data.get(frame_key)
        if frame_data is None:
            return

        frame_data.pop(str(obj_id), None)

        # Cleanup: remove the frame entry when it no longer holds any obj_id
        if not frame_data:
            tracking_data.pop(frame_key, None)

    def update_from_mask(self, frame_num, obj_id, mask_u8):
        """
        Update (or remove) the bbox for frame_num/obj_id from a binary uint8 mask.
        If the mask is empty (no active pixel), the bbox entry is removed.
        Otherwise it is set to the bounding rectangle of the active pixels.
        """
        ys, xs = np.where(mask_u8 > 127)
        if len(xs) == 0:
            self.remove_bbox(frame_num, obj_id)
            return

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        self.set_bbox(frame_num, obj_id, [x1, y1, x2, y2])

    def replace_frame_with_mask(self, frame_num, obj_id, mask_u8):
        """
        Replace all the bboxes of frame_num by a single bbox for obj_id computed from a binary uint8 mask.
        If the mask is empty, the frame entry is removed.
        """
        self.get_tracking_data().pop(str(frame_num), None)
        self.update_from_mask(frame_num, obj_id, mask_u8)
