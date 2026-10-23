import json
import math
from dataclasses import dataclass, field

SIZE_THRESHOLDS = [252, 504, 1008]

@dataclass
class TrackChunk:
    """A chunk of consecutive frames with their boxes."""
    start_frame : int
    end_frame   : int
    boxes       : dict = field(default_factory=dict)  # {frame_idx: [x1, y1, x2, y2]}

    def __repr__(self):
        return f"TrackChunk(frames={self.start_frame}-{self.end_frame}, n={len(self.boxes)})"

@dataclass
class TrackChunkWithSlices:
    """A chunk of consecutive slices of frames with their boxes."""
    start_frame   : int
    end_frame     : int
    slices        : list

    #boxes         : dict = field(default_factory=dict)  # {frame_idx: [x1, y1, x2, y2]}

    def __repr__(self):
        return f"TrackChunkWithSlices(frames={self.start_frame}-{self.end_frame}, s={len(self.slices)})"


def compute_par_dimensions(frame_w: int, frame_h: int, par: float) -> tuple[int, int]:
    """
    image_display_w = frame_w
    image_display_h = frame_h / par
    """
    display_w = frame_w
    display_h = int(round(frame_h / par))
    return display_w, display_h


def box_to_display(box: list, par: float) -> list:
    x1, y1, x2, y2 = box
    return [
        x1,
        int(round(y1 / par)),
        x2,
        int(round(y2 / par)),
    ]


def box_to_source(box: list, par: float) -> list:
    x1, y1, x2, y2 = box
    return [
        x1,
        int(round(y1 * par)),
        x2,
        int(round(y2 * par)),
    ]


def compute_iou(box1: list, box2: list) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def merge_boxes(box1: list, box2: list, iou_threshold: float = 0.5) -> tuple[list, str]:
    """
    Merge 2 boxes xyxy by taking the bounding box, if their IoU is higher than the threshold. 
    Else return the first box.
    """
    iou = compute_iou(box1, box2)

    if iou >= iou_threshold:
        merged = [
            min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3]),
        ]
        return merged, f"bounding (IoU={iou:.2f})"
    else:
        return box1, f"forward     (IoU={iou:.2f} < threshold={iou_threshold})"


def get_target_size(boxes: dict, par: float, roundSize: bool = True, squareBox: bool = True, extension_factor: float = 1.0):
    """
    Compute the target size for a set of boxes.
    Comparisons occur in the display space (with pixel aspect ratio applied)).
    """
    max_size_w = 0
    max_size_h = 0
    for box in boxes.values():
        display_box = box_to_display(box, par)
        x1, y1, x2, y2 = display_box
        w = x2 - x1
        h = y2 - y1
        max_size_w = max(max_size_w, w)
        max_size_h = max(max_size_h, h)

    max_size_w = int(extension_factor * max_size_w)
    max_size_h = int(extension_factor * max_size_h)

    if squareBox:
        max_size_w = max(max_size_w, max_size_h)
        max_size_h = max_size_w
        if roundSize:
            for threshold in SIZE_THRESHOLDS :
                if max_size_w < threshold and max_size_h < threshold:
                    return threshold, threshold

    return max_size_w, max_size_h


def expand_box(box: list, target_size_w: int, target_size_h: int, par: float, frame_w: int, frame_h: int) -> list:
    """
    Expand a box at target_size x target_size in display space,
    and convert back in source space.
    """
    display_w, display_h = compute_par_dimensions(frame_w, frame_h, par)

    # Conversion to display space
    display_box = box_to_display(box, par)
    x1, y1, x2, y2 = display_box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Centered expansion in display space
    new_x1 = cx - target_size_w / 2
    new_y1 = cy - target_size_h / 2
    new_x2 = cx + target_size_w / 2
    new_y2 = cy + target_size_h / 2

    # Horizontal adjust
    if new_x1 < 0:
        new_x2 = min(new_x2 - new_x1, display_w)
        new_x1 = 0
    elif new_x2 > display_w:
        new_x1 = max(new_x1 - (new_x2 - display_w), 0)
        new_x2 = display_w

    # Vertical adjust
    if new_y1 < 0:
        new_y2 = min(new_y2 - new_y1, display_h)
        new_y1 = 0
    elif new_y2 > display_h:
        new_y1 = max(new_y1 - (new_y2 - display_h), 0)
        new_y2 = display_h

    expanded_display = [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

    # Back conversion to source space
    return box_to_source(expanded_display, par)


def split_into_chunks(boxes: dict) -> list[TrackChunk]:
    """
    Split dictionnary {frame_idx: box} in chunks of consecutive frames.
    """
    if not boxes:
        return []

    sorted_frames = sorted(boxes.keys())
    chunks = []
    chunk_boxes = {sorted_frames[0]: boxes[sorted_frames[0]]}

    for prev_frame, curr_frame in zip(sorted_frames, sorted_frames[1:]):
        if curr_frame == prev_frame + 1:
            chunk_boxes[curr_frame] = boxes[curr_frame]
        else:
            chunks.append(TrackChunk(
                start_frame = min(chunk_boxes.keys()),
                end_frame   = max(chunk_boxes.keys()),
                boxes       = chunk_boxes
            ))
            chunk_boxes = {curr_frame: boxes[curr_frame]}

    chunks.append(TrackChunk(
        start_frame = min(chunk_boxes.keys()),
        end_frame   = max(chunk_boxes.keys()),
        boxes       = chunk_boxes
    ))

    return chunks


def split_into_chunks_with_slices(boxes: dict, slice_size: int, overlap: int) -> list[TrackChunkWithSlices]:
    """
    Split dictionnary {frame_idx: box} in sliced chunks of consecutive frames.
    """
    if not boxes:
        return []

    sorted_frames = sorted(boxes.keys())
    chunks = []

    slices = []
    slice = {sorted_frames[0]: boxes[sorted_frames[0]]}

    for prev_frame, curr_frame in zip(sorted_frames, sorted_frames[1:]):
        if curr_frame == prev_frame + 1:
            slice[curr_frame] = (boxes[curr_frame])
            if len(slice) == slice_size:
                slices.append(slice)
                keys = [] if overlap==0 else list(slice.keys())[-overlap:]
                slice = {k: slice[k] for k in keys}
        else:
            if slice and (not slices or len(slice) > overlap):
                slices.append(slice)
            chunks.append(TrackChunkWithSlices(
                start_frame = min(slices[0].keys()),
                end_frame   = max(slices[len(slices)-1].keys()),
                slices      = slices
            ))
            slices = []
            slice = {curr_frame: boxes[curr_frame]}

    if slice and (not slices or len(slice) > overlap):
        slices.append(slice)
    chunks.append(TrackChunkWithSlices(
        start_frame = min(slices[0].keys()),
        end_frame   = max(slices[len(slices)-1].keys()),
        slices      = slices
    ))

    return chunks

def extract_tracking(
    json_path     : str,
    frame_w       : int,
    frame_h       : int,
    x2_ok         : bool = True,
    x4_ok         : bool = True,
    roundCrop     : bool = True,
    squareBox     : bool = True,
    exp_factor    : float = 1.0,
    par           : float = 1.0,
    iou_threshold : float = 0.5
) -> dict:
    """
    Extract bounding boxes per object and organize them in chunck of consecutive frames.
    Coordinates in the json file are supposed to be in the original source space, with the pixel aspect ratio not applied.
    The pixel aspect ratio is applied by reducing the raw number to deliver coordinates in the display space.
    When possible, all delivered bounding boxes for a given object are sized as 252x252, 504x504 or 1008x1008,
    to match with sam3 optimal model input.
    Return a dictionnary :
    {
        "object_0_0": [TrackChunk(frames=0-10), TrackChunk(frames=15-20), ...],
        "object_0_1": [TrackChunk(frames=0-20)],
        ...
        "object_1_0": [TrackChunk(frames=0-5), TrackChunk(frames=6-10), ...],
        ...
    }
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    result = {}

    for label, directions in data.items():
        forward  = directions.get("forward",  {})
        merged = directions.get("merged", {})

        all_object_ids = set()
        if merged:
            for frame_data in merged.values():
                all_object_ids.update(frame_data.keys())
        else:
            for frame_data in forward.values():
                all_object_ids.update(frame_data.keys())

        for obj_id in sorted(all_object_ids, key=int):
            key = f"{label}_{obj_id}"
            raw_boxes = {}

            if merged:
                all_frames = sorted(set(merged.keys()),key=int)
                for frame_idx in all_frames:
                    box = merged.get(frame_idx,  {}).get(obj_id)
                    if box:
                        raw_boxes[int(frame_idx)] = box
            else:
                all_frames = sorted(set(forward.keys()),key=int)
                for frame_idx in all_frames:
                    box = forward.get(frame_idx,  {}).get(obj_id)
                    if box:
                        raw_boxes[int(frame_idx)] = box

            # --- compute target size ---
            target_size_w, target_size_h = get_target_size(raw_boxes, par, roundCrop, squareBox, exp_factor)

            # --- Expand boxes in display space ---
            expanded_boxes = {}
            for frame_idx, box in raw_boxes.items():
                if target_size_w is not None:
                    if target_size_w < SIZE_THRESHOLDS[1] and target_size_h < SIZE_THRESHOLDS[1] and not x4_ok and roundCrop:
                        target_size_w = SIZE_THRESHOLDS[1]
                        target_size_h = SIZE_THRESHOLDS[1]
                    if target_size_w < SIZE_THRESHOLDS[2] and target_size_h < SIZE_THRESHOLDS[2] and not x2_ok and roundCrop:
                        target_size_w = SIZE_THRESHOLDS[2]
                        target_size_h = SIZE_THRESHOLDS[2]
                    expanded = expand_box(box, target_size_w, target_size_h, par, frame_w, frame_h)
                    expanded_boxes[frame_idx] = expanded
                else:
                    expanded_boxes[frame_idx] = box

            # --- Split in chunks ---
            chunks = split_into_chunks(expanded_boxes)

            result[key] = chunks

    return result

def extract_tracking_with_slices(
    json_path     : str,
    frame_w       : int,
    frame_h       : int,
    slice_size    : int,
    overlap       : int,
    exp_factor    : float = 1.0,
    par           : float = 1.0,
) -> dict:
    """
    Extract bounding boxes per object and organize them in chunck and slices of consecutive frames.
    Coordinates in the json file are supposed to be in the original source space, with the pixel aspect ratio not applied.
    The pixel aspect ratio is applied by reducing the raw number to deliver coordinates in the display space.
    Return a dictionnary :
    {
        "object_0_0": [TrackChunk[slice_0(frames=0 - slice_size-1), slice_1(frames=slice_size-1-overlap - 2*slice_size-1-overlap), ...], ...]
        "object_0_1": [TrackChunk(slice_0(frames=0-3)]],
        ...
        "object_1_0": ...,
        ...
    }
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    result = {}

    for label, directions in data.items():
        forward  = directions.get("forward",  {})
        merged = directions.get("merged", {})

        all_object_ids = set()
        if merged:
            for frame_data in merged.values():
                all_object_ids.update(frame_data.keys())
        else:
            for frame_data in forward.values():
                all_object_ids.update(frame_data.keys())

        for obj_id in sorted(all_object_ids, key=int):
            key = f"{label}_{obj_id}"
            raw_boxes = {}

            if merged:
                all_frames = sorted(set(merged.keys()),key=int)
                for frame_idx in all_frames:
                    box = merged.get(frame_idx,  {}).get(obj_id)
                    if box:
                        raw_boxes[int(frame_idx)] = box
            else:
                all_frames = sorted(set(forward.keys()),key=int)
                for frame_idx in all_frames:
                    box = forward.get(frame_idx,  {}).get(obj_id)
                    if box:
                        raw_boxes[int(frame_idx)] = box

            # --- Split in chunks ---
            chunks = split_into_chunks_with_slices(raw_boxes, slice_size, overlap)

            for c, chunk in enumerate(chunks):
                for s, slice in enumerate(chunk.slices):
                    # --- compute target size ---
                    target_size_w, target_size_h = get_target_size(slice, par, False, False, exp_factor)

                    # --- Expand boxes in display space ---
                    expanded_boxes = {}
                    for frame_idx, box in slice.items():
                        if target_size_w is not None:
                            expanded = expand_box(box, target_size_w, target_size_h, par, frame_w, frame_h)
                            expanded_boxes[frame_idx] = expanded
                        else:
                            expanded_boxes[frame_idx] = box

                    chunks[c].slices[s] = expanded_boxes

            result[key] = chunks

    return result


def tile_chunk(chunk: TrackChunk, targetTileSize: int, min_overlap: int, max_tile_number_per_dimension: int, par: float, fitSam3: bool, logger) -> list[TrackChunk]:
    """
    Tile a chunk of consecutive frames by creating a set of chunks.
    One chunk on the same consecutive frames for every tiles.
    """
    box = chunk.boxes[chunk.start_frame]
    x1, y1, x2, y2 = box_to_display(box, par)
    box_w = x2 - x1
    box_h = y2 - y1

    squareBox = (box_w == box_h)

    logger.info(f"Boxes size {box_w}x{box_h}")

    tile_size_w_min = math.ceil((box_w + (max_tile_number_per_dimension - 1) * min_overlap) / max_tile_number_per_dimension)
    tile_size_h_min = math.ceil((box_h + (max_tile_number_per_dimension - 1) * min_overlap) / max_tile_number_per_dimension)
    tile_size_w = min(max(targetTileSize, tile_size_w_min), box_w)
    tile_size_h = min(max(targetTileSize, tile_size_h_min), box_h)

    if squareBox and fitSam3:
        if tile_size_w < SIZE_THRESHOLDS[0] and SIZE_THRESHOLDS[0] < box_w:
            tile_size_w = SIZE_THRESHOLDS[0]
            tile_size_h = SIZE_THRESHOLDS[0]
        elif tile_size_w < SIZE_THRESHOLDS[1] and SIZE_THRESHOLDS[1] < box_w:
            tile_size_w = SIZE_THRESHOLDS[1]
            tile_size_h = SIZE_THRESHOLDS[1]
        elif tile_size_w < SIZE_THRESHOLDS[2] and SIZE_THRESHOLDS[2] < box_w:
            tile_size_w = SIZE_THRESHOLDS[2]
            tile_size_h = SIZE_THRESHOLDS[2]

    tile_nb_w = (box_w // tile_size_w) + 1
    tile_nb_h = (box_h // tile_size_h) + 1
    overlap_w = 0
    overlap_h = 0
    start_cols = [0]
    start_rows = [0]

    if tile_nb_w > 1 and tile_size_w < box_w:
        overlap_w = ((tile_nb_w * tile_size_w) - box_w) // (tile_nb_w - 1)
        if overlap_w < min_overlap:
            tile_nb_w = tile_nb_w + 1
            new_overlap_w = ((tile_nb_w * tile_size_w) - box_w) // (tile_nb_w - 1)
            if new_overlap_w <= box_w // 2:
                overlap_w = new_overlap_w
        k = tile_size_w - overlap_w
        while k <= box_w - tile_size_w:
            start_cols.append(k)
            k = k + tile_size_w - overlap_w
        if start_cols[len(start_cols)- 1] != box_w - tile_size_w:
            start_cols.append(box_w - tile_size_w)

    if tile_nb_h > 1 and tile_size_h < box_h:
        overlap_h = ((tile_nb_h * tile_size_h) - box_h) // (tile_nb_h - 1)
        if overlap_h < min_overlap:
            tile_nb_h = tile_nb_h + 1
            new_overlap_h = ((tile_nb_h * tile_size_h) - box_h) // (tile_nb_h - 1)
            if new_overlap_h <= box_h // 2:
                overlap_h = new_overlap_h
        k = tile_size_h - overlap_h
        while k <= box_h - tile_size_h and box_h > tile_size_h:
            start_rows.append(k)
            k = k + tile_size_h - overlap_h
        if start_rows[len(start_rows)- 1] != box_h - tile_size_h:
            start_rows.append(box_h - tile_size_h)

    logger.info(f"Chunk tiling in {len(start_cols)}x{len(start_rows)}")
    logger.info(f"tile size: {tile_size_w}x{tile_size_h}")
    logger.info(f"min overlaps: {overlap_w}x{overlap_h}")

    chunk_tiles = []
    for r in start_rows:
        for c in start_cols:
            chunk_tile = TrackChunk(start_frame = chunk.start_frame, end_frame = chunk.end_frame, boxes = {})
            for frame_id, box in chunk.boxes.items():
                x1, y1, x2, y2 = box_to_display(box, par)
                tile_display = [x1 + c, y1 + r, x1 + c + tile_size_w, y1 + r + tile_size_h]
                chunk_tile.boxes[frame_id] = box_to_source(tile_display, par)
            chunk_tiles.append(chunk_tile)

    return chunk_tiles


def get_bbox_center(bbox):
    """Compute center (x, y) of a bbox (x1, y1, x2, y2)."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def compute_iob(box_1, box_2):
    """Proportion of box_1 included in box_2."""
    x1_i = max(box_1[0], box_2[0])
    y1_i = max(box_1[1], box_2[1])
    x2_i = min(box_1[2], box_2[2])
    y2_i = min(box_1[3], box_2[3])
    if x1_i >= x2_i or y1_i >= y2_i: return 0.0
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    box_1_area = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    return intersection_area / (box_1_area + 1e-6)

def associate_items(data, item1:str, item2:str, min_iob_threshold:float=0.1, motion_weight:float=2.0,
                     min_score:float=0.0):
    """
    Associate each object of item1 (secondary prompt) to at most one object of item2 (primary prompt).
    data: {item1: {frame: {obj_id: [x1, y1, x2, y2]}}, item2: {frame: {obj_id: [x1, y1, x2, y2]}}}
    For each pair of objects co-existing in a frame, a score combining the inclusion of the item1 bbox
    in the item2 bbox (iob) and the similarity of their motions is accumulated over the sequence, then
    averaged on the total number of frames, so that the association duration is taken into account.
    Each item1 object is then attributed to the item2 object with the highest score, if this score reaches
    min_score. Several item1 objects can be attributed to the same item2 object.
    Return {item1_obj_id: {"item2_id": item2_obj_id, "confidence_score": normalized score in [0, 1]}}
    """
    import numpy as np
    from collections import defaultdict

    traj_item1 = defaultdict(dict) # secondary
    traj_item2 = defaultdict(dict) # primary
    
    for frame, item1s in data[item1].items():
        for b_id, bbox in item1s.items():
            traj_item1[b_id][frame] = get_bbox_center(bbox)
            
    for frame, item2s in data[item2].items():
        for p_id, bbox in item2s.items():
            traj_item2[p_id][frame] = get_bbox_center(bbox)

    scoring_pool = defaultdict(lambda: defaultdict(float))
    frame_counters = defaultdict(lambda: defaultdict(int))
    
    all_frames = sorted(set(data[item1].keys()).union(data[item2].keys()), key=int)

    # 1. Computation and accumulation
    for idx, frame in enumerate(all_frames[:-1]):
        next_frame = all_frames[idx + 1]
        
        frame_item1s = data[item1].get(frame, {})
        frame_item2s = data[item2].get(frame, {})
        
        for b_id, b_bbox in frame_item1s.items():
            for p_id, p_bbox in frame_item2s.items():
                
                iob_score = compute_iob(b_bbox, p_bbox)
                if iob_score < min_iob_threshold:
                    continue  
                
                motion_score = 1.0  
                if (frame in traj_item1[b_id] and next_frame in traj_item1[b_id] and
                    frame in traj_item2[p_id] and next_frame in traj_item2[p_id]):
                    
                    v_item1 = np.array(traj_item1[b_id][next_frame]) - np.array(traj_item1[b_id][frame])
                    v_item2 = np.array(traj_item2[p_id][next_frame]) - np.array(traj_item2[p_id][frame])
                    velocity_diff = np.linalg.norm(v_item1 - v_item2)
                    motion_score = np.exp(-velocity_diff / 10.0)
                
                combined_score = iob_score + (motion_weight * motion_score)
                
                scoring_pool[b_id][p_id] += combined_score
                frame_counters[b_id][p_id] += 1  # Co-existent frame number accumulation

    # 2. Normalisation
    all_combinations = []
    max_score_per_frame = 1.0 + motion_weight 

    for b_id, item2_scores in scoring_pool.items():
        for p_id, total_accumulated_score in item2_scores.items():

            # A. Average score per frame on the full sequence to take into account the association duration
            mean_score = total_accumulated_score / len(all_frames)
            
            # B. Normalisation
            normalized_score = mean_score / max_score_per_frame
            
            all_combinations.append({
                'item1_id': b_id, 
                'item2_id': p_id, 
                'score': normalized_score
            })
            
    # 3. Sorting
    all_combinations.sort(key=lambda x: x['score'], reverse=True)

    # 4. Greedy algorithm (each item1 attributed once, an item2 can receive several item1s)
    final_associations = {}

    for combo in all_combinations:
        b_id = combo['item1_id']
        p_id = combo['item2_id']
        
        # Combos being sorted, the first one met for an item1 is its best one
        if b_id not in final_associations and combo['score'] >= min_score:
            final_associations[b_id] = {"item2_id": p_id, "confidence_score": round(combo['score'], 3)}
            
    return final_associations
