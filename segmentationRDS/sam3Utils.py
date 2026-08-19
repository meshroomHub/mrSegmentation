import numpy as np
from typing import Dict, Optional, Tuple

FrameResults = Dict[int, Dict[int, dict]]  # {frame_idx: {obj_id: {prob, box_xywh, mask}}}


def xywhNorm2xyxy(xywhNorm, width: int, height: int):
    # Convert a normalized rectangle (x_topLeft, y_topLeft, width, height) with all values in range [0, 1]
    # into a rectangle (x_topLeft, y_topLeft, x_bottomRight, y_bottomRight) with x values in range [0, width] and y values in range [0, height]
    xn, yn, wn, hn = xywhNorm
    x = int(xn * width)
    y = int(yn * height)
    w = int(wn * width)
    h = int(hn * height)
    return (x, y, x+w, y+h)


def prepareMasksForVisualization(frame_to_output) -> FrameResults:
    # frame_to_obj_masks --> {frame_idx: {'output_probs': np.array, `out_obj_ids`: np.array, `out_binary_masks`: np.array}}
    _processed_out = {}
    for frame_idx, out in frame_to_output.items():
        _processed_out[frame_idx] = {}
        for idx, obj_id in enumerate(out["out_obj_ids"].tolist()):
            if out["out_binary_masks"][idx].any():
                _processed_out[frame_idx][obj_id] = {"mask": out["out_binary_masks"][idx], "box_xywh": out["out_boxes_xywh"][idx], "prob": out["out_probs"][idx]}
    return _processed_out


def propagateInVideo(predictor, session_id, start_frame_idx=None, max_frame_num_to_track=None, direction="both"):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction=direction,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


def compute_box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def compute_similarity(det_a: dict, det_b: dict, use_mask: bool = True) -> float:
    if use_mask and det_a["mask"] is not None and det_b["mask"] is not None:
        iou = compute_mask_iou(det_a["mask"], det_b["mask"])
    else:
        iou = compute_box_iou(det_a["box_xywh"], det_b["box_xywh"])
    prob_score = (det_a["prob"] + det_b["prob"]) / 2.0
    return 0.7 * iou + 0.3 * prob_score


def match_detections(
    src_a: Dict[int, dict],
    src_b: Dict[int, dict],
    similarity_threshold: float = 0.3,
    use_mask: bool = True,
) -> Tuple[Dict[int, int], list, list]:
    """
    Match objects from src_a to src_b using the Hungarian algorithm.

    Returns:
        matches     : {id_a -> id_b}
        unmatched_a : ids in src_a with no match
        unmatched_b : ids in src_b with no match
    """
    from scipy.optimize import linear_sum_assignment

    ids_a = list(src_a.keys())
    ids_b = list(src_b.keys())

    if not ids_a or not ids_b:
        return {}, list(ids_a), list(ids_b)

    cost = np.array(
        [
            [1.0 - compute_similarity(src_a[a], src_b[b], use_mask) for b in ids_b]
            for a in ids_a
        ],
        dtype=np.float32,
    )

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = {}
    unmatched_a = list(ids_a)
    unmatched_b = list(ids_b)

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < (1.0 - similarity_threshold):
            a, b = ids_a[r], ids_b[c]
            matches[a] = b
            unmatched_a.remove(a)
            unmatched_b.remove(b)

    return matches, unmatched_a, unmatched_b


def merge_two_detections(det_fwd: Optional[dict], det_bwd: Optional[dict]) -> dict:
    """
    Merge two detections for the same instance on the same frame.
    If only one is available, return it as-is.
    """
    if det_fwd is None:
        return det_bwd
    if det_bwd is None:
        return det_fwd

    prob = max(det_fwd["prob"], det_bwd["prob"])

    w_f = det_fwd["prob"] / (det_fwd["prob"] + det_bwd["prob"] + 1e-8)
    w_b = 1.0 - w_f

    if det_fwd["mask"] is not None and det_bwd["mask"] is not None:
        score = (
            w_f * det_fwd["mask"].astype(np.float32)
            + w_b * det_bwd["mask"].astype(np.float32)
        )
        mask = (score > 0.5).astype(bool)
    else:
        mask = det_fwd["mask"] if det_fwd["mask"] is not None else det_bwd["mask"]

    box = w_f * det_fwd["box_xywh"] + w_b * det_bwd["box_xywh"]

    return {"prob": prob, "box_xywh": box, "mask": mask}


def merge_chunk_forward_backward(
    fwd_results: FrameResults,
    bwd_results: FrameResults,
    similarity_threshold: float = 0.3,
    use_mask: bool = True,
) -> FrameResults:
    """
    Merge forward and backward results for the same chunk.

    - Forward IDs are used as the local reference IDs.
    - Objects seen only in backward receive a temporary negative local ID.
    - The bwd->fwd mapping is stabilised by majority vote across all frames.
    """
    all_frames = sorted(set(fwd_results) | set(bwd_results))

    # Build a stable bwd_id -> fwd_id mapping via majority vote over all frames
    votes: Dict[int, Dict[int, int]] = {}  # {bwd_id: {fwd_id: count}}

    for frame_idx in all_frames:
        fwd_frame = fwd_results.get(frame_idx, {})
        bwd_frame = bwd_results.get(frame_idx, {})
        matches, _, _ = match_detections(fwd_frame, bwd_frame, similarity_threshold, use_mask)
        for fwd_id, bwd_id in matches.items():
            votes.setdefault(bwd_id, {})
            votes[bwd_id][fwd_id] = votes[bwd_id].get(fwd_id, 0) + 1

    bwd_to_fwd: Dict[int, int] = {
        bwd_id: max(fwd_votes, key=fwd_votes.get)
        for bwd_id, fwd_votes in votes.items()
    }

    # Frame-by-frame fusion
    bwd_only_map: Dict[int, int] = {}
    next_bwd_only_id = -1
    merged: FrameResults = {}

    for frame_idx in all_frames:
        fwd_frame = fwd_results.get(frame_idx, {})
        bwd_frame = bwd_results.get(frame_idx, {})
        merged_frame: Dict[int, dict] = {}
        seen_fwd_ids = set()

        for bwd_id, det_b in bwd_frame.items():
            if bwd_id in bwd_to_fwd:
                fwd_id = bwd_to_fwd[bwd_id]
                merged_frame[fwd_id] = merge_two_detections(fwd_frame.get(fwd_id), det_b)
                seen_fwd_ids.add(fwd_id)
            else:
                # Object seen only in backward pass
                if bwd_id not in bwd_only_map:
                    bwd_only_map[bwd_id] = next_bwd_only_id
                    next_bwd_only_id -= 1
                merged_frame[bwd_only_map[bwd_id]] = det_b

        for fwd_id, det_f in fwd_frame.items():
            if fwd_id not in seen_fwd_ids:
                merged_frame[fwd_id] = det_f

        merged[frame_idx] = merged_frame

    return merged


def merge_overlap_frame(
    prev_det: Dict[int, dict],  # global IDs — from the previous chunk
    curr_det: Dict[int, dict],  # local  IDs — from the current chunk
    similarity_threshold: float = 0.4,
    use_mask: bool = True,
) -> Tuple[Dict[int, int], Dict[int, dict], Dict[int, int]]:
    """
    Resolve the shared overlap frame between two consecutive chunks.

    The overlap frame has been produced twice:
      - once as the *last*  frame of chunk k-1  (global IDs, already finalised)
      - once as the *first* frame of chunk k    (local  IDs, to be mapped)

    This function:
      1. Matches local IDs to global IDs on the overlap frame.
      2. Merges the two detections for matched objects.
      3. Returns the local->global ID mapping to be applied to the whole chunk.

    Args:
        prev_det      : {global_id: det} — overlap frame from previous chunk
        curr_det      : {local_id:  det} — overlap frame from current chunk
        similarity_threshold : minimum similarity to accept a match

    Returns:
        local_to_global   : {local_id -> global_id} for matched objects
        merged_overlap    : {global_id: det} — merged detections on the overlap frame
        unmatched_local   : {local_id -> None} for unmatched local objects
                            (they will receive a new global ID later)
    """
    matches, unmatched_local_ids, unmatched_prev_ids = match_detections(
        curr_det,   # src_a : local IDs
        prev_det,   # src_b : global IDs
        similarity_threshold,
        use_mask,
    )
    # matches : {local_id -> global_id}

    local_to_global: Dict[int, int] = dict(matches)

    # Merge detections on the overlap frame for matched objects
    merged_overlap: Dict[int, dict] = {}

    for local_id, global_id in matches.items():
        merged_overlap[global_id] = merge_two_detections(
            curr_det[local_id], prev_det[global_id]
        )

    # Objects present in prev chunk but not matched — keep them from previous result
    for global_id in unmatched_prev_ids:
        merged_overlap[global_id] = prev_det[global_id]

    # Objects present only in current chunk on the overlap frame —
    # local_id kept as sentinel (None value); global ID assigned in step 3
    unmatched_local: Dict[int, None] = {lid: None for lid in unmatched_local_ids}

    return local_to_global, merged_overlap, unmatched_local


def assign_global_ids(
    chunk_merged: FrameResults,
    prev_overlap_detections: Dict[int, dict],
    next_global_id: int,
    similarity_threshold: float = 0.4,
    use_mask: bool = True,
) -> Tuple[FrameResults, Dict[int, dict], int]:
    """
    Replace local IDs in the merged chunk with globally consistent IDs,
    taking into account the 1-frame overlap with the previous chunk.

    The overlap frame (first frame of the current chunk == last frame of the
    previous chunk) is re-merged with the already-finalised previous result.

    Args:
        chunk_merged             : output of merge_chunk_forward_backward
        prev_overlap_detections  : {global_id: det} of the overlap frame
                                   from the previous chunk (empty for chunk 0)
        next_global_id           : next available global ID

    Returns:
        global_chunk        : {frame_idx: {global_id: det}}
        new_overlap_frame   : {global_id: det} of the last frame of this chunk
                              (to be passed as prev_overlap_detections for chunk k+1)
        next_global_id      : updated counter
    """
    all_frames = sorted(chunk_merged.keys())
    if not all_frames:
        return {}, dict(prev_overlap_detections), next_global_id
    overlap_frame_idx = all_frames[0]   # shared with previous chunk
    overlap_det_curr  = chunk_merged[overlap_frame_idx]

    # ── Resolve the overlap frame ────────────────────────────────────────────
    local_to_global, merged_overlap, unmatched_local = merge_overlap_frame(
        prev_overlap_detections,
        overlap_det_curr,
        similarity_threshold,
        use_mask,
    )

    # Assign new global IDs to unmatched local objects and insert them directly
    # into merged_overlap — no temporary IDs, no collision risk.
    for local_id in unmatched_local:
        local_to_global[local_id] = next_global_id
        merged_overlap[next_global_id] = overlap_det_curr[local_id]
        next_global_id += 1

    # ── Remap every frame of the chunk ──────────────────────────────────────
    global_chunk: FrameResults = {}

    for frame_idx in all_frames:
        if frame_idx == overlap_frame_idx:
            # Use the already-merged overlap frame
            global_chunk[frame_idx] = merged_overlap
            continue

        global_frame: Dict[int, dict] = {}
        for local_id, det in chunk_merged[frame_idx].items():
            if local_id not in local_to_global:
                # Object that appeared after the overlap frame
                local_to_global[local_id] = next_global_id
                next_global_id += 1
            global_frame[local_to_global[local_id]] = det
        global_chunk[frame_idx] = global_frame

    # Last frame of this chunk becomes the overlap frame for the next chunk
    new_overlap_frame = dict(global_chunk[all_frames[-1]])

    return global_chunk, new_overlap_frame, next_global_id


def merge_tracks(
    fwd_results: FrameResults,
    bwd_results: FrameResults,
    prev_overlap_detections: Dict[int, dict],
    next_global_id: int,
    similarity_threshold_merge: float = 0.3,
    similarity_threshold_track: float = 0.4,
    use_mask: bool = True,
) -> Tuple[FrameResults, Dict[int, dict], int]:
    """
    Process a single chunk end-to-end:
      1. Merge forward and backward results.
      2. Assign globally consistent IDs, resolving the overlap frame.
      3. Free intermediate data.

    Chunk layout (N=4, overlap=1):

        chunk k-1 : [f0  f1  f2  f3]
        chunk k   :             [f3  f4  f5  f6]
                                 ^^^ overlap frame

    Args:
        fwd_results              : {frame_idx: {local_id: det}}  — forward pass
        bwd_results              : {frame_idx: {local_id: det}}  — backward pass
        prev_overlap_detections  : {global_id: det} of the last frame of chunk k-1
                                   (empty dict for the very first chunk)
        next_global_id           : next available global ID (0 for the first chunk)

    Returns:
        global_chunk        : {frame_idx: {global_id: det}}
                              includes the overlap frame with merged detections
        new_overlap_frame   : {global_id: det} — last frame of this chunk,
                              to be passed as prev_overlap_detections for chunk k+1
        next_global_id      : updated, to be passed to the next call
    """
    # Step 1 — fuse forward and backward within the chunk
    chunk_merged = merge_chunk_forward_backward(
        fwd_results,
        bwd_results,
        similarity_threshold=similarity_threshold_merge,
        use_mask=use_mask,
    )
    fwd_results.clear()
    bwd_results.clear()

    # Step 2 — assign global IDs, handle overlap frame
    global_chunk, new_overlap_frame, next_global_id = assign_global_ids(
        chunk_merged,
        prev_overlap_detections,
        next_global_id,
        similarity_threshold=similarity_threshold_track,
        use_mask=use_mask,
    )
    chunk_merged.clear()

    return global_chunk, new_overlap_frame, next_global_id


def bond_masks(masks, colors=None, dilate_kernel_size: int = 11, conflict_kernel_size: int = 11,
              close_kernel_size: int = 11):
    """
    Unified bonding function. Resolves overlaps between multiple binary masks using EDT
    and optionally maps colors to the expanded areas without distortion.
    """
    import cv2
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    bonded_color_mask = None

    # Dilate each individual mask
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
    dilated_masks = [cv2.dilate(np.round(m).astype(np.uint8), kernel_dilate) for m in masks]
    dilated_stack = np.stack(dilated_masks, axis=0)

    # Detect overlapping regions to generate the activation mask
    overlap_map = np.sum(dilated_stack, axis=0)
    conflict_zone = (overlap_map >= 2).astype(np.uint8) * 255
    kernel_conflict = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (conflict_kernel_size, conflict_kernel_size))
    action_mask = cv2.dilate(conflict_zone, kernel_conflict)

    # Merge individual masks into a single global mask
    masks_stack = np.stack(masks, axis=0)
    global_mask_raw = (np.sum(masks_stack, axis=0) > 0).astype(np.uint8) * 255 

    # Apply morphological closing to fill holes in the global mask
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    global_mask_closed = cv2.morphologyEx(global_mask_raw, cv2.MORPH_CLOSE, kernel_close)

    # Apply the closed mask only within detected conflict regions
    bonded_binary_mask = np.where(action_mask > 0, global_mask_closed, global_mask_raw).astype(np.float32)

    # Harmonized color bonding steps
    if colors:
        height, width = masks[0].shape[:2]
        color_mask_raw = np.zeros((height, width, 3), dtype=np.float32)
        for mask, color in zip(masks, colors):
            color_mask_raw[mask > 0] = color

        # Identify new pixels created strictly by the bonding process
        # (Where the bonded binary is active, but raw binary wasn't)
        bonding_gap = (bonded_binary_mask > 0) & (global_mask_raw == 0)

        bonded_color_mask = color_mask_raw.copy()
        if np.any(bonding_gap):
            # Find coordinates of the closest original colored pixels
            _, indices = distance_transform_edt(global_mask_raw == 0, return_indices=True)
            # Map nearest colors directly into the gap
            bonded_color_mask[bonding_gap] = color_mask_raw[indices[0][bonding_gap], indices[1][bonding_gap]]

    return bonded_binary_mask, bonded_color_mask
