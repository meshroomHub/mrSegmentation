import json
from dataclasses import dataclass, field

THRESHOLDS = [252, 504, 1008]

@dataclass
class TrackChunk:
    """A chunk of consecutive frames with their boxes."""
    start_frame : int
    end_frame   : int
    boxes       : dict = field(default_factory=dict)  # {frame_idx: [x1, y1, x2, y2]}

    def __repr__(self):
        return f"TrackChunk(frames={self.start_frame}-{self.end_frame}, n={len(self.boxes)})"


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
    Merge 2 boxes xyxy by taking the bounding boxe, if their IoU is higher than the threshold. 
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
        return box1, f"forward     (IoU={iou:.2f} < seuil={iou_threshold})"


def get_target_size(boxes: dict, par: float):
    """
    Compute the target size for a set of boxes.
    Comparisons occur in the display space (with pixel aspect ratio applied)).
    """
    max_size = 0
    for box in boxes.values():
        display_box = box_to_display(box, par)
        x1, y1, x2, y2 = display_box
        w = x2 - x1
        h = y2 - y1
        max_size = max(max_size, w, h)

    for threshold in THRESHOLDS:
        if max_size < threshold:
            return threshold

    return max_size


def expand_box(box: list, target_size: int, par: float, frame_w: int, frame_h: int) -> list:
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
    new_x1 = cx - target_size / 2
    new_y1 = cy - target_size / 2
    new_x2 = cx + target_size / 2
    new_y2 = cy + target_size / 2

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

    # 3. Back conversion to source space
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


def extract_tracking(
    json_path     : str,
    frame_w       : int,
    frame_h       : int,
    x2_ok         : bool = True,
    x4_ok         : bool = True,
    par           : float = 1.0,
    iou_threshold : float = 0.5
) -> dict:
    """
    Extract bounding boxes per object and organize them in chunck of consecutive frames.
    Coordinates in the json file are supposed to be in the original source space, with the pixel aspect ratio not applicated.
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
        backward = directions.get("backward", {})

        all_object_ids = set()
        for frame_data in forward.values():
            all_object_ids.update(frame_data.keys())

        for obj_id in sorted(all_object_ids, key=int):
            key = f"{label}_{obj_id}"
            raw_boxes = {}

            all_frames = sorted(
                set(forward.keys()) | set(backward.keys()),
                key=int
            )

            # Merge forward/backward (in source space) ---
            for frame_idx in all_frames:
                fwd_box = forward.get(frame_idx,  {}).get(obj_id)
                bwd_box = backward.get(frame_idx, {}).get(obj_id)

                if fwd_box and bwd_box:
                    box, _ = merge_boxes(fwd_box, bwd_box, iou_threshold)
                elif fwd_box:
                    box = fwd_box
                elif bwd_box:
                    box = bwd_box
                else:
                    continue

                raw_boxes[int(frame_idx)] = box

            # --- compute target size ---
            target_size = get_target_size(raw_boxes, par)

            # --- Expand boxes in display space ---
            expanded_boxes = {}
            for frame_idx, box in raw_boxes.items():
                if target_size is not None:
                    if target_size < 504 and not x4_ok:
                        target_size = 504
                    if target_size < 1008 and not x2_ok:
                        target_size = 1008
                    expanded = expand_box(box, target_size, par, frame_w, frame_h)
                    expanded_boxes[frame_idx] = expanded
                else:
                    expanded_boxes[frame_idx] = box

            # --- Split in chunks ---
            chunks = split_into_chunks(expanded_boxes)

            result[key] = chunks

    return result


