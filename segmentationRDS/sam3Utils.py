import numpy as np

def xywhNorm2xyxy(xywhNorm, width: int, height: int):
    # Convert a normalized rectangle (x_topLeft, y_topLeft, width, height) with all values in range [0, 1]
    # into a rectangle (x_topLeft, y_topLeft, x_bottomRight, y_bottomRight) with x values in range [0, width] and y values in range [0, height]
    xn, yn, wn, hn = xywhNorm
    x = int(xn * width)
    y = int(yn * height)
    w = int(wn * width)
    h = int(hn * height)
    return (x, y, x+w, y+h)

def updateSam3ObjectIds(outputs_per_frame, mapping):
    # Deep copy a sam3 output coming from a propagation on video
    # Updates the object ids using mapping and return the result
    import copy
    import numpy as np
    outputs_per_frame_updated = copy.deepcopy(outputs_per_frame)
    for f, out in outputs_per_frame.items():
        new_ids = np.zeros_like(out["out_obj_ids"])
        for i, id in enumerate(out["out_obj_ids"]):
            if id in mapping.keys():
                new_ids[i] = mapping[id]
            else:
                new_ids[i] = id
        outputs_per_frame_updated[f]["out_obj_ids"] = new_ids
    return outputs_per_frame_updated

def boxInterOverSmallest(xywh1, xywh2):
    # return the ratio between the intersection area and the area of the smallest box
    x1, y1, w1, h1 = xywh1
    x2, y2, w2, h2 = xywh2
    area1 = w1 * h1
    area2 = w2 * h2
    if area1 == 0 or area2 == 0:
        return 0
    x_inter_min = max(x1, x2)
    y_inter_min = max(y1, y2)
    x_inter_max = min(x1 + w1, x2 + w2)
    y_inter_max = min(y1 + h1, y2 + h2)
    inter_w = max(0, x_inter_max - x_inter_min)
    inter_h = max(0, y_inter_max - y_inter_min)
    inter_area = inter_w * inter_h
    smallest_area = min(area1, area2)
    return (inter_area / smallest_area, (x_inter_min, y_inter_min, x_inter_max, y_inter_max))

def prepareMasksForVisualization(frame_to_output):
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

def displayAt(outputs_per_frame, keyFrameIdx, frameId, imgWidth, imageHeight, logger):
    logger.debug(f"frame {frameId} from key frame {keyFrameIdx}:")
    for i, obj_ids in enumerate(outputs_per_frame[keyFrameIdx][frameId]["out_obj_ids"].tolist()):
        box = xywhNorm2xyxy(outputs_per_frame[keyFrameIdx][frameId]["out_boxes_xywh"][i], imgWidth, imageHeight)
        prob = outputs_per_frame[keyFrameIdx][frameId]["out_probs"][i]
        logger.debug(f"{obj_ids}: {box} ; {prob}")

def mapIds(outputs_per_frame_detected, outputs_per_frame_propagated, w, h, logger):
    import numpy as np

    ids = outputs_per_frame_detected["out_obj_ids"]
    masks = outputs_per_frame_detected["out_binary_masks"]
    boxes = outputs_per_frame_detected["out_boxes_xywh"]
    idsRef = outputs_per_frame_propagated["out_obj_ids"]
    masksRef = outputs_per_frame_propagated["out_binary_masks"]
    boxesRef = outputs_per_frame_propagated["out_boxes_xywh"]

    logger.debug(f"{len(ids)}")
    logger.debug(f"{ids}")
    for k, id in enumerate(ids.tolist()):
        logger.debug(f"{id} ; {boxes[k]}")

    mapping = {}
    for d, id_det in enumerate(ids.tolist()):
        obj_id_min = -1
        iou_max = 0
        ios_best = 0
        for p, id_prop in enumerate(idsRef.tolist()):
            inter_over_smallest, box_inter = boxInterOverSmallest(boxes[d], boxesRef[p])
            logger.debug(f"{id_det} vs {id_prop}: ios = {inter_over_smallest} ({ios_best})")
            if  inter_over_smallest > 0.9:
                x1, y1, x2, y2 = box_inter
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)
                crop = masks[d][y1:y2, x1:x2]
                cropRef = masksRef[p][y1:y2, x1:x2]
                inter = np.logical_and(crop,cropRef).sum()
                union = np.logical_or(crop,cropRef).sum()
                iou = 0 if union == 0 else inter / union
                logger.debug(f"{id_det} vs {id_prop}: iou = {iou} ({iou_max})")
                if iou > iou_max:
                    iou_max = iou
                    if iou > 0.7:
                        obj_id_min = id_prop
                        ios_best = inter_over_smallest
        mapping[id_det] = obj_id_min

    if -1 in mapping.values():
        for idSrc in mapping:
            if mapping[idSrc] == -1:
                if idSrc in mapping.values():
                    mapping[idSrc] = max(mapping.values()) + 1
    return mapping
