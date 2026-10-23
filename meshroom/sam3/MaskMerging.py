__version__ = "1.0"

import logging
import os
from pathlib import Path

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

logger = logging.getLogger("MaskMerging")

class MaskMerging(desc.Node):
    """
## Mask Merging

Merges the masks of all the secondary prompts into the instances of the first prompt they are associated with.
The result is stored in a new prompt '<firstPrompt>-Full'.

- **Output Content**: 'all' outputs the original prompts along with '<firstPrompt>-Full',
  'mergedOnly' outputs only '<firstPrompt>-Full' (color masks and bboxes.json).
- **Merge Instances**: after the merging of the secondary prompts, all the instances of '<firstPrompt>-Full'
  are merged into a single instance (id 0).
"""
    category = "Segmentation"

    inputs = [
        desc.File(
            name="input",
            label="Input",
            description="SfMData file.",
            value="",
        ),
        desc.File(
            name="bboxesFileIn",
            label="Bounding Boxes File In",
            description="File containing mask bounding boxes (.json).",
            value="",
        ),
        desc.File(
            name="masksFolder",
            label="Masks Folder",
            description="Folder containing original masks.",
            value="",
        ),
        desc.IntParam(
            name="itemBondingKernelSize",
            label="Inter Item Bonding Kernel Size",
            description="Kernel size for morphological processing applied for masks bonding between items.",
            value=11,
            range=(1, 255, 2),
        ),
        desc.ChoiceParam(
            name="outputContent",
            label="Output Content",
            description="Content of the outputs:\n"
                        " - all: original prompts and the merged prompt '<firstPrompt>-Full'.\n"
                        " - mergedOnly: only the merged prompt '<firstPrompt>-Full'.",
            value="all",
            values=["all", "mergedOnly"],
            exclusive=True,
        ),
        desc.BoolParam(
            name="mergeInstances",
            label="Merge Instances",
            description="Merge all the instances of the merged prompt '<firstPrompt>-Full' into a single instance.",
            value=False,
        ),
        desc.IntParam(
            name="instanceBondingKernelSize",
            label="Inter Instance Bonding Kernel Size",
            description="Kernel size for morphological processing applied for masks bonding between instances of a same item.",
            value=11,
            range=(1, 255, 2),
            enabled=lambda node: node.mergeInstances.value,
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug).",
            value="info",
            values=VERBOSE_LEVEL,
            exclusive=True,
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Masks Folder",
            description="Output path for the masks.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="bboxesFileOut",
            label="Bounding Boxes File Out",
            description="Output json file containing mask bounding boxes.",
            value="{nodeCacheFolder}/bboxes.json",
        ),
        desc.File(
            name="colorMasks",
            label="Colored Masks",
            description="Colored segmentation masks of the new prompt '<firstPrompt>-Full' resulting from the iterative merging of the first prompt with all the others.",
            semantic="image",
            value=None,
        ),
    ]

    def resolve_paths(self, node, main_prompt):

        if os.path.isdir(os.path.join(node.masksFolder.value, "colorMasks", "merged")):
            color_mask_folder_out = os.path.join(node.output.value, "colorMasks", "merged")
            color_mask_filename = f"colorMask_{main_prompt}_merged_<FILESTEM>.exr"
        else:
            color_mask_folder_out = os.path.join(node.output.value, "colorMasks", "fwd")
            color_mask_filename = f"colorMask_{main_prompt}_fwd_<FILESTEM>.exr"

        node.colorMasks.value = os.path.join(color_mask_folder_out, color_mask_filename)


    def processChunk(self, chunk):
        import shutil
        import json
        import numpy as np
        from segmentationRDS import image, bboxUtils, sam3Utils
        from segmentationRDS.mask_label_manager import BBoxesManager

        logger.setLevel(chunk.node.verboseLevel.value.upper())

        if os.path.isdir(os.path.join(chunk.node.masksFolder.value, "colorMasks", "merged")):
            src = os.path.join(chunk.node.masksFolder.value, "colorMasks", "merged")
            color_masks_folder = os.path.join(chunk.node.output.value, "colorMasks", "merged")
            mask_type = "merged"
        elif os.path.isdir(os.path.join(chunk.node.masksFolder.value, "colorMasks", "fwd")):
            src = os.path.join(chunk.node.masksFolder.value, "colorMasks", "fwd")
            color_masks_folder = os.path.join(chunk.node.output.value, "colorMasks", "fwd")
            mask_type = "fwd"
        else:
            raise ValueError("No color mask folder")
        if not os.path.isfile(chunk.node.bboxesFileIn.value):
            raise ValueError("No bboxes.json file")

        merged_only = chunk.node.outputContent.value == "mergedOnly"
        merge_instances = chunk.node.mergeInstances.value

        # Input masks are never modified: expose them through symbolic links
        os.makedirs(color_masks_folder, exist_ok=True)
        if not merged_only:
            for filename in os.listdir(src):
                os.symlink(os.path.abspath(os.path.join(src, filename)), os.path.join(color_masks_folder, filename))
        shutil.copyfile(chunk.node.bboxesFileIn.value, chunk.node.bboxesFileOut.value)

        with open(chunk.node.bboxesFileIn.value, "r") as f:
            bboxes = json.load(f)
        prompts = list(bboxes.keys())

        if len(prompts) == 1 and not merged_only and not merge_instances:
            self.resolve_paths(chunk.node, prompts[0])
            return

        # Merging result is stored in a new prompt, initialized from the first prompt
        # and iteratively merged with all the other prompts
        new_prompt = f"{prompts[0]}-Full"

        # New prompt entry created in bboxes.json at start as a copy of the first prompt, which stays untouched
        bboxes_manager = BBoxesManager(chunk.node.bboxesFileOut.value, new_prompt)
        bboxes_manager.create_label_from(prompts[0])
        bboxes_manager.save()

        self.resolve_paths(chunk.node, new_prompt)

        frames = get_frame_pathes(chunk.node.input.value)
        main_prefix = f"colorMask_{prompts[0]}_{mask_type}_"
        color_palette = image.paletteGenerator()
        ks = chunk.node.itemBondingKernelSize.value

        written_paths = set()

        for second_prompt in prompts[1:]:
            data = {
                # Same tracking key rule as the new prompt: 'merged' if not empty, otherwise 'forward'
                second_prompt: bboxes[second_prompt].get(BBoxesManager.tracking_key_of(bboxes[second_prompt]), {}),
                new_prompt: bboxes_manager.get_tracking_data(),
            }

            mapping = bboxUtils.associate_items(data, second_prompt, new_prompt)

            logger.info(f"Mapping {second_prompt} => {new_prompt} = {mapping}")

            # {frame: {main item id: (main color, [second color, ...])}}
            # Several secondary items can be associated to the same main item
            merges_by_frame = {}
            for s_id, p_id in mapping.items():
                color_palette.generate_palette(max(int(s_id), int(p_id["item2_id"])) + 1)
                second_color = np.array(color_palette.at(int(s_id)), dtype=np.float32) / 255.0
                main_color = np.array(color_palette.at(int(p_id["item2_id"])), dtype=np.float32) / 255.0
                logger.info(f"{s_id} {second_color} <=> {p_id['item2_id']} {main_color} with score {p_id['confidence_score']}")

                frame_chunks = []
                frame_chunk = []
                prev_fid = -1
                for fid, pids in data[new_prompt].items():
                    if p_id["item2_id"] in pids and fid in data[second_prompt] and s_id in data[second_prompt][fid]:
                        # A chunk only holds consecutive frames passing the test
                        if frame_chunk and int(fid) != prev_fid + 1:
                            frame_chunks.append(frame_chunk)
                            frame_chunk = []
                        frame_chunk.append(int(fid))
                        prev_fid = int(fid)
                if len(frame_chunk):
                    frame_chunks.append(frame_chunk)

                for frame_chunk in frame_chunks:
                    for fid in frame_chunk:
                        frame_merges = merges_by_frame.setdefault(fid, {})
                        frame_merges.setdefault(p_id["item2_id"], (main_color, []))[1].append(second_color)

            # The mapping being fixed for this secondary prompt, all its merges in a frame are applied in memory
            # so that each mask is read and written only once
            for fid in sorted(merges_by_frame):
                frame_filename = get_mask_suffix(src, main_prefix, frames[fid])
                new_color_masks_path = os.path.join(color_masks_folder, f"colorMask_{new_prompt}_{mask_type}_{frame_filename}")
                # Accumulate merges: read the first prompt mask until the new prompt mask has been written
                if new_color_masks_path in written_paths:
                    main_color_masks_path = new_color_masks_path
                else:
                    main_color_masks_path = os.path.join(src, f"colorMask_{prompts[0]}_{mask_type}_{frame_filename}")
                main_color_mask, _, _, nc, par = load_exr_raw(main_color_masks_path)
                second_mask_filename = f"colorMask_{second_prompt}_{mask_type}_{frame_filename}"
                second_color_masks_path = os.path.join(src, second_mask_filename)
                second_color_mask, _, _, _, _ = load_exr_raw(second_color_masks_path)

                for main_id, (main_color, second_colors) in merges_by_frame[fid].items():
                    # The main item is bonded at once with all its secondary items
                    masks = [compute_color_mask_alpha(main_color, main_color_mask)]
                    masks += [compute_color_mask_alpha(second_color, second_color_mask) for second_color in second_colors]

                    bounded_mask, _ = sam3Utils.bond_masks(
                        masks=masks,
                        dilate_kernel_size=ks,
                        conflict_kernel_size=ks,
                        close_kernel_size=ks
                    )

                    paint_mask(main_color_mask, bounded_mask, main_color)
                    bboxes_manager.update_from_mask(fid, main_id, (bounded_mask > 0.5).astype(np.uint8) * 255)

                save_exr(new_color_masks_path, main_color_mask, nc, par)
                written_paths.add(new_color_masks_path)

        if merge_instances:
            # All the instances of the new prompt are merged into a single instance of id 0
            iks = chunk.node.instanceBondingKernelSize.value
            tracking_data = bboxes_manager.get_tracking_data()
            for fid in list(tracking_data.keys()):
                obj_ids = list(tracking_data[fid].keys())
                if not obj_ids:
                    continue
                frame_filename = get_mask_suffix(src, main_prefix, frames[int(fid)])
                new_color_masks_path = os.path.join(color_masks_folder, f"colorMask_{new_prompt}_{mask_type}_{frame_filename}")
                if new_color_masks_path in written_paths:
                    main_color_masks_path = new_color_masks_path
                else:
                    main_color_masks_path = os.path.join(src, f"colorMask_{prompts[0]}_{mask_type}_{frame_filename}")
                main_color_mask, _, _, nc, par = load_exr_raw(main_color_masks_path)

                color_palette.generate_palette(max(int(obj_id) for obj_id in obj_ids) + 1)
                masks = [
                    compute_color_mask_alpha(np.array(color_palette.at(int(obj_id)), dtype=np.float32) / 255.0,
                                             main_color_mask)
                    for obj_id in obj_ids
                ]

                if len(masks) > 1:
                    bounded_mask, _ = sam3Utils.bond_masks(
                        masks=masks,
                        dilate_kernel_size=iks,
                        conflict_kernel_size=iks,
                        close_kernel_size=iks
                    )
                else:
                    bounded_mask = 255.0 * masks[0]

                instance_color = np.array(color_palette.at(0), dtype=np.float32) / 255.0
                paint_mask(main_color_mask, bounded_mask, instance_color, keep_other_colors=False)
                save_exr(new_color_masks_path, main_color_mask, nc, par)
                written_paths.add(new_color_masks_path)
                bboxes_manager.replace_frame_with_mask(int(fid), 0, (bounded_mask > 0.5).astype(np.uint8) * 255)

        if merged_only:
            bboxes_manager.keep_only_label()
        bboxes_manager.save()

        # Complete the new prompt sequence with links to the first prompt masks for the frames without merging
        for filename in os.listdir(src):
            if filename.startswith(main_prefix):
                new_color_masks_path = os.path.join(color_masks_folder, f"colorMask_{new_prompt}_{mask_type}_{filename[len(main_prefix):]}")
                if new_color_masks_path not in written_paths:
                    os.symlink(os.path.abspath(os.path.join(src, filename)), new_color_masks_path)


def get_frame_pathes(input_path):
    from pyalicevision import sfmData, camera
    from pyalicevision import sfmDataIO
    from segmentationRDS.sfmData import isSfmDataFile

    frames = {}
    if isSfmDataFile(Path(input_path)):
        if Path(input_path).exists():
            dataAV = sfmData.SfMData()
            if sfmDataIO.load(dataAV, input_path, sfmDataIO.ALL):
                views = dataAV.getViews()
                commonParams = None
                for view_id, v in views.items():
                    image_path = Path(v.getImage().getImagePath())
                    frame_id = v.getFrameId()
                    intrinsic = dataAV.getIntrinsicSharedPtr(v.getIntrinsicId())
                    pinhole = camera.Pinhole.cast(intrinsic)
                    par = 1.0
                    if pinhole is not None:
                        par = pinhole.getPixelAspectRatio()
                    sequence_folder = os.path.dirname(str(image_path))
                    frames[frame_id] = (str(image_path), str(view_id))
                    params = [sequence_folder, v.getImage().getWidth(), v.getImage().getHeight(), par]
                    if commonParams is None:
                        commonParams = params
                    if commonParams != params:
                        raise ValueError(f"All images do not have same dimensions or are not located in the same folder: {params} vs {commonParams}")
        else:
            raise ValueError(f"Input path '{input_path}' does not exist.")
    else:
        raise ValueError(f"Input path '{input_path}' is not a valid sfmData file.")
    return frames


def get_mask_suffix(mask_folder, mask_prefix, frame):
    """
    Return the filename suffix of the color mask of a frame, named as in VideoSegmentationSam3Text:
    '<image stem>.exr' when the input filenames are kept, '<view id>.exr' otherwise.
    """
    image_path, view_id = frame
    suffix = Path(image_path).stem + ".exr"
    if os.path.exists(os.path.join(mask_folder, mask_prefix + suffix)):
        return suffix
    return f"{view_id}.exr"


def load_exr_raw(filepath):
    import OpenImageIO as oiio
    import numpy as np

    inp = oiio.ImageInput.open(filepath)
    if not inp:
        raise IOError(f"Unable to open {filepath}: {oiio.geterror()}")

    spec = inp.spec()
    width = spec.width
    height = spec.height
    nchannels = spec.nchannels
    par = spec.get_float_attribute("PixelAspectRatio", 1.0)

    pixels = inp.read_image()
    inp.close()

    if pixels is None:
        raise IOError(f"Unable to read the pixels of {filepath}")

    arr = np.array(pixels, dtype=np.float32).reshape(height, width, nchannels)
    return arr, width, height, nchannels, par

def compute_color_mask_alpha(color, mask_raw):
    """
    Compute a binary alpha of the pixels matching the given color.
    Palette colors are never black (grey colors are excluded), so a match also means the pixel belongs to a mask.
    """
    import numpy as np
    rgb = mask_raw[:, :, :3]
    target = np.array(color, dtype=np.float32)
    diff = rgb - target[np.newaxis, np.newaxis, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return (dist <= 0.01).astype(np.float32)

def paint_mask(mask_raw, mask, target, keep_other_colors=True):
    """
    In place, replace all the pixels of the target color of a color mask (H,W,C) with the new masked area (mask),
    keeping the other colors/masks already present unless keep_other_colors is False.
    """
    import numpy as np

    # 1. Clear the previous area of this color, or all the colors (reset to black)
    if keep_other_colors:
        rgb = mask_raw[:, :, :3]
        diff = rgb - target[np.newaxis, np.newaxis, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        old_zone = dist <= 0.01
        mask_raw[old_zone, :3] = 0.0
    else:
        mask_raw[:, :, :3] = 0.0

    # 2. Apply the target color on the new area (mask > 0.5)
    new_zone = mask > 0.5
    mask_raw[new_zone, 0] = target[0]
    mask_raw[new_zone, 1] = target[1]
    mask_raw[new_zone, 2] = target[2]

def save_exr(filepath, arr, nchannels, par):
    """Save a float (H,W,C) array as EXR using OpenImageIO"""
    import OpenImageIO as oiio

    h, w = arr.shape[0], arr.shape[1]
    out = oiio.ImageOutput.create(filepath)
    if not out:
        raise IOError(f"Unable to create the output file {filepath}: {oiio.geterror()}")

    spec = oiio.ImageSpec(w, h, nchannels, oiio.FLOAT)
    spec.attribute("PixelAspectRatio", float(par))
    out.open(filepath, spec)
    out.write_image(arr)
    out.close()
