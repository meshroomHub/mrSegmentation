import os
import json

import OpenImageIO as oiio
import numpy as np
import cv2

SHAPES = {'rect':cv2.MORPH_RECT, 'ellipse':cv2.MORPH_ELLIPSE, 'cross':cv2.MORPH_CROSS}

#NOTE: could move in operations
def _temporal_filtering(frames, masks, window_size=3, use_of=True, metric=np.median):
    """
    Ensure temporal consistency of segmentation masks across video frames using a sliding window and median filtering.
    Optionally uses optical flow.
    """

    if len(frames) != len(masks):
        raise ValueError("The number of frames and masks must be the same.")

    n_frames = len(frames)
    consistent_masks = []

    for i in range(n_frames):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(n_frames, i + window_size // 2 + 1)
    
        window_masks = []
        for j in range(start_idx, end_idx):
            if use_of:
                # OF calculation from j to i
                prev_gray = cv2.cvtColor(frames[j], cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                                    pyr_scale=0.5, levels=3, winsize=15,
                                                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                # warp the masks  j to  i 
                h, w, _ = masks[j].shape
                grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = grid_x + flow[..., 0]
                map_y = grid_y + flow[..., 1]
                warped_mask = cv2.remap(masks[j].astype(np.float32), map_x.astype(np.float32), 
                                        map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                # threshold needed because of interpolation
                warped_mask = (warped_mask > 0.5).astype(np.uint8)
                window_masks.append(warped_mask)
            else:
                window_masks.append(masks[j])

        # median of the masks in the window = final mask 
        #median_mask = metric(window_masks, axis=0).astype(np.uint8)
        median_mask = metric(window_masks, axis=0).astype(np.float32)
        consistent_masks.append(median_mask)

    return consistent_masks

class Operations():
    """
    Collections of functions that takes as input at least the masks
    and return the masks.
    Add your own function here, it should update the ui automatically.
    """
    #temporal filtering
    def temporal_filtering(masks, images, kernel_size, use_of, **kwargs):
        """
        Ensure temporal consistency of segmentation masks across video frames using a sliding window and median filtering.
        Optical flow is used to warp the images optionally.
        """
        return _temporal_filtering(images, masks, window_size=kernel_size, use_of=use_of)

    def guided_filtering(masks, images, kernel_size, smoothing_strength, **kwargs):
        """
        Apply guided filtering to a list of masks and their corresponding RGB frames using OpenCV's guidedFilter.
        """
        import numpy as np

        filtered_masks = []
    
        for mask, frame in zip(masks, images):
            # guided filter
            filtered = cv2.ximgproc.guidedFilter(
                guide=frame,
                src=mask.astype(np.float32),
                radius=kernel_size,
                eps=smoothing_strength
            )
            filtered_masks.append(filtered)
    
        return filtered_masks

    #morphology
    def dilation(masks, kernel_size=5, kernel_shape='rect', iterations=1, **kwargs):
        """
        Apply dilation to a list of masks.
        """
        kernel = cv2.getStructuringElement(SHAPES[kernel_shape], (kernel_size, kernel_size))   
        dilated_masks = [cv2.dilate(mask, kernel, iterations=iterations) for mask in masks] 
        return dilated_masks

    def erosion(masks, kernel_size=5, kernel_shape='rect', iterations=1, **kwargs):
        """
        Apply erosion to a list of masks.
        """
        kernel = cv2.getStructuringElement(SHAPES[kernel_shape], (kernel_size, kernel_size))   
        dilated_masks = [cv2.erode(mask, kernel, iterations=iterations) for mask in masks]  
        return dilated_masks

    def opening(masks,  kernel_size=5, kernel_shape='rect', **kwargs):
        """
        Apply opening to a list of masks.
        """ 
        kernel = cv2.getStructuringElement(SHAPES[kernel_shape], (kernel_size, kernel_size))   
        dilated_masks = [cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) for mask in masks]  
        return dilated_masks

    def closing(masks, kernel_size=5, kernel_shape='rect', **kwargs):
        """
        Apply closing to a list of masks.
        """
        kernel = cv2.getStructuringElement(SHAPES[kernel_shape], (kernel_size, kernel_size))   
        dilated_masks = [cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) for mask in masks]  
        return dilated_masks

    #other
    def blur(masks, kernel_size=5, **kwargs):
        '''
        Blur a list of masks.
        '''
        blurred_masks = [cv2.blur(mask, (kernel_size, kernel_size)) for mask in masks]
        return blurred_masks 

    def inversion(masks, **kwargs):
        '''
        Invert a list of masks.
        '''
        return [1-mask for mask in masks]  

