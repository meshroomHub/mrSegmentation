__version__ = "1.0"

import os

from meshroom.core import desc
import json
from inspect import getmembers, isfunction


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
                h, w, _ = masks[j].kernel_shape
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
        median_mask = metric(window_masks, axis=0).astype(np.uint8)
        consistent_masks.append(median_mask)

    return consistent_masks

def open_image(filename):
    """
    Opens an image with oiio
    """
    inp = oiio.ImageInput.open(filename)
    if inp :
        spec = inp.spec()
        xres = spec.width
        yres = spec.height
        nchannels = spec.nchannels
        pixels = inp.read_image(0, 0, 0, nchannels, "uint8")
        inp.close()
        return np.asarray(pixels)
    else:
        raise RuntimeError("Could not open file "+filename)

class Operations():
    """
    Collections of functions that takes as input at least the masks
    and return the masks.
    Add your own function here, it should update the ui automatically.
    """
    #dummy
    def identity(masks, **kwargs):
        """
        Doens nothing
        """
        return masks

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

    def invertion(masks, **kwargs):
        '''
        Invert a list of masks.
        '''
        return [1-mask for mask in masks]  

class FilterMasks(desc.Node):

    category = 'Utils'
    documentation = '''Apply selected operation to a set of input masks'''

    inputs = [
        desc.File(
            name='maskFolder',
            label='Mask Folder',
            description='maskFolder',
            value='Folder containing the masks to filter',
        ),

        desc.File(
            name='inputSfM',
            label='SfMData',
            description='SfMData file.',
            value='',
        ),

        desc.ChoiceParam(
            name='filterType',
            label='Filter Type',
            description='''Type of filtering to apply''',
            value='identity',
            values=[f[0]for f in getmembers(Operations, isfunction)],
            exclusive=True,
        ),

        desc.BoolParam(
            name='keepFilename',
            label='keepFilename',
            description='''Will save the masks keeping the original image name''',
            value=True
        ),

        desc.ChoiceParam(
            name='extension',
            label='Input/Output File Extension',
            description='Input/Output image file extension.',
            value='exr',
            values=['exr', 'png', 'jpg'],
            exclusive=True,
        ),
        
        #options
        desc.IntParam(
            name='kernel_size',
            label='Filter size',
            description='For spatial filtering, the size of the kernel. For temporal filtering, the size of the window',
            value=3,
            range=(3, 100000, 1),
            group="opt",
            enabled=lambda node:node.filterType.value in ['erosion', 'dilation', 'opening', 'closing', 'blur', 'temporal_filtering', 'guided_filtering']
        ),

        desc.ChoiceParam(
            name='kernel_shape',
            label='Filter kernel shape',
            description='For spatial filtering, the kernel shape of the kernel',
            value='rect',
            values=list(SHAPES.keys()),
            exclusive=True,
            group="opt",
            enabled=lambda node:node.filterType.value in ['erosion', 'dilation', 'opening', 'closing']
        ),

        desc.IntParam(
            name='iterations',
            label='Iterations',
            description='Number of time to perform the filtering',
            value=1,
            range=(1, 100000, 1),
            group="opt",
            enabled=lambda node:node.filterType.value in ['erosion', 'dilation']
        ),

        desc.BoolParam(
            name='use_of',
            label='Use optical flow',
            description='Will warp the images using optical flow',
            value=False,
            group="opt",
            enabled=lambda node:node.filterType.value =='temporal_filtering'
        ),

        desc.FloatParam(
            name='smoothing_strength',
            label='Smoothing Strength',
            description='Stength of the smoothing',
            value=0.05,
            range=(0.0, 0.05, 1.0),
            group="opt",
            enabled=lambda node:node.filterType.value =='guided_filtering',
        ),

        desc.ChoiceParam(
                name='verboseLevel',
                label='Verbose Level',
                description='''verbosity level (fatal, error, warning, info, debug, trace).''',
                value='info',
                values=['fatal', 'error', 'warning', 'info', 'debug', 'trace'],
                exclusive=True,
            ),

    ]

    outputs = [
        desc.File(
            name='outputFolder',
            label='outputFolder',
            description='outputFolder',
            value=desc.Node.internalFolder,
            group='',
        ),
        desc.File(
            name='masks',
            label='Masks',
            description='Generated segmentation masks.',
            semantic='image',
            value=lambda attr: desc.Node.internalFolder + '<VIEW_ID>.'+attr.node.extension.value,
            group='',
            visible=False
        ),
    ]

    def processChunk(self, chunk):
        """
        Appply the filters
        """
        chunk.logManager.start(chunk.node.verboseLevel.value)
        if chunk.node.inputSfM.value == '':
            raise RuntimeError('No inputSfM specified')
        if chunk.node.maskFolder.value == '':
            raise RuntimeError('No maskFolder specified')
        
        #loading and temporal sort
        chunk.logger.info('Loading masks')
        sfm_data=json.load(open(chunk.node.inputSfM.value,'r'))
        sfm_data['views']=sorted(sfm_data['views'], key=lambda v:int(v['frameId'])) 

        #opening images/masks
        images=[]
        masks=[]
        for view in sfm_data['views']:
            if chunk.node.keepFilename.value:
                image_basename = os.path.splitext(os.path.basename(view['path']))[0]
            else:
                image_basename = view['viewId']
            mask_file = os.path.join(chunk.node.maskFolder.value, image_basename+'.'+chunk.node.extension.value)
            if not os.path.exists(mask_file):
                raise FileNotFoundError(mask_file+" not found.")
            chunk.logger.info('Opening '+view['path'])
            
            images.append(open_image(view['path']))
            chunk.logger.info('Opening '+mask_file)
            masks.append(open_image(mask_file)/255.0)
            print((open_image(mask_file).shape))
            

        #filter
        chunk.logger.info('Applying filter')
        filter_function = eval('Operations.'+chunk.node.filterType.value)
        
        kargs={}
        for a in chunk.node.attributes:
            if a.attributeDesc.group=='opt':
                kargs[a.name]=a.value 
        chunk.logger.info(kargs)
        filtered_masks = filter_function(masks,images=images,**kargs)
            
        #saving
        chunk.logger.info('Saving masks')
        for view, mask in zip(sfm_data['views'], filtered_masks):
            if chunk.node.keepFilename.value:
                image_basename = os.path.splitext(os.path.basename(view['path']))[0]
            else:
                image_basename = view["viewId"]
            filename = os.path.join(chunk.node.outputFolder.value,
                                    image_basename+'.'+chunk.node.extension.value)
            out = oiio.ImageOutput.create(filename)
            if not out:
                raise RuntimeError("Could no create "+filename)
            
            spec = oiio.ImageSpec(mask.shape[1], mask.shape[0], 1, 'uint8')
            out.open(filename, spec)
            mask_uint=(255*mask).astype(np.uint8)
            if len(mask_uint.shape)<3:
                mask_uint=np.expand_dims(mask_uint, axis=-1)
            out.write_image(mask_uint)
            out.close()

        chunk.logManager.end()

