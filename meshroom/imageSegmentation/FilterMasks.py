__version__ = "1.0"

import os

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

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
            value='temporal_filtering',
            values=['temporal_filtering', 'erosion', 'dilation', 'opening', 'closing', 'blur', 'guided_filtering', 'inversion'],
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
            values=['rect', 'ellipse', 'cross'],
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
                values=VERBOSE_LEVEL,
                exclusive=True,
            ),

    ]

    outputs = [
        desc.File(
            name='outputFolder',
            label='outputFolder',
            description='outputFolder',
            value="{nodeCacheFolder}",
            group='',
        ),
        desc.File(
            name='masks',
            label='Masks',
            description='Generated segmentation masks.',
            semantic='image',
            value=lambda attr: "{nodeCacheFolder}/" + ("<FILESTEM>" if attr.node.keepFilename.value else "<VIEW_ID>") + "." + attr.node.extension.value,
            group='',
            visible=False
        ),
    ]

    def processChunk(self, chunk):
        """
        Apply the filters
        """
        import json
        import numpy as np
        import OpenImageIO as oiio
        from segmentationRDS import filtering, image

        chunk.logManager.start(chunk.node.verboseLevel.value)
        if chunk.node.inputSfM.value == '':
            error = 'No inputSfM specified'
            chunk.logger.error(error)
            raise RuntimeError(error)
        if chunk.node.maskFolder.value == '':
            error = 'No maskFolder specified'
            chunk.logger.error(error)
            raise RuntimeError(error)
        
        #loading and temporal sort
        chunk.logger.info('Loading masks')
        sfm_data=json.load(open(chunk.node.inputSfM.value,'r'))
        sfm_data['views']=sorted(sfm_data['views'], key=lambda v:int(v['frameId']))

        #opening images/masks
        images=[]
        masks=[]
        metas=[]
        for view in sfm_data['views']:
            if chunk.node.keepFilename.value:
                image_basename = os.path.splitext(os.path.basename(view['path']))[0]
            else:
                image_basename = view['viewId']
            mask_file = os.path.join(chunk.node.maskFolder.value, image_basename+'.'+chunk.node.extension.value)
            if not os.path.exists(mask_file):
                error = mask_file+" not found."
                chunk.logger.error(error)
                raise FileNotFoundError(error)
            
            chunk.logger.info('Opening '+view['path'])
            img, h_ori, w_ori, PAR, orientation = image.loadImage(view['path'], True)
            images.append(img)

            chunk.logger.info('Opening '+mask_file)
            mask_img, h_ori, w_ori, PAR, orientation = image.loadImage(mask_file, True)
            masks.append(mask_img)
            metas.append((h_ori, w_ori, PAR, orientation))

        #filter
        chunk.logger.info('Applying filter')
        filter_function = eval('filtering.Operations.'+chunk.node.filterType.value)
        
        kargs={}
        for a in chunk.node.attributes:
            if a.attributeDesc.group=='opt':
                kargs[a.name]=a.value 
        chunk.logger.info(kargs)
        filtered_masks = filter_function(masks,images=images,**kargs)
            
        #saving
        chunk.logger.info('Saving masks')
        for view, mask, meta in zip(sfm_data['views'], filtered_masks, metas):
            if chunk.node.keepFilename.value:
                image_basename = os.path.splitext(os.path.basename(view['path']))[0]
            else:
                image_basename = view["viewId"]
            filename = os.path.join(chunk.node.outputFolder.value, image_basename+'.'+chunk.node.extension.value)
            if len(mask.shape)<3:
                mask=np.expand_dims(mask, axis=-1)
            image.writeImage(filename, mask, meta[0], meta[1], meta[3], meta[2])

        chunk.logManager.end()

