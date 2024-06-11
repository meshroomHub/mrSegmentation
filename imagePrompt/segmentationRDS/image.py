import os
import numpy as np
import OpenImageIO as oiio

knowncolorspaces = ['srgb', 'linear_srgb', 'aces', 'acescg']

IDENTITY_mat = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]])

ACES_AP1_TO_AP0_mat = np.array([[0.6954522414, 0.1406786965, 0.1638690622],
                                [0.0447945634, 0.8596711185, 0.0955343182],
                                [-0.0055258826, 0.0040252103, 1.0015006723]])

ACES_AP0_TO_sRGB_mat = np.array([[2.52140088857822, -1.13399574938275, -0.387561856768867],
                                 [-0.276214061561748, 1.37259556630409, -0.0962823557364663],
                                 [-0.0153202000774786, -0.152992561800699, 1.16838719961932]])

ACES_AP1_TO_sRGB_mat = np.matmul(ACES_AP0_TO_sRGB_mat, ACES_AP1_TO_AP0_mat)

def get_conversion_matrix(fromcolorspace:str, tocolorspace:str) -> np.ndarray:
    if not fromcolorspace in knowncolorspaces:
        print("Warning: {} color space is unknown, identity matrix returned !!!".format(fromcolorspace))
        return IDENTITY_mat
    if not tocolorspace in knowncolorspaces:
        print("Warning: {} color space is unknown, identity matrix returned !!!".format(tocolorspace))
        return IDENTITY_mat
    if fromcolorspace == tocolorspace:
        return IDENTITY_mat
    if fromcolorspace == 'srgb' or fromcolorspace == 'linear_srgb':
        if tocolorspace == 'aces':
            return ACES_AP0_TO_sRGB_mat.I
        elif tocolorspace == 'acescg':
            return ACES_AP1_TO_sRGB_mat.I
        else:
            return IDENTITY_mat
    elif fromcolorspace == 'aces':
        if tocolorspace == 'srgb' or tocolorspace == 'linear_srgb':
            return ACES_AP0_TO_sRGB_mat
        elif tocolorspace == 'acescg':
            return ACES_AP1_TO_AP0_mat.I
        else:
            return IDENTITY_mat
    elif fromcolorspace == 'acescg':
        if tocolorspace == 'srgb' or tocolorspace == 'linear_srgb':
            return ACES_AP1_TO_sRGB_mat
        elif tocolorspace == 'aces':
            return ACES_AP1_TO_AP0_mat
        else:
            return IDENTITY_mat

def srgb_gamma_inv(x):
    if x < 0.003039935:
        return 12.92321018 * max(0.0, x)
    else:
        return 1.055 * pow(min(1.0, x), 1.0/2.4) - 0.055

def loadImage(imagePath: str, incolorspace: str = 'acescg') -> np.ndarray:
    oiio_input = oiio.ImageInput.open(imagePath)
    oiio_spec = oiio_input.spec ()
    oiio_image = oiio_input.read_image(0, 3) # keep only R,G,B channels, float data type
    if imagePath[-4:].lower() == ".exr":
        h,w,c = oiio_image.shape
        convmat = get_conversion_matrix(incolorspace, 'srgb')
        oiio_image = oiio_image.reshape(h*w,3).T
        np.matmul(convmat, oiio_image, oiio_image)
        oiio_image = oiio_image.T.reshape(h,w,3)
        np_srgb_gamma_inv = np.frompyfunc(srgb_gamma_inv, 1, 1)
        ginv = np_srgb_gamma_inv(np.array(range(100000)) / 100000)
        oiio_image = ginv[np.clip(100000 * oiio_image, 0, 99999).astype('int')].astype('float32')

    oiio_input.close()

    return oiio_image

def writeImage(imagePath: str, image: np.ndarray) -> None:
    h,w,c = image.shape

    output_image = oiio.ImageOutput.create(imagePath)
    output_image_spec = oiio.ImageSpec(w, h, c, oiio.UINT8)
    if imagePath[-4:].lower() == ".exr":
        output_image_spec.attribute('compression', 'zips') # required to get zip (1 scanline) compression method
    output_image.open(imagePath, output_image_spec)

    output_image.write_image(image)
    output_image.close()

def loadSequence(sequencePath: str, incolorspace: str = 'acescg', start: int = 0, stop: int = -1, verbose = False):
    rawfiles = []
    if os.path.isdir(sequencePath):
        rawfiles = sorted(os.listdir(sequencePath))
    sequence = []
    listnames = []
    for i, f in enumerate(rawfiles):
        if i >= start and (i < stop or stop == -1):
            if verbose:
                print("read {}".format(f), end=chr(13))
            sequence.append(loadImage(os.path.join(sequencePath, f), incolorspace))
            listnames.append(f)
    if verbose:
        print('\n')
    return (listnames, np.stack(sequence))

def addRectangle(image: np.ndarray, rect, color = (255, 0, 0), fill = False) -> np.ndarray:
    buf = oiio.ImageBuf(image)
    oiio.ImageBufAlgo.render_box(buf, int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]), color, fill)
    return buf.get_pixels(format='uint8')

def addPoint(image: np.ndarray, point) -> np.ndarray:
    buf = oiio.ImageBuf(image)
    oiio.ImageBufAlgo.render_text(buf, int(point[0]) - 8, int(point[1]) - 8, '*')
    return buf.get_pixels(format='uint8')
