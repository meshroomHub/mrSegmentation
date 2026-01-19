import math
import random
import os
import numpy as np
import OpenImageIO as oiio
from pathlib import Path
from pyalicevision import image as avimg

def find_metadata(oiio_spec, name: str, default, exact: bool = True):
    values = []
    oiio_extra_attribs = oiio_spec.extra_attribs
    for i in range(len(oiio_extra_attribs)):
        pos = oiio_extra_attribs[i].name.find(name)
        if pos == 0:
            values.insert(0, oiio_spec.getattribute(oiio_extra_attribs[i].name))
        elif pos != -1 and not exact:
            values.append(oiio_spec.getattribute(oiio_extra_attribs[i].name))
    if len(values) == 0:
        values.append(default)
    return values

def apply_orientation(oiio_image, orientation, reverse: bool = False):
    if orientation > 1:
        oiio_image_buf = oiio.ImageBuf(oiio_image)
        if orientation == 2:
            oiio_image_buf = oiio.ImageBufAlgo.flop(oiio_image_buf)
        elif orientation == 3:
            oiio_image_buf = oiio.ImageBufAlgo.rotate180(oiio_image_buf)
        elif orientation == 4:
                oiio_image_buf = oiio.ImageBufAlgo.flip(oiio_image_buf)
        elif orientation == 5:
            if reverse:
                oiio_image_buf = oiio.ImageBufAlgo.flop(oiio_image_buf)
                oiio_image_buf = oiio.ImageBufAlgo.rotate270(oiio_image_buf)
            else:
                oiio_image_buf = oiio.ImageBufAlgo.rotate90(oiio_image_buf)
                oiio_image_buf = oiio.ImageBufAlgo.flop(oiio_image_buf)
        elif orientation == 6:
            if reverse:
                oiio_image_buf = oiio.ImageBufAlgo.rotate270(oiio_image_buf)
            else:
                oiio_image_buf = oiio.ImageBufAlgo.rotate90(oiio_image_buf)
        elif orientation == 7:
            if reverse:
                 oiio_image_buf = oiio.ImageBufAlgo.flop(oiio_image_buf)
                 oiio_image_buf = oiio.ImageBufAlgo.rotate90(oiio_image_buf)
            else:
                 oiio_image_buf = oiio.ImageBufAlgo.rotate270(oiio_image_buf)
                 oiio_image_buf = oiio.ImageBufAlgo.flop(oiio_image_buf)
        elif orientation == 8:
            if reverse:
                oiio_image_buf = oiio.ImageBufAlgo.rotate90(oiio_image_buf)
            else:
                oiio_image_buf = oiio.ImageBufAlgo.rotate270(oiio_image_buf)
        oiio_image = oiio_image_buf.get_pixels(format=oiio.FLOAT)
    return oiio_image

def fromUsualToRawOrientation(x, y, width, height, PAR, orientation):
    x1 = x
    y1 = y
    if orientation == 2:
        x1 = width - x - 1
    elif orientation == 3:
        x1 = width - x - 1
        y1 = int(height / PAR) - y - 1
    elif orientation == 4:
        y1 = int(height / PAR) - y - 1
    elif orientation == 5:
        xtmp = y
        y1 = x
        x1 = xtmp
    elif orientation == 6:
        xtmp = y
        y1 = width - x - 1
        x1 = xtmp
    elif orientation == 7:
        xtmp = int(height / PAR) - y - 1
        y1 = width - x - 1
        x1 = xtmp
    elif orientation == 8:
        xtmp = int(height / PAR) - y - 1
        y1 = x
        x1 = xtmp
    return(x1, y1 * PAR)

def fromRawToUsualOrientation(x, y, width, height, PAR, orientation):
    x1 = x
    y1 = y / PAR
    if orientation == 2:
        x1 = width - x - 1
    elif orientation == 3:
        x1 = width - x - 1
        y1 = int(height / PAR) - y1 - 1
    elif orientation == 4:
        y1 = int(height / PAR) - y1 - 1
    elif orientation == 5:
        xtmp = int(height / PAR) - y1 - 1
        y1 = width - x - 1
        x1 = xtmp
    elif orientation == 6:
        xtmp = width - y1 - 1
        y1 = x
        x1 = xtmp
    elif orientation == 7:
        xtmp = y1
        y1 = x
        x1 = xtmp
    elif orientation == 8:
        xtmp = y1
        y1 = int(height / PAR) - x - 1
        x1 = xtmp
    return(x1, y1)

def loadImage(imagePath: str, applyPAR: bool = False, applyOrientation: bool = True):
    oiio_input = oiio.ImageInput.open(imagePath)
    oiio_spec = oiio_input.spec()
    oiio_input.close()

    av_image = avimg.Image_RGBfColor()
    avOptRead = avimg.ImageReadOptions(avimg.EImageColorSpace_SRGB)
    avimg.readImage(imagePath, av_image, avOptRead)
    oiio_image = av_image.getNumpyArray()

    pixelAspectRatio = oiio_spec.get_float_attribute('PixelAspectRatio', 1.0)
    orientation = int(find_metadata(oiio_spec, 'Orientation', 1)[0])
    oiio_spec.attribute('Orientation', orientation)

    if orientation > 1 and applyOrientation:
        oiio_image = apply_orientation(oiio_image, orientation)

    h,w,c = oiio_image.shape

    if pixelAspectRatio != 1.0 and applyPAR:
        oiio_image_buf = oiio.ImageBuf(oiio_image)
        nh = int(float(h) / pixelAspectRatio)
        oiio_image_buf = oiio.ImageBufAlgo.resize(oiio_image_buf, roi=oiio.ROI(0, w, 0, nh, 0, 1, 0, c+1))
        oiio_image = oiio_image_buf.get_pixels(format=oiio.FLOAT)

    oiio_image_buf = oiio.ImageBuf(oiio_image)
    oiio.ImageBufAlgo.max(oiio_image_buf, oiio_image_buf, 0.0)
    oiio.ImageBufAlgo.min(oiio_image_buf, oiio_image_buf, 1.0)
    oiio_image = oiio_image_buf.get_pixels(format=oiio.FLOAT)

    return (oiio_image, h, w, pixelAspectRatio, orientation)

def writeImage(imagePath: str, image: np.ndarray, h_tgt: int, w_tgt: int, orientation: int = 1, pixelAspectRatio: float = 1.0, metadata=None, optWrite=None) -> None:
    if metadata is None:
        metadata = {}
    if orientation > 1:
        image = apply_orientation(image, orientation, reverse=True)
        if orientation > 4:
            tmp = h_tgt
            h_tgt = w_tgt
            w_tgt = tmp
    h,w,c = image.shape

    if h != h_tgt or w != w_tgt:
        oiio_image_buf = oiio.ImageBuf(image)
        oiio_image_buf = oiio.ImageBufAlgo.resize(oiio_image_buf, roi=oiio.ROI(0, w_tgt, 0, h_tgt, 0, 1, 0, c+1))
        image = oiio_image_buf.get_pixels(format=oiio.FLOAT)
        w = w_tgt
        h = h_tgt

    if image.dtype == 'uint8':
        if c == 1:
            av_image = avimg.Image_uchar()
        elif c == 3:
            av_image = avimg.Image_RGBColor()
        else:
            av_image = avimg.Image_RGBAColor()
    else:
        if image.dtype != 'float32':
            image = image.astype('float32')
        if c == 1:
            av_image = avimg.Image_float()
        elif c == 3:
            av_image = avimg.Image_RGBfColor()
        else:
            av_image = avimg.Image_RGBAfColor()
    av_image.fromNumpyArray(image)

    if optWrite is None:
        optWrite = avimg.ImageWriteOptions()
        if Path(imagePath).suffix.lower() == ".exr":
            optWrite.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)
            compression = "zips"
        else:
            optWrite.toColorSpace(avimg.EImageColorSpace_SRGB)
            compression = ""
    else:
        compression = avimg.EImageExrCompression_enumToString(optWrite.getExrCompressionMethod())

    oiio_params = avimg.oiioParams(orientation, pixelAspectRatio, compression)

    for name, value in metadata.items():
        if isinstance(name, str) and isinstance(value, str) and name != "" and value != "":
            oiio_params.add(name, value)

    avimg.writeImage(imagePath, av_image, optWrite, oiio_params.get())

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

def addPoint(image: np.ndarray, point, color = (255, 0, 0)) -> np.ndarray:
    buf = oiio.ImageBuf(image)
    oiio.ImageBufAlgo.render_box(buf, int(point[0]) - 4, int(point[1]) - 4, int(point[0]) + 4, int(point[1]) + 4, color = color)
    oiio.ImageBufAlgo.render_line(buf, int(point[0]) - 4, int(point[1]) - 4, int(point[0]) + 4, int(point[1]) + 4, color = color)
    oiio.ImageBufAlgo.render_line(buf, int(point[0]) - 4, int(point[1]) + 4, int(point[0]) + 4, int(point[1]) - 4, color = color)
    return buf.get_pixels(format='uint8')

def addText(image: np.ndarray, text, x, y, size, color = (255, 0, 0)) -> np.ndarray:
    buf = oiio.ImageBuf(image)
    oiio.ImageBufAlgo.render_text(buf, int(x), int(y), text, int(size), "", color)
    return buf.get_pixels(format='uint8')

class paletteGenerator:
    def __init__(self, seed=42):
        self.seed = seed
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    def _dist(self, c1, c2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

    def _is_grey(self, color, threshold=50):
        return (max(color) - min(color)) < threshold

    def add_color(self, color=None):
        if color:
            self.colors.append(color)
            return color
        
        random.seed(self.seed + len(self.colors))
        best_color = None
        max_dist_min = -1

        for _ in range (2000):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if self._is_grey(color):
                continue
            dist_min = min(self._dist(color, c) for c in self.colors)
            if dist_min > max_dist_min:
                max_dist_min = dist_min
                best_color = color
        
        if best_color is None:  # Edge case: after 2000 iterations only grey-ish colors have been generated
            best_color = color

        self.colors.append(best_color)
        return best_color

    def generate_palette(self, n):

        if n < 0:
            return None
        if n <= len(self.colors):
            return self.colors[0:n]
        
        missing = n - len(self.colors)
        for _ in range(missing):
            self.add_color()
        return self.colors

    def index(self, color):
        if color in self.colors:
            return self.colors.index(color)
        return None

    def at(self, idx):
        if idx >=0 and idx < len(self.colors):
            return list(self.colors[idx])
        return None