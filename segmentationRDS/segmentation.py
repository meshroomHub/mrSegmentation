import os
import numpy as np
import torch
import torchvision

from math import sqrt

# Grounding DINO
from groundingdino.util.inference import predict
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig

# Segment Anything
from segment_anything import build_sam, SamPredictor, sam_model_registry
# Recognize Anything
from ram.models import ram_plus
from ram import inference_ram as inference


def get_size_with_aspect_ratio(image_size: int, size: int, max_size: int = None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return (oh, ow)

def cleanstr(s: str) -> str:
    sclean = s
    while sclean.startswith(' '):
        sclean = sclean[1:]
    while sclean.endswith(' '):
        sclean = sclean[:-1]
    return sclean


class SegmentationRDS:

    def __init__(self, RAM_CHECKPOINT_PATH:str, GD_CONFIG_PATH:str, GD_CHECKPOINT_PATH:str, SAM_CHECKPOINT_PATH:str, RAM_VIT:str='swin_l', RAM_IMAGE_SIZE:int=384, SAM_ENCODER_VERSION:str='vit_h', useGPU:bool=True):
        self.DEVICE = 'cuda' if useGPU and torch.cuda.is_available() else 'cpu'
        if self.DEVICE == 'cpu' and useGPU:
            print("Cannot execute on GPU, fallback to CPU execution mode")
        self.RAM_IMAGE_SIZE = RAM_IMAGE_SIZE
        # Load models
        # Recognize Anything
        text_encoder_type = os.getenv('RDS_TOKENIZER_PATH',"")
        if text_encoder_type == '':
            text_encoder_type = 'bert-base-uncased'
        self.ram = ram_plus(pretrained=RAM_CHECKPOINT_PATH, image_size=RAM_IMAGE_SIZE, vit=RAM_VIT, text_encoder_type=text_encoder_type)
        self.ram.eval()
        self.ram = self.ram.to(torch.device(self.DEVICE))
        # Grounded DINO
        args = SLConfig.fromfile(GD_CONFIG_PATH)
        args.device = self.DEVICE
        self.gdm = build_model(args)
        checkpoint = torch.load(GD_CHECKPOINT_PATH, map_location="cpu")
        self.gdm.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.gdm.eval()
        # Segment Anything
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(self.DEVICE)
        self.sam_predictor = SamPredictor(sam)

    def __del__(self):
        del self.ram
        del self.gdm
        del self.sam_predictor


    def get_tags(self, image:np.ndarray) -> list[str]:

        ram_Transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Resize((self.RAM_IMAGE_SIZE, self.RAM_IMAGE_SIZE)),
                                                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        ram_image = ram_Transform(image).unsqueeze(0)
        res = inference(ram_image.to(self.DEVICE), self.ram)
        tags = res[0].split('|')
        for idx in range(len(tags)):
            tags[idx] = cleanstr(tags[idx])

        return tags


    def recognize(self, wordlist:list[str], image:np.ndarray, verbose=False):

        tags = self.get_tags(image)
        if verbose:
            print(tags)
        tagInWordList = False
        for tag in tags:
            if not tagInWordList and wordlist.count(tag):
                tagInWordList = True
        if verbose:
            print("tagInWordList = {}".format(tagInWordList))

        return (tagInWordList, tags)


    def detect(self, image:np.ndarray, TEXT_PROMPT:str, BOX_THRESHOLD=0.25, TEXT_THRESHOLD=0.25, NMS_THRESHOLD=0.8):

        source_h, source_w, _ = image.shape
        gd_imgSize = get_size_with_aspect_ratio((source_w, source_h), 800, max_size=1333)
        gd_Transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Resize(gd_imgSize),
                                                       torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        gd_image = gd_Transform(image)
        # Detection
        boxes, logits, phrases = predict(self.gdm, gd_image, TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, self.DEVICE)
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = torchvision.ops.box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.numpy()
        # NMS
        nms_idx = torchvision.ops.nms(torch.from_numpy(xyxy), torch.from_numpy(confidence), NMS_THRESHOLD).numpy().tolist()
        xyxy = xyxy[nms_idx]
        confidence = confidence[nms_idx]

        return xyxy, confidence


    def segment(self, image_uint8: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.sam_predictor.set_image(image_uint8)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(box=box, multimask_output=True)
            index = np.argmax(scores)
            result_masks.append(masks[index])

        return np.array(result_masks)


    def process(self, image: np.ndarray, prompt: str, synonyms: str = '', invert: bool = False, threshold: float = 0.25, force: bool = False, bboxMargin:int = 0, verbose: bool = False) -> np.ndarray:

        listSynonyms = synonyms.split('\n')
        listPrompt = prompt.split('\n')
        wordlist = []
        for syn in listSynonyms:
            ls = syn.split(',')
            for s in ls:
                wordlist.append(cleanstr(s).lower())
        for pr in listPrompt:
            lp = pr.split('.')
            for p in lp:
                plower = cleanstr(p).lower()
                if plower not in wordlist:
                    wordlist.append(plower)
                    
        if verbose:
            print("wordlist: {}".format(wordlist))

        recoOK, tags = self.recognize(wordlist, image, verbose)
        bboxes = []
        mask_image = np.zeros_like(image)
        if recoOK or force:
            bboxes, confidence = self.detect(image=image, TEXT_PROMPT=prompt, BOX_THRESHOLD=threshold)
            if bboxMargin > 0:
                ratio = 1.0 + (min(max(0, bboxMargin), 100.0) / 100.0)
                H,W,_ = image.shape
                for k,bbox in enumerate(bboxes):
                    if bbox[0] > bbox[2]:
                        bbox[0], bbox[2] = bbox[2], bbox[0]
                        bbox[1], bbox[3] = bbox[3], bbox[1]
                    xc = (bbox[2] + bbox[0]) / 2
                    yc = (bbox[3] + bbox[1]) / 2
                    halfNewW = (bbox[2] - bbox[0]) * ratio / 2.0
                    halfNewH = (bbox[3] - bbox[1]) * ratio / 2.0
                    bboxes[k][0] = max(0, xc - halfNewW)
                    bboxes[k][2] = min(W - 1, xc + halfNewW)
                    bboxes[k][1] = max(0, yc - halfNewH)
                    bboxes[k][3] = min(H - 1, yc + halfNewH)
            masks = self.segment((255.0*image).astype('uint8'), bboxes)
            for idx in range(len(masks)):
                mask_image[masks[idx]] = [255, 255, 255]

        if invert:
            return ((mask_image[:,:,0:1] == 0).astype('float32'), bboxes, confidence, tags)
        else:
            return ((mask_image[:,:,0:1] > 0).astype('float32'), bboxes, confidence, tags)


class SegmentAnything:

    def __init__(self, SAM_CHECKPOINT_PATH:str, SAM_ENCODER_VERSION:str='vit_h', useGPU:bool=True):
        self.DEVICE = 'cuda' if useGPU and torch.cuda.is_available() else 'cpu'
        if self.DEVICE == 'cpu' and useGPU:
            print("Cannot execute on GPU, fallback to CPU execution mode")
        # Load models
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(self.DEVICE)
        self.sam_predictor = SamPredictor(sam)

    def __del__(self):
        del self.sam_predictor


    def segment(self, image_uint8: np.ndarray, xyxy: np.ndarray, point_coords: np.ndarray, point_labels: np.ndarray) -> np.ndarray:
        self.sam_predictor.set_image(image_uint8)
        result_masks = []
        for box in xyxy:
            clicks = []
            labels = []
            for k,p in enumerate(point_coords):
                if p[0] >= box[0] and p[0] <= box[2] and p[1] >= box[1] and p[1] <= box[3]:
                    clicks.append(p)
                    labels.append(point_labels[k])
            if len(clicks):
                masks, scores, logits = self.sam_predictor.predict(box=box, point_coords=np.array(clicks), point_labels=np.array(labels), multimask_output=True)
            else:
                masks, scores, logits = self.sam_predictor.predict(box=box, multimask_output=True)
            index = np.argmax(scores)
            result_masks.append(masks[index])

        return np.array(result_masks)

    def process(self, image: np.ndarray, bboxes = [], clicksIn: np.ndarray = [], clicksOut: np.ndarray = [], invert: bool = False, verbose: bool = False) -> np.ndarray:
        if len(clicksIn):
            point_coords = clicksIn
            point_labels = [1 for k in clicksIn]
        else:
            point_coords = []
            point_labels = []
        if len(clicksOut):
            point_coords.extend(clicksOut)
            point_labels_out = [0 for k in clicksOut]
        else:
            point_labels_out = []

        point_labels.extend(point_labels_out)

        mask_image = np.zeros_like(image)
        masks = self.segment((255.0*image).astype('uint8'), bboxes, np.asarray(point_coords), np.asarray(point_labels))
        for idx in range(len(masks)):
            mask_image[masks[idx]] = [255, 255, 255]

        if invert:
            return (mask_image[:,:,0:1] == 0).astype('float32')
        else:
            return (mask_image[:,:,0:1] > 0).astype('float32')

class RecognizeAnything:

    def __init__(self, RAM_CHECKPOINT_PATH:str, RAM_VIT:str='swin_l', RAM_IMAGE_SIZE:int=384, useGPU:bool=True):
        self.DEVICE = 'cuda' if useGPU and torch.cuda.is_available() else 'cpu'
        if self.DEVICE == 'cpu' and useGPU:
            print("Cannot execute on GPU, fallback to CPU execution mode")
        self.RAM_IMAGE_SIZE = RAM_IMAGE_SIZE
        # Load models
        # Recognize Anything
        text_encoder_type = os.getenv('RDS_TOKENIZER_PATH',"")
        if text_encoder_type == '':
            text_encoder_type = 'bert-base-uncased'
        self.ram = ram_plus(pretrained=RAM_CHECKPOINT_PATH, image_size=RAM_IMAGE_SIZE, vit=RAM_VIT, text_encoder_type=text_encoder_type)
        self.ram.eval()
        self.ram = self.ram.to(torch.device(self.DEVICE))

    def __del__(self):
        del self.ram

    def get_tags(self, image:np.ndarray) -> list[str]:

        ram_Transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Resize((self.RAM_IMAGE_SIZE, self.RAM_IMAGE_SIZE)),
                                                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        ram_image = ram_Transform(image).unsqueeze(0)
        res = inference(ram_image.to(self.DEVICE), self.ram)
        tags = res[0].split('|')
        for idx in range(len(tags)):
            tags[idx] = cleanstr(tags[idx])

        return tags

class DetectAnything:

    def __init__(self, RAM_CHECKPOINT_PATH:str, GD_CONFIG_PATH:str, GD_CHECKPOINT_PATH:str, RAM_VIT:str='swin_l', RAM_IMAGE_SIZE:int=384, useGPU:bool=True):
        self.DEVICE = 'cuda' if useGPU and torch.cuda.is_available() else 'cpu'
        if self.DEVICE == 'cpu' and useGPU:
            print("Cannot execute on GPU, fallback to CPU execution mode")
        self.RAM_IMAGE_SIZE = RAM_IMAGE_SIZE
        # Load models
        # Recognize Anything
        text_encoder_type = os.getenv('RDS_TOKENIZER_PATH',"")
        if text_encoder_type == '':
            text_encoder_type = 'bert-base-uncased'
        self.ram = ram_plus(pretrained=RAM_CHECKPOINT_PATH, image_size=RAM_IMAGE_SIZE, vit=RAM_VIT, text_encoder_type=text_encoder_type)
        self.ram.eval()
        self.ram = self.ram.to(torch.device(self.DEVICE))
        # Grounded DINO
        args = SLConfig.fromfile(GD_CONFIG_PATH)
        args.device = self.DEVICE
        self.gdm = build_model(args)
        checkpoint = torch.load(GD_CHECKPOINT_PATH, map_location="cpu")
        self.gdm.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.gdm.eval()

    def __del__(self):
        del self.ram
        del self.gdm

    def get_tags(self, image:np.ndarray) -> list[str]:

        ram_Transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Resize((self.RAM_IMAGE_SIZE, self.RAM_IMAGE_SIZE)),
                                                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        ram_image = ram_Transform(image).unsqueeze(0)
        res = inference(ram_image.to(self.DEVICE), self.ram)
        tags = res[0].split('|')
        for idx in range(len(tags)):
            tags[idx] = cleanstr(tags[idx])

        return tags


    def recognize(self, wordlist:list[str], image:np.ndarray, verbose=False):

        tags = self.get_tags(image)
        if verbose:
            print(tags)
        tagInWordList = False
        for tag in tags:
            if not tagInWordList and wordlist.count(tag):
                tagInWordList = True
        if verbose:
            print("tagInWordList = {}".format(tagInWordList))

        return (tagInWordList, tags)


    def detect(self, image:np.ndarray, TEXT_PROMPT:str, BOX_THRESHOLD=0.25, TEXT_THRESHOLD=0.25, NMS_THRESHOLD=0.8):

        source_h, source_w, _ = image.shape
        gd_imgSize = get_size_with_aspect_ratio((source_w, source_h), 800, max_size=1333)
        gd_Transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Resize(gd_imgSize),
                                                       torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        gd_image = gd_Transform(image)
        # Detection
        boxes, logits, phrases = predict(self.gdm, gd_image, TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, self.DEVICE)
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = torchvision.ops.box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.numpy()
        # NMS
        nms_idx = torchvision.ops.nms(torch.from_numpy(xyxy), torch.from_numpy(confidence), NMS_THRESHOLD).numpy().tolist()
        xyxy = xyxy[nms_idx]
        confidence = confidence[nms_idx]

        return xyxy, confidence


    def process(self, image: np.ndarray, prompt: str, synonyms: str = '', threshold: float = 0.2, force: bool = False, bboxMargin:int = 0, verbose: bool = False) -> np.ndarray:

        listSynonyms = synonyms.split('\n')
        listPrompt = prompt.split('\n')
        wordlist = []
        for syn in listSynonyms:
            ls = syn.split(',')
            for s in ls:
                wordlist.append(cleanstr(s).lower())
        for pr in listPrompt:
            lp = pr.split('.')
            for p in lp:
                plower = cleanstr(p).lower()
                if plower not in wordlist:
                    wordlist.append(plower)
                    
        if verbose:
            print("wordlist: {}".format(wordlist))

        recoOK, tags = self.recognize(wordlist, image, verbose)
        bboxes = np.array([])
        confidence = np.array([])
        if recoOK or force:
            bboxes, confidence = self.detect(image=image, TEXT_PROMPT=prompt, BOX_THRESHOLD=threshold)
            if bboxMargin > 0:
                ratio = 1.0 + (min(max(0, bboxMargin), 100.0) / 100.0)
                H,W,_ = image.shape
                for k,bbox in enumerate(bboxes):
                    if bbox[0] > bbox[2]:
                        bbox[0], bbox[2] = bbox[2], bbox[0]
                        bbox[1], bbox[3] = bbox[3], bbox[1]
                    xc = (bbox[2] + bbox[0]) / 2
                    yc = (bbox[3] + bbox[1]) / 2
                    halfNewW = (bbox[2] - bbox[0]) * ratio / 2.0
                    halfNewH = (bbox[3] - bbox[1]) * ratio / 2.0
                    bboxes[k][0] = max(0, xc - halfNewW)
                    bboxes[k][2] = min(W - 1, xc + halfNewW)
                    bboxes[k][1] = max(0, yc - halfNewH)
                    bboxes[k][3] = min(H - 1, yc + halfNewH)

        return (bboxes, confidence, tags)

class BiRefNetSeg:

    def __init__(self, modelType:str, useGPU:bool=True):
        from birefnet.models.birefnet import BiRefNet

        self.DEVICE = 'cuda' if useGPU and torch.cuda.is_available() else 'cpu'
        if self.DEVICE == 'cpu' and useGPU:
            print("Cannot execute on GPU, fallback to CPU execution mode")
        # Load models
        pretrainedModel = 'ZhengPeng7/BiRefNet_HR-matting'
        if modelType == 'BiRefNet LR':
            pretrainedModel = 'ZhengPeng7/BiRefNet'
        elif modelType == 'BiRefNet HR':
            pretrainedModel = 'ZhengPeng7/BiRefNet_HR'

        self.brn = BiRefNet.from_pretrained(pretrainedModel)

        torch.set_float32_matmul_precision('highest')

        self.brn.to(self.DEVICE)
        self.brn.eval()
        self.brn.half()

    def __del__(self):
        del self.brn

    def segment(self, image_uint8: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        max_image_size = (2048, 4096)
        input_image_size = (image_uint8.shape[0], image_uint8.shape[1])
        matte_image = np.zeros(input_image_size)
        result_masks = [matte_image[..., None]]

        for box in xyxy:

            bboxImage = image_uint8[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
            bboxSize = (bboxImage.shape[0], bboxImage.shape[1])

            padMode = bboxSize[0] * bboxSize[1] < max_image_size[0] * max_image_size[1]
            if padMode:
                padding = [((imgDim - 1) // 32 + 1) * 32 - imgDim for imgDim in bboxSize]
                resizeTransf = torchvision.transforms.Pad((0, 0, padding[1], padding[0]), padding_mode='symmetric')
            else:
                resize_ratio = sqrt((max_image_size[0] * max_image_size[1])/(bboxSize[0] * bboxSize[1]))
                resize = [(round(resize_ratio * imgDim - 1) // 32 + 1) * 32 for imgDim in bboxSize]
                resize = (resize[1], resize[0])
                resizeTransf = torchvision.transforms.Resize(resize)

            transform_image = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                resizeTransf,
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_images = transform_image(bboxImage).unsqueeze(0).to(self.DEVICE).half()

            # Prediction
            with torch.no_grad():
                preds = self.brn(input_images)[-1].sigmoid().cpu()

            if padMode:
                pred = torchvision.transforms.functional.crop(preds[0], 0, 0, bboxSize[0], bboxSize[1])
            else:
                pred = torchvision.transforms.Resize(bboxSize)(preds[0])

            pred = pred[0].numpy()
            matte_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = pred

            result_masks.append(matte_image[..., None])
            matte_image = np.zeros(input_image_size)

        return np.array(result_masks)

    def process(self, image: np.ndarray, bboxes: np.ndarray = [], invert: bool = False, verbose: bool = False) -> np.ndarray:

        masks = self.segment((255.0*image).astype('uint8'), bboxes)
        masksMax = np.max(masks, axis=0)

        if invert:
            oneImg = np.ones_like(masksMax)
            masksMax = oneImg - masksMax
        
        return masksMax
