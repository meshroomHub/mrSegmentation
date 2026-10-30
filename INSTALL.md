# Installation

## With the Meshroom plugin manager

mrSegmentation can be installed from the Meshroom plugin manager (meshroomHub registry), from the UI or with:
```
meshroom_plugins install mrSegmentation
```

The plugin manager creates the plugin's virtual environment from `pyproject.toml` (PyTorch with CUDA 12.4) and
downloads the RAM++, Grounding DINO and SAM models (about 6.4 GB) into its `models` folder.
The SAM 3 checkpoint is not downloaded: see [SAM 3 model](#sam-3-model).

## Manual installation

For the installation, a version of Python >= 3.9 is required.

### Dependencies

It is recommended to install the dependencies of mrSegmentation through a virtual environment. 

> [!TIP]
> Meshroom expects the virtual environment to be named "venv", otherwise it will not be recognised.

> [!NOTE]  
> By default, some of the dependencies in the `requirements.txt` file install a CPU-only version of PyTorch. The plugin will run smoothly on the CPU, but for better performances, it is advised to install the GPU-enabled version of PyTorch.

- On Linux:
```
cd /path/to/mrSegmentation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# For GPU support (optional)
pip install -r torch-requirements.txt --upgrade --force-reinstall

deactivate
```

- On Windows:
```
cd /path/to/mrSegmentation
python -m venv venv
.\venv\Scripts\activate.bat
pip install -r requirements.txt

# For GPU support (optional)
pip install -r torch-requirements.txt --upgrade --force-reinstall

deactivate
```

### Models

#### Downloading models

Models are available on the [following repository](https://gitlab.com/alicevision/imagepromptnnmodels).
```
git clone -n --depth=1 --filter=tree:0 https://gitlab.com/alicevision/imagepromptnnmodels
cd imagepromptnnmodels
git sparse-checkout set --no-cone /models
git checkout
```

It is recommended to move the `models` folder to the mrSegmentation repository, at the top-level.

#### Setting up environment variables

The following environment variables need to be set for the nodes to find the appropriate models and use the correct settings:

```
TRANSFORMERS_OFFLINE=TRUE
RDS_DETECTION_MODEL_PATH=/path/to/models/groundingdino_swint_ogc.pth
RDS_RECOGNITION_MODEL_PATH=/path/to/models/ram_plus_swin_large_14m.pth
RDS_SEGMENTATION_MODEL_PATH=/path/to/models/sam_vit_h_4b8939.pth
RDS_TOKENIZER_PATH=/path/to/models/tokenizer
```

### Setting up mrSegmentation for Meshroom

- On Linux:
```
export MESHROOM_PLUGINS_PATH=/path/to/mrSegmentation:$MESHROOM_PLUGINS_PATH
```

- On Windows:
```
set MESHROOM_PLUGINS_PATH=/path/to/mrSegmentation;%MESHROOM_PLUGINS_PATH%
```

## SAM 3 model

The SAM 3 nodes (`ImageSegmentationSam3`, `VideoSegmentationSam3*`) need the `sam3.pt` checkpoint of
[facebook/sam3](https://huggingface.co/facebook/sam3). Access to it is gated: accept its license on Hugging Face,
then download it with your Hugging Face token and point the nodes to it:
```
hf download facebook/sam3 sam3.pt --local-dir /path/to/sam3
export RDS_SAM3_MODEL_PATH=/path/to/sam3/sam3.pt
```

For a manual installation, the `sam3` package must also be installed in the virtual environment:
```
pip install "sam3 @ git+https://github.com/facebookresearch/sam3.git@7f473cfd983d5eb43040ff608f66bff0d7bdcb53" "setuptools<81" psutil
```
