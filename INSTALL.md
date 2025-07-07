# Installation

For the installation, a version of Python >= 3.9 is required.

## Dependencies

It is recommended to install the dependencies of mrSegmentation through a virtual environment. 

> [!TIP]
> Meshroom expects the virtual environment to be named after the plugin, so the created venv will be named "mrSegmentation".

> [!NOTE]  
> By default, some of the dependencies in the `requirements.txt` file install a CPU-only version of PyTorch. The plugin will run smoothly on the CPU, but for better performances, it is advised to install the GPU-enabled version of PyTorch.

- On Linux:
```
cd /path/to/mrSegmentation
python -m venv mrSegmentation
source mrSegmentation/bin/activate
pip install -r requirements.txt

# For GPU support (optional)
pip install -r torch-requirements.txt --upgrade --force-reinstall

deactivate
```

- On Windows:
```
cd /path/to/mrSegmentation
python -m venv mrSegmentation
.\mrSegmentation\Scripts\activate.bat
pip install -r requirements.txt

# For GPU support (optional)
pip install -r torch-requirements.txt --upgrade --force-reinstall

deactivate
```

## Models

### Downloading models

Models are available on the [following repository](https://gitlab.com/alicevision/imagepromptnnmodels).
```
git clone -n --depth=1 --filter=tree:0 https://gitlab.com/alicevision/imagepromptnnmodels
cd imagepromptnnmodels
git sparse-checkout set --no-cone /models
git checkout
```

It is recommended to move the `models` folder to the mrSegmentation repository, at the top-level.

### Setting up environment variables

The following environment variables need to be set for the nodes to find the appropriate models and use the correct settings:

```
TRANSFORMERS_OFFLINE=TRUE
RDS_DETECTION_MODEL_PATH=/path/to/models/groundingdino_swint_ogc.pth
RDS_RECOGNITION_MODEL_PATH=/path/to/models/ram_plus_swin_large_14m.pth
RDS_SEGMENTATION_MODEL_PATH=/path/to/models/sam_vit_h_4b8939.pth
RDS_TOKENIZER_PATH=/path/to/models/tokenizer
```


## Setting up mrSegmentation for Meshroom

- On Linux:
```
export MESHROOM_PLUGINS_PATH=/path/to/mrSegmentation:$MESHROOM_PLUGINS_PATH
```

- On Windows:
```
set MESHROOM_PLUGINS_PATH=/path/to/mrSegmentation;%MESHROOM_PLUGINS_PATH%
```