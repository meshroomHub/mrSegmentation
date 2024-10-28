#!/usr/bin/env python
# coding:utf-8

""" 
Tests ensuring that all nodes can be loaded correctly, even if the environment is not correctly setup.
"""

NODE_TYPES = ["ImageDetectionPrompt", "ImageSegmentationBox", "ImageSegmentationPrompt", "ImageTagsExtraction", "SegmentAnything"]

from os import listdir
from os.path import basename, splitext, isfile, join

NODE_TYPES = [splitext(basename(f))[0] for f in listdir("../meshroom/nodes/imageSegmentation") if f != "__init__.py"]

import meshroom.core

def test_load_nodes_without_env():
    meshroom.core.initNodes()

    for type in NODE_TYPES:
        assert type in meshroom.core.desc
