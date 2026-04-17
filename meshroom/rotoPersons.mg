{
    "header": {
        "releaseVersion": "2026.1.0+develop",
        "fileVersion": "2.0",
        "nodesVersions": {
            "CameraInit": "12.1",
            "CopyFiles": "1.3",
            "VideoSegmentationSam3Boxes": "2.0",
            "VideoSegmentationSam3Text": "1.0"
        },
        "template": true
    },
    "graph": {
        "CameraInit_1": {
            "nodeType": "CameraInit",
            "position": [
                -452,
                94
            ],
            "inputs": {}
        },
        "CopyFiles_1": {
            "nodeType": "CopyFiles",
            "position": [
                229,
                73
            ],
            "inputs": {
                "output": "{VideoSegmentationSam3Boxes_1.output}"
            }
        },
        "VideoSegmentationSam3Boxes_1": {
            "nodeType": "VideoSegmentationSam3Boxes",
            "position": [
                9,
                41
            ],
            "inputs": {
                "input": "{VideoSegmentationSam3Text_1.input}",
                "masksFolder": "{VideoSegmentationSam3Text_1.output}",
                "bboxesFolder": "{VideoSegmentationSam3Text_1.output}",
                "verboseLevel": "debug"
            }
        },
        "VideoSegmentationSam3Text_1": {
            "nodeType": "VideoSegmentationSam3Text",
            "position": [
                -221,
                61
            ],
            "inputs": {
                "input": "{CameraInit_1.output}",
                "timeSlicing": true,
                "sliceSize": 64,
                "verboseLevel": "debug"
            }
        }
    }
}