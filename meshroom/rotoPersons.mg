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
                0,
                0
            ],
            "inputs": {}
        },
        "CopyFiles_1": {
            "nodeType": "CopyFiles",
            "position": [
                600,
                0
            ],
            "inputs": {
                "inputFiles": [
                    "{VideoSegmentationSam3Boxes_1.output}"
                ]
            }
        },
        "VideoSegmentationSam3Boxes_1": {
            "nodeType": "VideoSegmentationSam3Boxes",
            "position": [
                400,
                0
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
                200,
                0
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