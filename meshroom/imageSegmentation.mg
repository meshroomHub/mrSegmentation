{
    "header": {
        "releaseVersion": "2025.1.0-develop",
        "fileVersion": "2.0",
        "nodesVersions": {
            "CameraInit": "12.0",
            "ImageBiRefNetMatting": "0.1",
            "ImageDetectionPrompt": "0.1",
            "ImageSegmentationBox": "0.1",
            "Publish": "1.3"
        },
        "template": true
    },
    "graph": {
        "CameraInit_1": {
            "nodeType": "CameraInit",
            "position": [
                -2,
                -60
            ],
            "inputs": {},
            "internalInputs": {
                "color": "#575963"
            }
        },
        "ImageBiRefNetMatting_1": {
            "nodeType": "ImageBiRefNetMatting",
            "position": [
                423,
                -116
            ],
            "inputs": {
                "input": "{ImageDetectionPrompt_1.input}",
                "bboxFolder": "{ImageDetectionPrompt_1.output}"
            }
        },
        "ImageDetectionPrompt_1": {
            "nodeType": "ImageDetectionPrompt",
            "position": [
                198,
                -60
            ],
            "inputs": {
                "input": "{CameraInit_1.output}"
            },
            "internalInputs": {
                "color": "#575963"
            }
        },
        "ImageSegmentationBox_1": {
            "nodeType": "ImageSegmentationBox",
            "position": [
                423,
                -4
            ],
            "inputs": {
                "input": "{ImageDetectionPrompt_1.input}",
                "bboxFolder": "{ImageDetectionPrompt_1.output}",
                "maskInvert": true,
                "keepFilename": true
            },
            "internalInputs": {
                "color": "#575963"
            }
        },
        "Publish_1": {
            "nodeType": "Publish",
            "position": [
                630,
                -50
            ],
            "inputs": {
                "inputFiles": [
                    "{ImageSegmentationBox_1.output}",
                    "{ImageBiRefNetMatting_1.output}"
                ]
            }
        }
    }
}