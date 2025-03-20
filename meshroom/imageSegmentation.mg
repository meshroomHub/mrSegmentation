{
    "header": {
        "releaseVersion": "2025.1.0-develop",
        "fileVersion": "2.0",
        "nodesVersions": {
            "CameraInit": "12.0",
            "ImageBiRefNetMatting": "0.2",
            "ImageDetectionPrompt": "0.1",
            "ImageSegmentationBox": "0.2",
            "Publish": "1.3"
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
        "ImageBiRefNetMatting_1": {
            "nodeType": "ImageBiRefNetMatting",
            "position": [
                400,
                -80
            ],
            "inputs": {
                "input": "{ImageDetectionPrompt_1.input}",
                "bboxFolder": "{ImageDetectionPrompt_1.output}"
            }
        },
        "ImageDetectionPrompt_1": {
            "nodeType": "ImageDetectionPrompt",
            "position": [
                200,
                0
            ],
            "inputs": {
                "input": "{CameraInit_1.output}"
            }
        },
        "ImageSegmentationBox_1": {
            "nodeType": "ImageSegmentationBox",
            "position": [
                400,
                60
            ],
            "inputs": {
                "input": "{ImageDetectionPrompt_1.input}",
                "bboxFolder": "{ImageDetectionPrompt_1.output}",
                "keepFilename": true
            }
        },
        "Publish_1": {
            "nodeType": "Publish",
            "position": [
                600,
                0
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