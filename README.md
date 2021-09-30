# Object Detection With The ONNX TensorRT Backend In Python

**Table Of Contents**
- [Description](#description)
- [How does this work?](#how-does-this-work)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, yolov3_onnx, implements a full ONNX-based pipeline for performing inference with the YOLOv3 network, with an input size of 608 x 608 pixels, including pre and post-processing. This sample is based on the [YOLOv3-608](https://pjreddie.com/media/files/papers/YOLOv3.pdf) paper.

## How does this work?

First, the original YOLOv3 specification from the paper is converted to the Open Neural Network Exchange (ONNX) format in `yolov3_to_onnx.py` (only has to be done once).

Second, this ONNX representation of YOLOv3 is used to build a TensorRT engine, followed by inference on a sample image in `onnx_to_tensorrt.py`. The predicted bounding boxes are finally drawn to the original input image and saved to disk.

After inference, post-processing including bounding-box clustering is applied. The resulting bounding boxes are eventually drawn to a new image file and stored on disk for inspection.

**Note:** This sample is not supported on Ubuntu 14.04 and older.

## Prerequisites

For specific software versions, see the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html).

1.  Install the dependencies for Python.
    ```sh
    python3 -m pip install -r requirements.txt
    ```

2.  Download sample data. 

    ```sh
    python3 download.py
    ```

    This will download the config file, weights, testing image and video

## Running the sample

1.  Create an ONNX version of YOLOv3 with the following command.
    ```sh
    python3 yolov3_to_onnx.py -c <path to config file> -w <path to weight file> -o <path to output onnx model>
    ```

    ```sh
    python3 yolov3_to_onnx.py -c yolov3.cfg -w yolov3.weights -o yolov3.onnx
    ```

2.  Build a TensorRT engine from the generated ONNX file and run inference on a sample image
    ```sh
    python3 onnx_to_tensorrt.py -o <path to onnx model> -e <path to the engine file> -i <path to the input video file> -f <number of frames to run the script>
    ```

    ```sh
    python3 onnx_to_tensorrt.py -o yolov3.onnx -e yolov3.trt -i test_video.mp4 -f 100
    ```