#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import print_function
import common
import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw, Image
import argparse
import time
import cv2
from halo import Halo

from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

import os

TRT_LOGGER = trt.Logger()


def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    if any(param is None for param in [bboxes, confidences, categories]):
        return image_raw
    draw = ImageDraw.Draw(image_raw)
    # print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(
            x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(
            y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12),
                  '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << 28  # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(
                    onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 608, 608]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(
                onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def convertCV2PIL(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image


def convertPIL2CV(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def main():
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--onnx', type=str,
                        help="File path to the onnx model")
    parser.add_argument('-e', '--engine', type=str,
                        help="File path to store the engine")
    parser.add_argument('-v', '--video', type=str,
                        help="Path to the video file")
    parser.add_argument('-f', '--frame', type=int,
                        help="Number of frames to run the script")
    parser.add_argument('-s', '--save', type=str, default="result.mp4", help="Path to save the output result")
    args = parser.parse_args()

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    # onnx_file_path = 'yolov3.onnx'
    # engine_file_path = "yolov3.trt"

    onnx_file_path = args.onnx
    engine_file_path = args.engine

    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_yolov3_HW = (608, 608)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],                    # A list of 3 three-dimensional tuples for the YOLO masks
                          "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),  # A list of 9 two-dimensional tuples for the YOLO anchors
                                                     (59, 119), (116, 90), (156, 198), (373, 326)],
                          # Threshold for object coverage, float value between 0 and 1
                          "obj_threshold": 0.6,
                          # Threshold for non-max suppression algorithm, float value between 0 and 1
                          "nms_threshold": 0.5,
                          "yolo_input_resolution": input_resolution_yolov3_HW}

    postprocessor = PostprocessYOLO(**postprocessor_args)

    # Output shapes expected by the post-processor
    output_shapes = [(1, 30, 19, 19), (1, 30, 38, 38), (1, 30, 76, 76)]

    # Do inference with TensorRT
    # trt_outputs = []

    input_video = cv2.VideoCapture(args.video)
    frame_width = input_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height =input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_size = (int(frame_width), int(frame_height))

    if os.path.exists(args.save):
        print(f"Removing the already existing the {args.save}")
        os.remove(args.save)

    input_video_fps = int(input_video.get(cv2.CAP_PROP_FPS))
    output_video_writer = cv2.VideoWriter(
        args.save, cv2.VideoWriter_fourcc(*'MP4V'), input_video_fps, frame_size)

    frame_count = 0


    # testing row major
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with Halo(spinner="dots", text="loading frames") as sp:
            while True:
                ret, frame = input_video.read()

                if ret:
                    frame_count += 1
                    image = convertCV2PIL(frame)

                    image_raw, image = preprocessor.process_image(image)
                    # Store the shape of the original input image in WH format, we will need it for later
                    shape_orig_WH = image_raw.size


                    # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
                    inputs[0].host = image

                    
                    # starting the timer
                    start = time.time()

                    trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

                    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
                    trt_outputs = [output.reshape(shape) for output, shape in zip(
                        trt_outputs, output_shapes)]

                    # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
                    boxes, classes, scores = postprocessor.process(
                        trt_outputs, (shape_orig_WH))
                    
                    # ending of the timer
                    end = time.time()

                    inference_fps = round(1 / (end - start), 2)
                    sp.text = f"Frame {frame_count} Inference Fps {inference_fps}"

                    # Draw the bounding boxes onto the original input image and save it as a PNG file
                    obj_detected_img = draw_bboxes(
                        image_raw, boxes, scores, classes, ALL_CATEGORIES)

                    detection = convertPIL2CV(obj_detected_img)
                    cv2.putText(detection, f"Input FPS: {input_video_fps} | Inference FPS {inference_fps}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    output_video_writer.write(detection)

                    if args.frame is not None and args.frame == frame_count:
                        break
                else:
                    break

    input_video.release()
    output_video_writer.release()

    # output_image_path = 'dog_bboxes.png'
    # obj_detected_img.save(output_image_path, 'PNG')
    # print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))


if __name__ == '__main__':
    main()
