import os
import argparse
import numpy as np
import tensorflow as tf
# --tensorflow 1.13
# import tensorflow.contrib.tensorrt as trt 
# --tensorflow 1.14
from tensorflow.python.compiler.tensorrt import trt_convert as trt

INPUT_NAME = 'image_tensor'
BOXES_NAME = 'detection_boxes'
CLASSES_NAME = 'detection_classes'
SCORES_NAME = 'detection_scores'
MASKS_NAME = 'detection_masks'
NUM_DETECTIONS_NAME = 'num_detections'

def saved_model_trt(
    input_saved_model_dir, 
    output_dir,
    max_batch_size, 
    precision_mode,
    is_dynamic_op):
    '''
    create a TensorRT inference graph from a SavedModel
    '''
    output_frozen_graph_path = os.path.join(output_dir, 'trt_frozen_graph.pb')
    trt_graph = trt.create_inference_graph(
        input_graph_def=None,
        outputs=None,
        input_saved_model_dir=input_saved_model_dir,
        input_saved_model_tags=['serve'],
        max_batch_size=max_batch_size,
        max_workspace_size_bytes=trt.DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
        precision_mode=precision_mode,
        output_saved_model_dir=output_dir,
        is_dynamic_op=False)
    with open(output_frozen_graph_path, 'wb') as f:
        f.write(trt_graph.SerializeToString())

def frozen_graph_trt(
    input_frozen_graph_path,
    output_dir,
    max_batch_size,
    precision_mode,
    is_dynamic_op):
    '''
    create a TensorRT inference graph from a Frozen Graph
    '''
    output_node_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_frozen_graph_path = os.path.join(output_dir, 'trt_frozen_graph.pb')
    with tf.io.gfile.GFile(input_frozen_graph_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    trt_graph = trt.create_inference_graph(
        input_graph_def=graph_def,
        outputs=output_node_names,
        max_batch_size=max_batch_size,
        max_workspace_size_bytes=trt.DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
        precision_mode=precision_mode,
        is_dynamic_op=False)

    with open(output_frozen_graph_path, 'wb') as f:
        f.write(trt_graph.SerializeToString())


def ckpt_trt():
    '''
    create a TensorRT inference graph from a `MetaGraph` and checkpoint files
    '''

    # use tf.graph_util.convert_variables_to_constants freeze ckpt to frozen graph
    # and then use frozen_graph_trr()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a TF-TRT inference graph.')
    parser.add_argument('-t', '--input_model_type', 
        type=str, 
        choices=['sm', 'fg', 'ck'],
        default='sm', 
        help='Input model type, sm means SavedModel, fg means Frozen Graph and ck means MetaGraph+checkpoint. Default is sm.')
    parser.add_argument('-i', '--input_model_path', 
        type=str, 
        required=True,
        help='Input model path, /path/.pb or /path/saved_model/')
    parser.add_argument('-o', '--output_dir',
        type=str,
        required=True,
        help='Output dir where the TRT inference graph will be saved')
    parser.add_argument('-b', '--max_batch_size',
        type=int,
        default=32)
    parser.add_argument('-p', '--precision_mode',
        type=str,
        default='FP32')
    parser.add_argument('-d', '--is_dynamic_op',
        type=bool,
        default=False)

    args = parser.parse_args()

    if args.input_model_type == 'sm':
        saved_model_trt(args.input_model_path,
            args.output_dir,
            args.max_batch_size,
            args.precision_mode,
            args.is_dynamic_op)
    if args.input_model_type == 'fg':
        frozen_graph_trt(args.input_model_path,
            args.output_dir,
            args.max_batch_size,
            args.precision_mode,
            args.is_dynamic_op)
