import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util

def get_graph_info(model_path, model_type):
	if model_type == 'fg':
		with tf.Graph().as_default():
			frozen_graph = tf.compat.v1.GraphDef()
			with tf.io.gfile.GFile(model_path, 'rb') as fid:
				frozen_graph.ParseFromString(fid.read())
		graph_size = len(frozen_graph.SerializeToString())
		num_nodes_total = len(frozen_graph.node)
		num_nodes_trt = len([1 for n in frozen_graph.node if str(n.op)=='TRTEngineOp'])
		print("graph_size(MB): %.1f" % (float(graph_size)/(1<<20)))	

	else:
		with tf.Session(graph = tf.Graph()) as sess:
			tf.saved_model.loader.load(sess, ['serve'], model_path)
			graph = tf.get_default_graph()
		
		sess = tf.Session(graph=graph)
		ops = graph.get_operations()
		num_nodes_total = len(ops)
		num_nodes_trt = len([1 for op in ops if str(op)=='TRTEngineOp'])
		for op in ops:
			print(str(op[op]))
	
	print("num_nodes(total): %d" % num_nodes_total)
	print("num_nodes(trt_only): %d" % num_nodes_trt)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--model_type', type = str, choices=['fg', 'sm'], default = 'fg')
	parser.add_argument('-p', '--model_path', type = str, required=True)
	args = parser.parse_args()
	get_graph_info(args.model_path, args.model_type)


