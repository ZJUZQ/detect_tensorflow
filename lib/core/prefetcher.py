"""Provides functions to prefetch tensors to feed into models."""
import tensorflow as tf


def prefetch(tensor_dict, capacity):
	"""Creates a prefetch queue for tensors.

	Creates a FIFO queue to asynchronously enqueue tensor_dicts and returns a
	dequeue op that evaluates to a tensor_dict. This function is useful in
	prefetching preprocessed tensors so that the data is readily available for
	consumers.

	Example input pipeline when you don't need batching:
	----------------------------------------------------
	key, string_tensor = slim.parallel_reader.parallel_read(...)
	tensor_dict = decoder.decode(string_tensor)
	tensor_dict = preprocessor.preprocess(tensor_dict, ...)
	prefetch_queue = prefetcher.prefetch(tensor_dict, capacity=20)
	tensor_dict = prefetch_queue.dequeue()
	outputs = Model(tensor_dict)
	...
	----------------------------------------------------

	For input pipelines with batching, refer to core/batcher.py

	Args:
		tensor_dict: a dictionary of tensors to prefetch; one queue element;
		capacity: the size of the prefetch queue.

	Returns:
		a FIFO prefetcher queue
	"""

	names = list(tensor_dict.keys())
	dtypes = [t.dtype for t in tensor_dict.values()]
	shapes = [t.get_shape() for t in tensor_dict.values()]

	# Creates a queue that dequeues elements in a first-in first-out order
	prefetch_queue = tf.PaddingFIFOQueue(capacity, 
										dtypes=dtypes,
										shapes=shapes,
										names=names,
										name='prefetch_queue')
	# Enqueues one element to this queue
	enqueue_op = prefetch_queue.enqueue(tensor_dict)

	# Adds a QueueRunner to a collection in the graph
	tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(queue=prefetch_queue, 
																			 enqueue_ops=[enqueue_op]), 
										   collection=tf.GraphKeys.QUEUE_RUNNERS)
	tf.summary.scalar(name='queue/%s/fraction_of_%d_full' % (prefetch_queue.name, capacity),
	        		  tensor=tf.to_float(prefetch_queue.size()) * (1. / capacity))

	return prefetch_queue
