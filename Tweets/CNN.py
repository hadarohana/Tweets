import tensorflow as tf
import tensorboard as tb
import numpy as np
import datetime
import os
import time


class CNN:
    #split is tuple of floats representing cutoff locations for train valid test split eg (.6, .8)
    def __init__(self, x, y, split, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, embedding_matrix, batch_size, num_epochs):

        train_x = x[:split[0]*len(x)]
        valid_x = x[split[0]*len(x):split[1]*len(x)]
        test_x = x[split[1]*len(x):]

        train_y = y[:split[0] * len(x)]
        valid_y = y[split[0] * len(x):split[1] * len(x)]
        test_y = y[split[1] * len(x):]

        train_xy = np.hstack([train_x, train_y])
        test_xy = np.hstack([test_x, test_y])
        valid_xy = np.hstack([valid_x, valid_y])

        self.cnn = TextCNN(sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, embedding_matrix, train_xy, test_xy, valid_xy)
        TextCNN.train_and_evaluate(batch_size, num_epochs)



class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, embedding_matrix, train_xy, test_xy, valid_xy):
            self.train_xy = train_xy
            self.test_xy = test_xy
            self.valid_xy = valid_xy

            # Placeholders for input, output and dropout
            self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]),
                                trainable=False, name="W")
                self.embedding_placeholder = embedding_matrix
                self.embedding_init = W.assign(self.embedding_placeholder)
                # print(self.embedding_init)
                # print(self.input_x)
                self.embedded = tf.nn.embedding_lookup(self.embedding_init, self.input_x)
                #print(self.embedded)
                self.embedded = tf.expand_dims(self.embedded, -1)
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W", dtype=tf.float32) # reshape W to reflect filters
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID", data_format='NHWC',
                        name="conv") # batch, height, width, channels
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # print(h)
                    # Max-pooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1], # add 1 due to padding
                        strides=[1, 1, 1, 1],
                        padding='VALID', # pad the input before pooling
                        name="pool")
                    # print(pooled)
                    pooled_outputs.append(pooled)
            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(axis=3, values=pooled_outputs)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
            # print(self.h_pool)
            # print(self.h_pool_flat)

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # Calculate mean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y, name= "losses")
                self.loss = tf.reduce_mean(losses)
            # Calculate Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            with tf.name_scope("precision/recall"):
                # print(self.predictions)
                # print(self.input_y)
                self.input_y_casted = tf.cast(tf.argmax(self.input_y, 1), tf.int64)
                TP = tf.count_nonzero(self.predictions * self.input_y_casted)
                #TN = tf.count_nonzero((self.predictions - 1) * (self.input_y_casted - 1))
                FP = tf.count_nonzero(self.predictions * (self.input_y_casted - 1))
                FN = tf.count_nonzero((self.predictions - 1) * self.input_y_casted)
                self.precision = TP / (TP + FP)
                self.recall = TP / (TP + FN)

    def train_and_evaluate(self, batch_size, num_epochs):
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3) # abstract this
                grads_and_vars = optimizer.compute_gradients(self.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                print("Writing to {}\n".format(out_dir))

                # Maintain summaries for various metrics to be plotted in tensorboard

                # Initial values for metrics
                loss_var = self.loss
                accuracy_var = self.accuracy
                precision_var = self.precision
                recall_var = self.recall

                loss_summary = tf.summary.scalar("loss", loss_var)
                acc_summary = tf.summary.scalar("accuracy", accuracy_var)
                precision_summary = tf.summary.scalar("precision", precision_var)
                recall_summary = tf.summary.scalar("recall", recall_var)

                train_summary_op = tf.summary.merge(
                    [loss_summary, acc_summary, precision_summary, recall_summary, grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                summary_op = tf.summary.merge([loss_summary, acc_summary, precision_summary, recall_summary])

                test_summary_dir = os.path.join(out_dir, "summaries", "test")
                test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

                valid_summary_dir = os.path.join(out_dir, "summaries", "valid")
                valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

                # Checkpointing to save model periodically during training
                checkpoint_dir = os.path.abspath(os.path.join("checkpoints", "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                # Tensorflow assumes this directory already exists so we need to create it
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables())

                sess.run(tf.global_variables_initializer())

                # Generates a batch iterator for a dataset.
                def batch_iter(batch_size, num_epochs, features_and_labels, shuffle=True):
                    num_batches_per_epoch = int((len(features_and_labels) - 1) / batch_size) + 1
                    for epoch in range(num_epochs):
                        # Shuffle the data at each epoch
                        if shuffle:
                            shuffle_indices = np.random.permutation(np.arange(len(features_and_labels)))
                            shuffled_data = features_and_labels[shuffle_indices]
                        else:
                            shuffled_data = features_and_labels
                        for batch_num in range(num_batches_per_epoch):
                            start_index = batch_num * batch_size
                            end_index = min((batch_num + 1) * batch_size, len(features_and_labels))
                            yield shuffled_data[start_index:end_index]
                # Single training step
                def train_step(x_batch, y_batch):
                    print('training...')
                    feed_dict = {
                        self.input_x: x_batch,
                        self.input_y: y_batch,
                        self.dropout_keep_prob: self.dropout_keep_prob
                    }
                    time_str = datetime.datetime.now().isoformat()

                    _, step, summary, loss, accuracy, precision, recall = sess.run(
                        [train_op, global_step, train_summary_op, self.loss, self.accuracy, self.precision, self.recall],
                        feed_dict)

                    print("{}: step {}, loss {:g}, acc {:g}, precision {:g}, recall {:g}".format(time_str, step, loss,
                                                                                                 accuracy, precision,
                                                                                                 recall))
                    train_summary_writer.add_summary(summary, step)
                    print('done')
                # Evaluates on validation or test set
                def eval_step(eval_batches, summary_writer):
                    print("evaluating...")
                    accuracies = []
                    losses = []
                    precisions = []
                    recalls = []
                    step = 0
                    for eval_batch in eval_batches:
                        x_batch, y_batch = eval_batch[:, :-2], eval_batch[:, -2:]
                        feed_dict = {
                            self.input_x: x_batch,
                            self.input_y: y_batch,
                            self.dropout_keep_prob: 1.0
                        }
                        step, accuracy, loss, precision, recall = (
                            sess.run([global_step, self.accuracy, self.loss, self.precision, self.recall], feed_dict))
                        accuracies.append(accuracy)
                        losses.append(loss)
                        precisions.append(precision)
                        recalls.append(recall)
                    accuracy = np.mean(accuracies)
                    loss = np.mean(losses)
                    precision = np.mean(precisions)
                    recall = np.mean(recalls)

                    summary = sess.run(summary_op, {loss_var: loss, accuracy_var: accuracy, precision_var: precision,
                                                    recall_var: recall})
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}, prec {:g}, rec {:g}".format(time_str, step, loss, accuracy,
                                                                                         precision, recall))
                    summary_writer.add_summary(summary, step)
                    print("done")

                train_batches = batch_iter(batch_size, num_epochs, self.train_xy)
                test_batches = batch_iter(batch_size, 1, self.test_xy)

                for batch in train_batches:
                    x_batch, y_batch = batch[:, :-2], batch[:, -2:]
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    # Save every 200 steps
                    if current_step % 200 == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                    # Run validation set every 75 steps
                    if (current_step % 75 == 0):
                        valid_batches = batch_iter(batch_size, 1, self.valid_xy)
                        eval_step(valid_batches, valid_summary_writer)
                # Evaluate the test set
                eval_step(test_batches, test_summary_writer)
