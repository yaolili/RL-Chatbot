#-*- coding: utf-8 -*-

from __future__ import print_function

from gensim.models import KeyedVectors
from data_reader import Data_Reader
import data_parser
import config

from model import Seq2Seq_chatbot

import tensorflow as tf
import numpy as np

import os
import time

from utils import make_batch_X, make_batch_Y


### Global Parameters ###
training_data_path = config.training_data_path
pretrain_emb = config.pretrain_emb
checkpoint = config.CHECKPOINT
model_path = config.reversed_model_path
model_name = config.reversed_model_name
start_epoch = config.start_epoch

word_count_threshold = config.WC_threshold

### Train Parameters ###
learning_rate = config.learning_rate
dim_wordvec = config.dim_wordvec
dim_hidden = config.dim_hidden

n_encode_lstm_step = config.n_encode_lstm_step
n_decode_lstm_step = config.n_decode_lstm_step

epochs = config.max_epochs
batch_size = config.batch_size


def train():
    wordtoix, ixtoword, bias_init_vector = data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)
    word_vector = KeyedVectors.load_word2vec_format(pretrain_emb, binary=True)

    model = Seq2Seq_chatbot(
            dim_wordvec=dim_wordvec,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_encode_lstm_step=n_encode_lstm_step,
            n_decode_lstm_step=n_decode_lstm_step,
            bias_init_vector=bias_init_vector,
            lr=learning_rate)

    train_op, tf_loss, word_vectors, tf_caption, tf_caption_mask, inter_value = model.build_model()

    saver = tf.train.Saver(max_to_keep=100)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.InteractiveSession()
    
    if checkpoint:
        print("Use Model {}.".format(model_name))
        saver.restore(sess, os.path.join(model_path, model_name))
        print("Model {} restored.".format(model_name))
    else:
        print("Restart training...")
        tf.global_variables_initializer().run()

    dr = Data_Reader(training_data_path)

    for epoch in range(start_epoch, epochs):
        n_batch = dr.get_batch_num(batch_size)
        for batch in range(n_batch):
            start_time = time.time()

            # reverse X and Y
            batch_Y, batch_X = dr.generate_training_batch(batch_size)
            current_feats = make_batch_X(batch_X, n_encode_lstm_step, dim_wordvec, word_vector)
            current_caption_matrix, current_caption_masks = make_batch_Y(batch_Y, wordtoix, n_decode_lstm_step)

            if batch % 100 == 0:
                _, loss_val = sess.run(
                        [train_op, tf_loss],
                        feed_dict={
                            word_vectors: current_feats,
                            tf_caption: current_caption_matrix,
                            tf_caption_mask: current_caption_masks
                        })
                print("Epoch: {}, batch: {}, loss: {}, Elapsed time: {}".format(epoch, batch, loss_val, time.time() - start_time))
            else:
                _ = sess.run(train_op,
                             feed_dict={
                                word_vectors: current_feats,
                                tf_caption: current_caption_matrix,
                                tf_caption_mask: current_caption_masks
                            })


        print("Epoch ", epoch, " is done. Saving the model ...")
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

if __name__ == "__main__":
    train()
