#-*- coding: utf-8 -*-

from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import cPickle as pkl

from gensim.models import KeyedVectors
import data_parser
import config

from model import Seq2Seq_chatbot
import tensorflow as tf
import numpy as np

import re
import sys
import time

import utils
from data_reader import Data_Reader
from utils import make_batch_X, make_batch_Y, index2sentence  # x

sess = tf.InteractiveSession()

#=====================================================
# Global Parameters
#=====================================================
default_model_path = config.test_model_path
testing_data_path = config.test_data_path if len(sys.argv) <= 2 else sys.argv[2]
output_path = config.test_out_path if len(sys.argv) <= 3 else sys.argv[3]

word_count_threshold = config.WC_threshold

#=====================================================
# Train Parameters
#=====================================================
dim_wordvec = config.dim_wordvec
dim_hidden = config.dim_hidden

n_encode_lstm_step = config.n_encode_lstm_step
n_decode_lstm_step = config.n_decode_lstm_step

batch_size = config.batch_size

def test(model_path=default_model_path):

    word_vector = KeyedVectors.load_word2vec_format(config.pretrain_emb, binary=True)

    wordtoix, ixtoword, bias_init_vector = data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)

    model = config.test_model_proto(
            dim_wordvec=dim_wordvec,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_encode_lstm_step=n_encode_lstm_step,
            n_decode_lstm_step=n_decode_lstm_step,
            bias_init_vector=bias_init_vector)

    word_vectors, caption_tf, probs, _ = model.build_generator()

    global sess

    saver = tf.train.Saver()
    try:
        print('\n=== Use model', model_path, '===\n')
        saver.restore(sess, model_path)
    except:
        print('\nUse default model\n')
        saver.restore(sess, default_model_path)

    with open(output_path, 'w') as fout:
        generated_sentences = []
        bleu_score_avg = [0., 0.]
        dr = Data_Reader(config.test_data_path)
        n_batch = dr.get_batch_num(batch_size)
        for batch in range(n_batch):
            batch_X, batch_Y = dr.generate_training_batch(batch_size)
            feats = make_batch_X(batch_X, n_encode_lstm_step, dim_wordvec, word_vector)
            for idx, (x, y) in enumerate(zip(batch_X, batch_Y)):
                    
                generated_words_index, prob_logit = sess.run([caption_tf, probs], feed_dict={word_vectors: [feats[idx]]})
            
                # print("test query: ", x)
                # print("golden reply: ", y)
                sent = ""
                for index in generated_words_index:
                    sent += ixtoword[index].decode("utf-8")
                # print("Generated reply: ", sent.encode('utf-8'))
                # print("***********************")
                fout.write("test query: " + x)
                fout.write("\n")
                fout.write("golden reply: " + y)
                fout.write("\n")
                fout.write("Generated reply: " + sent.encode('utf-8'))
                fout.write("\n")
                fout.write("***********************")
                fout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test(model_path=sys.argv[1])
    else:
        test()
