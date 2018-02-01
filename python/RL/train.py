#-*- coding: utf-8 -*-

from __future__ import print_function

import os
import time
import sys
import copy

sys.path.append("python")
from model import Seq2Seq_chatbot
from utils import make_batch_X, make_batch_Y, get_pmi_kw, get_textRank_kw
from data_reader import Data_Reader
import data_parser
import config
import re

from gensim.models import KeyedVectors
from scipy import spatial
import tensorflow as tf
import numpy as np
import math


### Global Parameters ###
checkpoint = config.CHECKPOINT
training_data_path = config.rl_data_path
training_type = config.training_type
model_path = config.rl_model_path
model_name = config.rl_model_name

start_epoch = config.start_epoch
start_batch = config.start_batch
epochs = config.rl_epochs
batch_size = config.batch_size

### Train Parameters ###
learning_rate = config.learning_rate
dim_wordvec = config.dim_wordvec
dim_hidden = config.dim_hidden

n_encode_lstm_step = config.n_encode_lstm_step
n_decode_lstm_step = config.n_decode_lstm_step

word_count_threshold = config.WC_threshold

alpha = config.alpha
max_turns = cofig.MAX_TURNS


def index2sentence(generated_word_index, prob_logit, ixtoword):
    generated_words = []
    for cur_ind in generated_word_index:
        # pad 0, bos 1, eos 2, unk 3
        if cur_ind == 2: break
        
        '''
        # remove <unk> <pad> <bos> to second high prob. word
        if cur_ind <= 3:
            sort_prob_logit = sorted(prob_logit[i])
            curindex = np.where(prob_logit[i] == sort_prob_logit[-2])[0][0]
            count = 1
            while curindex <= 3:
                curindex = np.where(prob_logit[i] == sort_prob_logit[(-2)-count])[0][0]
                count += 1
            cur_ind = curindex
        '''
        generated_words.append(ixtoword[cur_ind])
    
    return generated_words


# TODO
def count_rewards(query, sess_reply, cur_depth, total_depth, cur_reward):
    total_loss = np.zeros([batch_size, n_decode_lstm_step])
    for i in range(batch_size):
        s1 = 1.0 - spatial.distance.cosine(query[i], keywords[i])
        s2 = 1.0 - spatial.distance.cosine(reply[i], keywords[i])
        total_loss[i, :] += alpha * s1 + (1 - alpha) * s2 + 1  # here puls 1 to exceed original reward 1
    return total_loss


def train():
    wordtoix, ixtoword, bias_init_vector = data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)
    word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)
    # ones_reward = np.ones([batch_size, n_decode_lstm_step])

    g1 = tf.Graph()
    default_graph = tf.get_default_graph() 
    with g1.as_default():
        model = Seq2Seq_chatbot(
                dim_wordvec=dim_wordvec,
                n_words=len(wordtoix),
                dim_hidden=dim_hidden,
                batch_size=batch_size,
                n_encode_lstm_step=n_encode_lstm_step,
                n_decode_lstm_step=n_decode_lstm_step,
                bias_init_vector=bias_init_vector,
                lr=learning_rate)
        train_op, loss, input_tensors, inter_value = model.build_model()
        word_vectors, generated_words, probs, embs = model.build_generator()
        
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))   
        sess = tf.InteractiveSession()
        saver = tf.train.Saver(max_to_keep=100)
        if checkpoint:
            print("Use Model {}.".format(model_name))
            saver.restore(sess, os.path.join(model_path, model_name))
            print("Model {} restored.".format(model_name))
        else:
            print("Restart training...")
            tf.global_variables_initializer().run()

         
    # TODO: figure out load_list
    dr = Data_Reader(training_data_path, cur_train_index=config.cur_train_index, load_list=config.load_list)

    # simulation
    for turn in range(2, max_turns):
        for epoch in range(start_epoch, epochs):
            n_batch = dr.get_batch_num(batch_size)
            sb = start_batch if epoch == start_epoch else 0
            for batch in range(sb, n_batch):
                start_time = time.time()

                # only batch_x is used as the seed for agent A to start the conversation
                batch_X, _ = dr.generate_training_batch(batch_size)

                current_feats = make_batch_X(
                                batch_X=copy.deepcopy(batch_X), 
                                n_encode_lstm_step=n_encode_lstm_step, 
                                dim_wordvec=dim_wordvec,
                                word_vector=word_vector)

                # rl action: generate batch_size sents
                action_word_indexs, action_probs, action_embs = sess.run([generated_words, probs, embs],
                    feed_dict={
                       word_vectors: current_feats
                    })
                action_word_indexs = np.array(action_word_indexs).reshape(batch_size, n_decode_lstm_step)
                action_probs = np.array(action_probs).reshape(batch_size, n_decode_lstm_step, -1)

                    # Here I use the last encoder hidden state as the representation of query
                    query = np.array(enc_feats['encode_states'][-1]).reshape(batch_size, dim_hidden)
                    keywords = np.array(enc_feats['keywords']).reshape(batch_size, dim_hidden)
                    
                    # To get the representation of reply, encode the reply sequence
                    generated_actions_list = []
                    for i in range(len(action_word_indexs)):
                        action = index2sentence(
                                    generated_word_index=action_word_indexs[i], 
                                    prob_logit=action_probs[i],
                                    ixtoword=ixtoword)
                        generated_actions_list.append(action)

                    action_feats = make_batch_X(
                                    batch_X=copy.deepcopy(generated_actions_list), 
                                    n_encode_lstm_step=n_encode_lstm_step, 
                                    dim_wordvec=dim_wordvec,
                                    word_vector=word_vector)

                    enc_feats = sess.run(encode_feats,
                                            feed_dict={
                                               word_vectors: action_feats
                                            })
                    reply = np.array(enc_feats['encode_states'][-1]).reshape(batch_size, dim_hidden)

                    # get reward given query, keyword, reply
                    rewards = count_rewards(query, keywords, reply)
        
                    feed_dict = {
                        input_tensors['word_vectors']: current_feats,
                        input_tensors['caption']: current_caption_matrix,
                        input_tensors['caption_mask']: current_caption_masks,
                        input_tensors['reward']: rewards
                    }
                    
                if training_type == 'normal':
                    feed_dict = {
                        input_tensors['word_vectors']: current_feats,
                        input_tensors['caption']: current_caption_matrix,
                        input_tensors['caption_mask']: current_caption_masks,
                        input_tensors['reward']: ones_reward
                    }

                if batch % 10 == 0:
                    _, loss_val = sess.run([train_op, loss], feed_dict = feed_dict)
                    print("Epoch: {}, batch: {}, loss: {}, Elapsed time: {}".format(epoch, batch, loss_val, time.time() - start_time))
                else:
                    _ = sess.run(train_op, feed_dict = feed_dict)

                if batch % 1000 == 0 and batch != 0:
                    print("Epoch {} batch {} is done. Saving the model ...".format(epoch, batch))
                    saver.save(sess, os.path.join(model_path, 'model-{}-{}'.format(epoch, batch)))

            print("Epoch ", epoch, " is done. Saving the model ...")
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

if __name__ == "__main__":
    train()
