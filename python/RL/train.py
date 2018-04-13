#-*- coding: utf-8 -*-

from __future__ import print_function

import os
import time
import sys
reload(sys)                      
sys.setdefaultencoding('utf-8')
import copy

sys.path.append("python")
from rl_model import PolicyGradient_chatbot
from model import Seq2Seq_chatbot
from utils import make_batch_X, make_batch_Y, get_pmi_kw, get_textRank_kw, index2sentence
from data_reader import Data_Reader
import data_parser
import config
import re

from gensim.models import KeyedVectors
from scipy import spatial
import tensorflow as tf
import numpy as np
import math
from numpy import dot
from numpy.linalg import norm



### Global Parameters ###
training_data_path = config.training_data_path
reversed_model_path = config.reversed_model_path
reversed_model_name = config.reversed_model_name

pretrain_emb = config.pretrain_emb
checkpoint = config.CHECKPOINT
model_path = config.rl_model_path
model_name = config.rl_model_name
start_epoch = config.start_epoch
start_batch = config.start_batch

word_count_threshold = config.WC_threshold

### Train Parameters ###
learning_rate = config.learning_rate
dim_wordvec = config.dim_wordvec
dim_hidden = config.dim_hidden

n_encode_lstm_step = config.n_encode_lstm_step
n_decode_lstm_step = config.n_decode_lstm_step
r_n_encode_lstm_step = config.r_n_encode_lstm_step
r_n_decode_lstm_step = config.r_n_decode_lstm_step

epochs = config.max_epochs
batch_size = config.batch_size

discount = float(config.discount)
alpha1 = float(config.alpha1)
alpha2 = float(config.alpha2)
alpha3 = float(config.alpha3)
max_turns = config.MAX_TURNS

summary_dir = 'summary'

# here should be unicode
dull_set = [u"我 是 小 仙女", u"你 是 <unk>", u"我 也 想 你", u"我 在 <unk>", u"你 开心 就 好"]

ones_reward = np.ones([batch_size, n_decode_lstm_step])


def ease_of_answer_reward(sess, feats, input_tensors, action_feats, dull_matrix, dull_mask):
    dull_reward = np.zeros(batch_size)
    # Each action vector should calculate the reward of each dull_sentence in dull set
    for i, (cur_dull_matrix, cur_dull_mask) in enumerate(zip(dull_matrix, dull_mask)):
        d_feats = sess.run(feats,
                     feed_dict={
                        input_tensors['word_vectors']: action_feats,
                        input_tensors['caption']: cur_dull_matrix,
                        input_tensors['caption_mask']: cur_dull_mask,
                        input_tensors['reward']: ones_reward
                    })
        d_entropies = np.array(d_feats['entropies']).reshape(batch_size, n_decode_lstm_step)

        cur_len = len(dull_set[i].strip().split())
        dull_reward += np.sum(d_entropies, axis=1) / cur_len    
    dull_reward /= len(dull_set)
    return dull_reward


def semantic_coherence_rewards(forward_entropy, backward_entropy, forward_target, backward_target):
    forward_entropy = np.array(forward_entropy).reshape(batch_size, n_decode_lstm_step)
    backward_entropy = np.array(backward_entropy).reshape(batch_size, n_decode_lstm_step)
        
    forward_reward = []
    backward_reward = []
    semantic_reward = []
    for i in range(batch_size):
        forward_len = len(forward_target[i].split())
        backward_len = len(backward_target[i].split())
        # assert forward_len > 0 and backward_len > 0, "Empty forward_target or backward_target"
        if not (forward_len > 0 and backward_len > 0):
            print("i: {}".format(i))
            print("forward_target[i]: {}".format(forward_target[i]))
            print("backward_target[i]: {}".format(backward_target[i]))
            exit()
        forward_reward.append(np.sum(forward_entropy[i]) / forward_len)
        backward_reward.append(np.sum(backward_entropy[i]) / backward_len)
        # print("forward_reward {}: {}".format(i, forward_reward[i]))
        # print("backward_reward {}: {}".format(i, backward_reward[i]))
        semantic_reward.append(-1. * (forward_reward[i] + backward_reward[i]))
    return semantic_reward


def info_flow_reward(sess, word_vectors, encode_feats, action_feats, states):
    cur_turn_state = sess.run(encode_feats,
                     feed_dict={
                        word_vectors: action_feats,
                    })
    # last encode hidden state
    cur_turn_state = cur_turn_state['encode_states']
    states.append(cur_turn_state)
    # FIXME: if there is no former consecutive turns, assign a positive reward 100
    if len(states) < 3:
        return [100.] * batch_size
    last_turn_state = states[-3]
    information_reward = []
    for i in range(batch_size):
        cosine_sim = abs(1. - spatial.distance.cosine(last_turn_state[i], cur_turn_state[i]))
        information_reward.append(-1. * math.log(cosine_sim))
    return information_reward



def total_reward(dull_reward, information_reward, semantic_reward):
    # print("dull_reward: ", dull_reward)
    # print("information_reward: ", information_reward)
    # print("semantic_reward: ", semantic_reward)
    
    dull_reward = alpha1 * np.array(dull_reward)
    information_reward = alpha2 * np.array(information_reward)
    semantic_reward = alpha3 * np.array(semantic_reward)
    all_reward = (dull_reward + information_reward + semantic_reward) 
    
    # Future work: scale
    # rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    
    all_reward = all_reward.reshape(all_reward.shape+(1,))
    all_reward = np.tile(all_reward, n_decode_lstm_step)
    return all_reward



def train():
    # Cannot remove "GLOBAL" as the "dull_set" will be changed as global variable
    global dull_set

    wordtoix, ixtoword, bias_init_vector = data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)
    word_vector = KeyedVectors.load_word2vec_format(pretrain_emb, binary=True)

    # Prepare for ease of answering 
    origin_dull_matrix, origin_dull_mask = make_batch_Y(
                                batch_Y=dull_set, 
                                wordtoix=wordtoix, 
                                n_decode_lstm_step=n_decode_lstm_step)
                                
    # batch normalization, len(dull_set) * batch * n_decode_lstm_step
    dull_matrix = []
    dull_mask = []
    for i, (cp, cp_m) in enumerate(zip(origin_dull_matrix, origin_dull_mask)):
        cur_dull_matrix = np.asarray([cp for _ in range(batch_size)])
        cur_dull_mask = np.asarray([cp_m for _ in range(batch_size)])
        dull_matrix.append(cur_dull_matrix)
        dull_mask.append(cur_dull_mask)
        

    g1 = tf.Graph()
    g2 = tf.Graph()
    default_graph = tf.get_default_graph() 
    with g1.as_default():
        model = PolicyGradient_chatbot(
                dim_wordvec=dim_wordvec,
                n_words=len(wordtoix),
                dim_hidden=dim_hidden,
                batch_size=batch_size,
                n_encode_lstm_step=n_encode_lstm_step,
                n_decode_lstm_step=n_decode_lstm_step,
                bias_init_vector=bias_init_vector,
                lr=learning_rate)
        train_op, loss, input_tensors, feats = model.build_model()
        word_vectors, generated_words, encode_feats, decode_feats = model.build_generator()

        sess = tf.InteractiveSession()
        saver = tf.train.Saver(max_to_keep=100)
        if checkpoint:
            print("RL baseline Use Model {}.".format(model_name))
            saver.restore(sess, os.path.join(model_path, model_name))
            print("RL baseline Model {} restored.".format(model_name))
        else:
            print("RL baseline restart training...")
            tf.global_variables_initializer().run()

    with g2.as_default():
        reversed_model = Seq2Seq_chatbot(
            dim_wordvec=dim_wordvec,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_encode_lstm_step=r_n_encode_lstm_step,
            n_decode_lstm_step=r_n_decode_lstm_step,
            bias_init_vector=bias_init_vector,
            lr=learning_rate)
        _, _, r_word_vectors, caption, caption_mask, reverse_inter = reversed_model.build_model()
        sess2 = tf.InteractiveSession()
        saver2 = tf.train.Saver()
        saver2.restore(sess2, os.path.join(reversed_model_path, reversed_model_name))
        print("Reversed model {} restored.".format(reversed_model_name))
         
    dr = Data_Reader(training_data_path, cur_train_index=config.cur_train_index, load_list=config.load_list)
    
    with tf.name_scope("train"):
        train_summary = tf.summary.scalar('loss', loss)
    with tf.name_scope("valid"):
        valid_summary = tf.summary.scalar('loss', loss)
    writer = tf.summary.FileWriter(summary_dir, sess.graph)
    
    
    # simulation
    for epoch in range(start_epoch, epochs):
        n_batch = dr.get_batch_num(batch_size)
        sb = start_batch if epoch == start_epoch else 0
        for batch in range(sb, n_batch):
            start_time = time.time()

            print("Epoch {} batch {}: ".format(epoch, batch))
            
            # batch_x is used as the seed for agent A to start the conversation
            batch_X, batch_Y, former, _ = dr.generate_training_batch_with_former(batch_size)
            
            # for i in range(batch_size):
                # print("query {}: {}".format(i, batch_X[i]))
            
            current_feats = make_batch_X(
                                batch_X=batch_X, 
                                n_encode_lstm_step=n_encode_lstm_step, 
                                dim_wordvec=dim_wordvec,
                                word_vector=word_vector)

            current_caption_matrix, current_caption_masks = make_batch_Y(batch_Y, wordtoix, n_decode_lstm_step)
            
            states = []
            
            expected_reward = 0.
            
            next_feats = copy.deepcopy(current_feats)
            next_former = former
            
            for i in range(max_turns):
                print("current turn {}: ".format(i))
                
                t1 = time.time()
                # generate reply
                current_feats_states, action_word_indexs, action_decode_feats = sess.run([encode_feats, generated_words, decode_feats],
                    feed_dict={
                       word_vectors: next_feats
                    })
                
                # states used for info_flow_reward, Only the first step we regard the query as first agent's state
                if i == 0:
                    states.append(current_feats_states['encode_states'])
                    
                action_word_indexs = np.array(action_word_indexs).reshape(batch_size, n_decode_lstm_step)           

                t_m = time.time()
                # print("Before make batch x & y, Elapsed time: {}".format(t_m - t1))
                
                # To speed up, we don't use make_batch_X & make_batch_Y from utils
                actions = [] # Each element is Chinese sentence
                action_feats = [] # make_batch_X
                action_caption_matrix = [] # make_batch_Y
                action_caption_masks = [] # make_batch_Y
                
                for j in range(batch_size):
                    action = []
                    cur_feats = []
                    # add bos 
                    cur_caption = [0] * (n_decode_lstm_step + 1)
                    cur_caption[0] = wordtoix['<bos>']
                    cur_mask = np.zeros(n_decode_lstm_step + 1)
                    cur_mask[0] = 1.
                    for k in range(len(action_word_indexs[j])):
                        index = action_word_indexs[j][k]
                        if k < n_decode_lstm_step - 1:
                            cur_caption[k+1] = index
                        word = ixtoword[index].decode("utf-8")
                        action.append(word)
                        if word not in word_vector:
                            cur_feats.append(np.zeros(dim_wordvec))
                        else: cur_feats.append(word_vector[word])
                        if index == 2: break # eos
                    
                    # make_batch_X
                    if len(action) > n_encode_lstm_step:
                        cur_feats = cur_feats[:n_encode_lstm_step]
                    else:
                        for _ in range(len(action), n_encode_lstm_step):
                            cur_feats.append(np.zeros(dim_wordvec))
                    action_feats.append(cur_feats)
                    
                    # make_batch_Y
                    pos = len(action) if len(action) < n_decode_lstm_step else n_decode_lstm_step - 1
                    cur_caption[pos] = wordtoix['<eos>']
                    
                    action_caption_matrix.append(cur_caption)
                    cur_mask[:pos+2] = 1.
                    action_caption_masks.append(cur_mask)

                    
                    actions.append(" ".join(action))
                    # if j < 5:
                        # print("generated {} reply: {}".format(j, " ".join(action)))
                    
                action_feats = np.asarray(action_feats)
                action_caption_matrix = np.asarray(action_caption_matrix)
                action_caption_masks = np.asarray(action_caption_masks)
                
                # print("action_feats:{}".format(action_feats))
                # print("action_caption:{}".format(action_caption_matrix))
                # print("action_mask:{}".format(action_caption_masks))
                
                t2 = time.time()
                print("make_batch_X & Y, Elapsed time: {}".format(t2 - t_m))
                
                ################ ease of answering ################                              

                dull_reward = ease_of_answer_reward(sess, feats, input_tensors, action_feats, dull_matrix, dull_mask)
                
                t3 = time.time()
                print("dull reward, Elapsed time: {}".format(t3 - t2))
                
                ################ information flow ################
                information_reward = info_flow_reward(sess, word_vectors, encode_feats, action_feats, states)
                
                t4 = time.time()
                print("information reward, Elapsed time: {}".format(t4 - t3))

                ################ semantic coherence ################
                forward_inter = sess.run(feats,
                                 feed_dict={
                                    input_tensors['word_vectors']: next_feats,
                                    input_tensors['caption']: action_caption_matrix,
                                    input_tensors['caption_mask']: action_caption_masks,
                                    input_tensors['reward']: ones_reward
                                })
                forward_entropies = forward_inter['entropies']
                
                # for j in range(batch_size):
                    # print("caption(next_former) {}: {}".format(j, next_former[j]))
                    # print("query(actions) {}: {}".format(j, actions[j]))
                
                former_caption_matrix, former_caption_masks = make_batch_Y(
                                                                batch_Y=next_former, 
                                                                wordtoix=wordtoix, 
                                                                n_decode_lstm_step=n_decode_lstm_step)
                r_action_feats = action_feats
                backward_inter = sess2.run(reverse_inter,
                                 feed_dict={
                                    r_word_vectors: r_action_feats,
                                    caption: former_caption_matrix,
                                    caption_mask: former_caption_masks
                                })
                backward_entropies = backward_inter['entropies']

                semantic_reward = semantic_coherence_rewards(forward_entropies, backward_entropies, actions, next_former)
                
                t5 = time.time()
                print("semantic reward, Elapsed time: {}".format(t5 - t4))
 
                reward = total_reward(dull_reward, information_reward, semantic_reward)
                # print("reward is: {}".format(reward))

                # next_batch_X = former + actions
                next_batch_X = []
                for j in range(batch_size):
                    actions[j] = actions[j].replace("<eos>", "").strip() 
                    next_batch_X.append(former[j] + " " + actions[j])
                    # if j < 5:
                        # print("next {} query: {}".format(j, former[j] + " " + actions[j]))
                    # remember to update former!
                    former[j] = actions[j]
                    
                next_former = actions
                print("**********************************")
                

                next_feats = make_batch_X(
                                batch_X=next_batch_X, 
                                n_encode_lstm_step=n_encode_lstm_step, 
                                dim_wordvec=dim_wordvec,
                                word_vector=word_vector)
                                
                                
                expected_reward += pow(discount, i) * reward
            
            # NOTICE: loss = -log p() * reward , so must be negative
            expected_reward = 1.0 / max_turns * expected_reward
            
            
            feed_dict = {
                    input_tensors['word_vectors']: current_feats,
                    input_tensors['caption']: current_caption_matrix,
                    input_tensors['caption_mask']: current_caption_masks,
                    input_tensors['reward']: expected_reward
                }

            if batch % config.print_every == 0:
                _, loss_val, t_summary = sess.run(
                        [train_op, loss, train_summary], feed_dict = feed_dict)
                writer.add_summary(t_summary, n_batch * epoch + batch)  # x
                print("Epoch: {}, batch: {}, loss: {}, Elapsed time: {}".format(epoch, batch, loss_val, time.time() - start_time))
            else:
                _ = sess.run(train_op, feed_dict = feed_dict)
                print("Epoch: {}, batch: {}, Elapsed time: {}".format(epoch, batch, time.time() - start_time))
            
            
        print("Epoch ", epoch, " is done. Saving the model ...")
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

if __name__ == "__main__":
    train()
