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
dull_set = [u"哦哦哦 好 的", u"嘿嘿", u"嘻嘻 嘻", u"么 么 哒", u"啊啊啊 啊啊啊 啊", u"那 你 很 棒棒 呦", u"哈哈哈哈 哈哈哈", u"厉害 了", u"啧啧 啧", u"是 滴", u"怎么 啦"]

ones_reward = np.ones([batch_size, n_decode_lstm_step])


def ease_of_answer_reward(sess, feats, input_tensors, action_feats, dull_matrix, dull_mask):
    dull_reward = []
    # Each action vector should calculate the reward of each dull_sentence in dull set
    for vector in action_feats:
        action_batch_X = np.array([vector for _ in range(batch_size)])
        d_feats = sess.run(feats,
                     feed_dict={
                        input_tensors['word_vectors']: action_batch_X,
                        input_tensors['caption']: dull_matrix,
                        input_tensors['caption_mask']: dull_mask,
                        input_tensors['reward']: ones_reward
                    })
        d_entropies = np.array(d_feats['entropies']).reshape(batch_size, n_decode_lstm_step)
        cur_loss = 0.
        for i in range(batch_size):
            cur_len = len(dull_set[i].strip().split())
            if cur_len == 0: break
            cur_loss += np.sum(d_entropies[i]) / cur_len

        d_loss = 1. / len(dull_set) * cur_loss
        dull_reward.append(d_loss)

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
    cur_turn_state = cur_turn_state['encode_states'][-1]
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
    if len(dull_set) > batch_size:
        dull_set = dull_set[:batch_size]
    else:
        for _ in range(len(dull_set), batch_size):
            dull_set.append('')
    dull_matrix, dull_mask = make_batch_Y(
                                batch_Y=dull_set, 
                                wordtoix=wordtoix, 
                                n_decode_lstm_step=n_decode_lstm_step)

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
                                batch_X=copy.deepcopy(batch_X), 
                                n_encode_lstm_step=n_encode_lstm_step, 
                                dim_wordvec=dim_wordvec,
                                word_vector=word_vector)

            current_caption_matrix, current_caption_masks = make_batch_Y(batch_Y, wordtoix, n_decode_lstm_step)
            
            states = []
            
            expected_reward = 0.
            
            next_feats = copy.deepcopy(current_feats)
            next_former = copy.deepcopy(former)
            
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
                    states.append(current_feats_states['encode_states'][-1])
                    
                action_word_indexs = np.array(action_word_indexs).reshape(batch_size, n_decode_lstm_step)                    
                
                actions = []
                for j in range(len(action_word_indexs)):
                    action = []
                    for index in action_word_indexs[j]:
                        action.append(ixtoword[index].decode("utf-8"))
                        if index == 2: break # eos
                    # print("generated {} reply: {}".format(j, " ".join(action)))
                    actions.append(" ".join(action))
                
                t2 = time.time()
                print("Before reward calculate, Elapsed time: {}".format(t2 - t1))
                
                ################ ease of answering ################
                action_feats = make_batch_X(
                                batch_X=copy.deepcopy(actions), 
                                n_encode_lstm_step=n_encode_lstm_step, 
                                dim_wordvec=dim_wordvec,
                                word_vector=word_vector)
                                

                dull_reward = ease_of_answer_reward(sess, feats, input_tensors, action_feats, dull_matrix, dull_mask)
                
                t3 = time.time()
                print("dull reward, Elapsed time: {}".format(t3 - t2))
                
                ################ information flow ################
                information_reward = info_flow_reward(sess, word_vectors, encode_feats, action_feats, states)
                
                t4 = time.time()
                print("information reward, Elapsed time: {}".format(t4 - t3))

                ################ semantic coherence ################
                action_caption_matrix, action_caption_masks = make_batch_Y(
                                                                batch_Y=copy.deepcopy(actions),
                                                                wordtoix=wordtoix, 
                                                                n_decode_lstm_step=n_decode_lstm_step)


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
                                                                batch_Y=copy.deepcopy(next_former), 
                                                                wordtoix=wordtoix, 
                                                                n_decode_lstm_step=n_decode_lstm_step)
                r_action_feats = make_batch_X(
                                batch_X=copy.deepcopy(actions), 
                                n_encode_lstm_step=r_n_encode_lstm_step, 
                                dim_wordvec=dim_wordvec,
                                word_vector=word_vector)
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
                    # print("next {} query: {}".format(j, former[j] + " " + actions[j]))
                    # remember to update former!
                    former[j] = copy.deepcopy(actions[j])
                    
                next_former = copy.deepcopy(actions)
                # print("**********************************")
                

                next_feats = make_batch_X(
                                batch_X=copy.deepcopy(next_batch_X), 
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
