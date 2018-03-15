#-*- coding: utf-8 -*-

from __future__ import print_function

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import os
import time
import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors

sys.path.append("python")
from utils import make_batch_X, make_batch_Y, index2sentence  # x
from data_reader import Data_Reader
import data_parser
import config
import chardet


from kw_model import Kw_chatbot

sess = tf.InteractiveSession()  # x


### Global Parameters ###
training_data_path = config.training_data_path
valid_data_path = config.valid_data_path  # x
pretrain_emb = config.pretrain_emb
checkpoint = config.CHECKPOINT
model_path = config.kw_model_path
model_name = config.kw_model_name
start_epoch = config.start_epoch
summary_dir = 'summary_kw'  # x

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

    model = Kw_chatbot(
            dim_wordvec=dim_wordvec,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_encode_lstm_step=n_encode_lstm_step,
            n_decode_lstm_step=n_decode_lstm_step,
            bias_init_vector=bias_init_vector,
            lr=learning_rate)

    train_op, tf_loss, word_vectors, kw_vectors, tf_caption, tf_caption_mask, inter_value = model.build_model()
    single_vector, single_kw, generated_words, probs, _ = model.build_generator()  # x
    
    saver = tf.train.Saver(max_to_keep=100)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    
    # x a
    with tf.name_scope("train"):
        train_summary = tf.summary.scalar('loss', tf_loss)
    with tf.name_scope("valid"):
        valid_summary = tf.summary.scalar('loss', tf_loss)
    writer = tf.summary.FileWriter(summary_dir, sess.graph)
    # x z
    
    if checkpoint:
        print("Use Model {}.".format(model_name))
        saver.restore(sess, os.path.join(model_path, model_name))
        print("Model {} restored.".format(model_name))
    else:
        print("Restart training...")
        tf.global_variables_initializer().run()

    dr = Data_Reader(training_data_path)
    valid_dr = Data_Reader(valid_data_path)  # x

    for epoch in range(start_epoch, epochs):
        n_batch = dr.get_batch_num(batch_size)
        for batch in range(n_batch):
        
            # x a
            if batch % config.valid_every == 0:
                valid_batch_X, valid_batch_Y, _, valid_kw = valid_dr.generate_training_batch_with_former(batch_size)
                
                for i in range(len(valid_kw)):
                    # Notice: original kw[i] is string, but len(kw[i]) is calculated by bytes
                    # use split() to make sure one word
                    # after decode('utf-8'), kw[i] is unicode
                    valid_kw[i] = valid_kw[i].decode('utf-8')
                    valid_kw[i] = valid_kw[i].strip().split(" ")
                    assert len(valid_kw[i]) == 1, 'Length of kw is not 1'
                    valid_kw[i] = valid_kw[i][0]
                    
                    # FIXME: cannot hash kw[i] directly
                    a = valid_kw[i]
                    valid_kw[i] = word_vector[a] if a in word_vector else np.zeros(dim_wordvec)
                valid_current_kw = np.array(valid_kw)
                
                valid_feats = make_batch_X(valid_batch_X, n_encode_lstm_step, dim_wordvec, word_vector)
                valid_caption_matrix, valid_caption_masks = make_batch_Y(valid_batch_Y, wordtoix, n_decode_lstm_step)
                valid_loss, v_summary = sess.run(
                                            [tf_loss, valid_summary],
                                                feed_dict={
                                                    word_vectors: valid_feats,
                                                    kw_vectors: valid_current_kw,
                                                    tf_caption: valid_caption_matrix,
                                                    tf_caption_mask: valid_caption_masks
                                                })
                writer.add_summary(v_summary, n_batch * epoch + batch)
                print("Epoch: {}, batch: {}, VALID loss: {}".format(epoch, batch, valid_loss))
                
                for i in range(0, len(valid_feats)):
                    if i > 4: break
                    cur_kw = [valid_current_kw[i]]
                    cur_feats = valid_feats[i]
                    cur_feats = cur_feats.reshape(1, cur_feats.shape[0], -1)
                    generated_words_index, prob_logit = sess.run(
                                                    [generated_words, probs],
                                                    feed_dict={
                                                        single_vector: cur_feats,
                                                        single_kw: cur_kw,
                                                    })
                    
                    print("valid query: ", valid_batch_X[i])
                    print("golden reply: ", valid_batch_Y[i])
                    # stc = index2sentence(generated_words_index, prob_logit, ixtoword)
                    sent = ""
                    for index in generated_words_index:
                        sent += ixtoword[index].decode("utf-8")
                    print("Generated reply: ", sent.encode('utf-8'))
                    print("***********************")
            # x z
            
            start_time = time.time()

            # Here get keywords
            batch_X, batch_Y, _, kw = dr.generate_training_batch_with_former(batch_size)
            current_feats = make_batch_X(batch_X, n_encode_lstm_step, dim_wordvec, word_vector)
            current_caption_matrix, current_caption_masks = make_batch_Y(batch_Y, wordtoix, n_decode_lstm_step)

            # Deal with kw
            for i in range(len(kw)):
                # Notice: original kw[i] is string, but len(kw[i]) is calculated by bytes
                # use split() to make sure one word
                # after decode('utf-8'), kw[i] is unicode
                kw[i] = kw[i].decode('utf-8')
                kw[i] = kw[i].strip().split(" ")
                assert len(kw[i]) == 1, 'Length of kw is not 1'
                kw[i] = kw[i][0]
                
                # FIXME: cannot hash kw[i] directly
                a = kw[i]
                kw[i] = word_vector[a] if a in word_vector else np.zeros(dim_wordvec)
            current_kw = np.array(kw)

            if batch % config.print_every == 0:  # x
                _, loss_val, t_summary = sess.run(
                        [train_op, tf_loss, train_summary],
                        feed_dict={
                            word_vectors: current_feats,
                            kw_vectors: current_kw,
                            tf_caption: current_caption_matrix,
                            tf_caption_mask: current_caption_masks
                        })
                writer.add_summary(t_summary, n_batch * epoch + batch)  # x
                print("Epoch: {}, batch: {}, loss: {}, Elapsed time: {}".format(epoch, batch, loss_val, time.time() - start_time))
            else:
                _ = sess.run(train_op,
                             feed_dict={
                                word_vectors: current_feats,
                                kw_vectors: current_kw,
                                tf_caption: current_caption_matrix,
                                tf_caption_mask: current_caption_masks
                            })


        print("Epoch ", epoch, " is done. Saving the model ...")
        saver.save(sess, model_path, global_step=epoch)

if __name__ == "__main__":
    train()
