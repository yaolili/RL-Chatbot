#-*- coding: utf-8 -*-

from __future__ import print_function

print('version control yao')

from gensim.models import KeyedVectors
from data_reader import Data_Reader
import data_parser
import os
import time
import tensorflow as tf
import numpy as np

import config
from model import Seq2Seq_chatbot
from utils import make_batch_X, make_batch_Y, index2sentence  # x

sess = tf.InteractiveSession()

### Global Parameters ###
training_data_path = config.training_data_path
valid_data_path = config.valid_data_path  # x
pretrain_emb = config.pretrain_emb
checkpoint = config.CHECKPOINT
model_path = config.train_model_path
model_name = config.train_model_name
start_epoch = config.start_epoch
summary_dir = 'summary'  # x

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
    
    single_vector, generated_words, probs, _ = model.build_generator()  # x

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
                valid_batch_X, valid_batch_Y = valid_dr.generate_training_batch(batch_size)
                
                valid_feats = make_batch_X(valid_batch_X, n_encode_lstm_step, dim_wordvec, word_vector) 
                valid_caption_matrix, valid_caption_masks = make_batch_Y(valid_batch_Y, wordtoix, n_decode_lstm_step)
               
                valid_loss, v_summary = sess.run(
                                        [tf_loss, valid_summary],
                                        feed_dict={
                                            word_vectors: valid_feats,
                                            tf_caption: valid_caption_matrix,
                                            tf_caption_mask:valid_caption_masks
                                        })
                writer.add_summary(v_summary, n_batch * epoch + batch)
                print("Epoch: {}, batch: {}, VALID loss: {}".format(epoch, batch, valid_loss))

                for i in range(0, len(valid_feats)):
                    if i > 4: break
                    cur_feats = valid_feats[i]
                    cur_feats = cur_feats.reshape(1, cur_feats.shape[0], -1)
                    print("shape:", cur_feats.shape)
                    generated_words_index, prob_logit = sess.run(
                                                    [generated_words, probs],
                                                    feed_dict={
                                                        single_vector: cur_feats,
                                                    })
                    
                    print("valid query: ", valid_batch_X[i])
                    print("golden reply: ", valid_batch_Y[i])
                    # stc = index2sentence(generated_words_index, prob_logit, ixtoword)
                    sent = ""
                    for index in generated_words_index:
                        sent += ixtoword[index].decode("utf-8")
                    print("Generated reply: ", sent)
                    print("***********************")
            # x z
            
            start_time = time.time()

            batch_X, batch_Y = dr.generate_training_batch(batch_size)
            current_feats = make_batch_X(batch_X, n_encode_lstm_step, dim_wordvec, word_vector)
            current_caption_matrix, current_caption_masks = make_batch_Y(batch_Y, wordtoix, n_decode_lstm_step)
            
            if batch % config.print_every == 0: # x 
                _, loss_val, t_summary = sess.run(
                        [train_op, tf_loss, train_summary],
                        feed_dict={
                            word_vectors: current_feats,
                            tf_caption: current_caption_matrix,
                            tf_caption_mask: current_caption_masks
                        })
                writer.add_summary(t_summary, n_batch * epoch + batch)  # x
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
