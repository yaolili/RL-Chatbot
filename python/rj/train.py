# -*- coding: utf-8 -*-

"""
Train procedure [main]
"""

from __future__ import print_function

# get graphic card ###
import os
import tensorflow as tf
from sess_factory import SESS

# import python library ###
import time
from gensim.models import KeyedVectors
from collections import Counter

# import our utils, config and model ###
from utils import make_batch_X, make_batch_Y
import data_parser
from data_reader import Data_Reader
import config

model_proto = config.model_proto

# paths
training_data_path = config.training_data_path
valid_data_path = config.valid_data_path
pre_train_emb = config.pretrain_emb
b_reload = config.B_RELOAD
model_path = config.model_path
model_name = config.checkpoint
summary_dir = config.summary_dir

# training configuration
start_epoch = config.start_epoch
epochs = config.max_epochs
word_count_threshold = config.WC_threshold

# model hyper Parameters
learning_rate = config.learning_rate
dim_word_vec = config.dim_wordvec
dim_hidden = config.dim_hidden
n_encode_lstm_step = config.n_encode_lstm_step
n_decode_lstm_step = config.n_decode_lstm_step
batch_size = config.batch_size

# build dictionary
w2i, i2w, bias_init_vector = data_parser.preProBuildWordVocab(
    word_count_threshold=word_count_threshold)
word_vector = KeyedVectors.load_word2vec_format(pre_train_emb, binary=True)


def has_kw_action():
    return 'alpha' in config.model_type


if has_kw_action():
    kw_index_counter = Counter()


def run_model(dr, operate, summary_op, word2vector, _w2i, output):
    if has_kw_action():
        loss_val, summary, kw_ixs = operate(
            SESS, summary_op, dr, i2w, _w2i, word2vector, output)
        for kw_ix in kw_ixs:
            kw_index_counter.update(kw_ix)
        return loss_val, summary

    batch_x, batch_y, _, kw = dr.generate_training_batch_with_former(batch_size)
    if config.model_type == 'rev':
        tmp = batch_x
        batch_x = batch_y
        batch_y = tmp
    feats = make_batch_X(batch_x, n_encode_lstm_step, dim_word_vec, word2vector)
    caption_matrix, caption_masks = make_batch_Y(batch_y, _w2i, n_decode_lstm_step)

    loss_val, summary = operate(
        SESS, summary_op, (feats, kw, caption_matrix, caption_masks), word2vector)
    return loss_val, summary


def train():
    """
    Training procedure
    """
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    ''' build model, summary and saver'''
    model = model_proto(
        dim_wordvec=dim_word_vec,
        n_words=len(w2i),
        dim_hidden=dim_hidden,
        batch_size=batch_size,
        n_encode_lstm_step=n_encode_lstm_step,
        n_decode_lstm_step=n_decode_lstm_step,
        bias_init_vector=bias_init_vector,
        lr=learning_rate)
    if has_kw_action():
        model.build_generator(b_simulate=False)
        if 'alpha' in config.model_type or 'ks' in config.model_type:
            model.kw_stc_sim()
        loss, true_loss = model.build_model()
        train_entropy_summary = tf.summary.scalar('train_entropy', true_loss)
        valid_entropy_summary = tf.summary.scalar('valid_entropy', true_loss)
        train_loss_summary = tf.summary.scalar('train_loss', loss)
        valid_loss_summary = tf.summary.scalar('valid_loss', loss)
        train_summary = tf.summary.merge([train_loss_summary, train_entropy_summary])
        valid_summary = tf.summary.merge([valid_loss_summary, valid_entropy_summary])
    elif config.model_type == 'kw_sample':
        entropy, kw_entropy, loss = model.build_model()
        train_loss_summary = tf.summary.scalar('train_loss', loss)
        valid_loss_summary = tf.summary.scalar('valid_loss', loss)
        train_entropy_summary = tf.summary.scalar('train_entropy', entropy)
        valid_entropy_summary = tf.summary.scalar('valid_entropy', entropy)
        train_kw_entropy_summary = tf.summary.scalar('train_kw_entropy', kw_entropy)
        valid_kw_entropy_summary = tf.summary.scalar('valid_kw_entropy', kw_entropy)
        train_summary = tf.summary.merge(
            [train_loss_summary, train_entropy_summary, train_kw_entropy_summary])
        valid_summary = tf.summary.merge(
            [valid_loss_summary, valid_entropy_summary, valid_kw_entropy_summary])
    else:
        loss = model.build_model()
        train_summary = tf.summary.scalar('train_loss', loss)
        valid_summary = tf.summary.scalar('valid_loss', loss)

    variables = [v for v in tf.trainable_variables()]
    print('\n------- Variables begin -------\n')
    [print(v) for v in variables]
    print('\n------- Variables end -------\n')

    saver = tf.train.Saver(max_to_keep=100)
    writer = tf.summary.FileWriter(summary_dir, SESS.graph)

    ''' reload model or restart trainning '''
    print('\n------- Loading/Initialization begin -------\n')
    if b_reload:
        saver.restore(SESS, os.path.join(model_path, model_name))
        print("previous model {} in {} restored, continue training"
              .format(model_name, model_path))
    else:
        tf.global_variables_initializer().run()
        if has_kw_action():
            # SESS.run(model.op_kw_emb_init)
            # print('kw embeddings inited')
            restorer = tf.train.Saver()
            previous_model = '/local/WIP/xurj/RL-Chatbot-save/model/kw_sample/' + config.checkpoint
            restorer.restore(SESS, previous_model)
            print('pretrained model {} restored'.format(previous_model))
        elif 'kw_sample' in config.model_type:
            SESS.run(model.op_kw_emb_init)
            print('kw embeddings inited')
            restorer = tf.train.Saver(variables[:5] + variables[8:12])
            previous_model = '/local/WIP/xurj/RL-Chatbot-save/model/seq2seq/' \
                             + config.checkpoint
            restorer.restore(SESS, previous_model)
            print('pretrained model {} restored'.format(previous_model))
        print("Start brand new training...")
    print('\n------- Loading/Initialization end -------\n')

    ''' prepare data '''
    dr = Data_Reader(training_data_path)
    valid_dr = Data_Reader(valid_data_path)

    ''' running epoches '''
    n_batch = dr.get_batch_num(batch_size)
    print('data prepared.\ntraining batch number is {}'.format(n_batch))
    v_n_batch = valid_dr.get_batch_num(batch_size)
    print('valid batch number is {}'.format(v_n_batch))

    global kw_index_counter

    for epoch in range(start_epoch, epochs + 1):

        '''
        validation. begin at epoch -1
        '''
        print('\n-------\nvalidation epoch: {}\n-------\n'.format(epoch - 1))

        total_valid_loss = 0
        start_time = time.time()

        for batch in range(v_n_batch):
            start_time2 = time.time()
            loss_val, v_summary = run_model(
                valid_dr, model.valid, valid_summary, word_vector, w2i,
                output=batch % config.print_every == 0)

            writer.add_summary(v_summary, v_n_batch * epoch + batch)
            total_valid_loss += loss_val
            if (epoch * n_batch + batch) % config.print_every == 0:
                print("Valid Epoch: {}, batch: {}, total_loss: {}, Elapsed time: {}"
                      .format(epoch, batch, loss_val, time.time() - start_time2))

        if has_kw_action():
            print('*****************')
            for ix in kw_index_counter.most_common(33):
                print('kw: {} cnt: {} prob: {}'.format(
                    ix[0], ix[1],
                    float(ix[1]) / ((config.n_step + 1) * v_n_batch * batch_size)))
            print('*****************')
            kw_index_counter = Counter()
        print("valid epoch: {} average total_loss: {}, Elapsed time: {}"
              .format(epoch - 1, total_valid_loss / v_n_batch, time.time() - start_time))

        # valid before train, and valid times is train times + 1
        if epoch >= epochs:
            break

        '''
        training. begin at epoch 0
        '''
        print('\n-------\ntraining epoch: {}\n-------\n'.format(epoch))

        for batch in range(n_batch):
            start_time = time.time()
            loss_val, t_summary = run_model(
                dr, model.run, train_summary, word_vector, w2i,
                output=batch % config.print_every == 0)
            writer.add_summary(t_summary, n_batch * epoch + batch)
            if (epoch * n_batch + batch) % config.print_every == 0:
                print("Train Epoch: {}, batch: {}, total_loss: {}, Elapsed time: {}"
                      .format(epoch, batch, loss_val, time.time() - start_time))

        print("Epoch ", epoch, " is done. Saving the model ...")
        saver.save(SESS, os.path.join(model_path, 'model'), global_step=epoch)
        if has_kw_action():
            print('*****************')
            for ix in kw_index_counter.most_common(33):
                print('kw: {} cnt: {} prob: {}'.format(
                    ix[0], ix[1], float(ix[1]) / ((config.n_step + 1) * n_batch * batch_size)))
            print('*****************')
            kw_index_counter = Counter()


if __name__ == "__main__":
    train()
