# -*- coding: utf-8 -*-

"""
Training procedure [main]
checked by xurj
2018/5/30
"""

from __future__ import print_function

# get graphic card
from sess_factory import SESS

# import python library
import os
import tensorflow as tf
from gensim.models import KeyedVectors
from collections import Counter
import time

# import our utils, config and model
from config import *
from data_reader import Data_Reader
from utils import make_batch_X, make_batch_Y, preProBuildWordVocab


def count_kw():
    """
    judge whether we need to count keyword selection
    :return: True or False
    """
    return TYPE_OF_OUR_MODEL in model_type


def run_model(dr, operate, summary_op, word2vector, w2i, i2w, kw_index_counter, output):
    """
    train or valid
    :param kw_index_counter: counter usage of keywords
    :param dr: data reader
    :param operate: train or valid operation
    :param summary_op: train or valid summary
    :param word2vector:
    :param w2i: word to index
    :param i2w: index to word
    :param output: whether to print model debug message
    :return: loss result and summary result
    """
    if count_kw():
        # run model
        loss_val, summary, kw_ixs = operate(SESS, summary_op, dr, i2w, w2i, word2vector, output)

        # update keyword index counter
        for kw_ix in kw_ixs:
            kw_index_counter.update(kw_ix)

    else:
        # prepare data
        batch_x, batch_y, _, kw = dr.generate_training_batch_with_former(batch_size)
        if model_type == 'rev':
            tmp = batch_x
            batch_x = batch_y
            batch_y = tmp
        assert type(batch_x[0]) == unicode
        assert type(batch_y[0]) == unicode
        assert type(kw[0]) == unicode
        feats = make_batch_X(batch_x, n_encode_lstm_step, dim_wordvec, word2vector)
        caption_matrix, caption_masks = make_batch_Y(batch_y, w2i, n_decode_lstm_step)

        # run model
        loss_val, summary = operate(SESS, summary_op, (feats, kw, caption_matrix, caption_masks), word2vector)

    return loss_val, summary


def train():
    """
    Training procedure
    """

    if count_kw():
        kw_index_counter = Counter()
    else:
        kw_index_counter = None

    '''
    build dictionary
    '''
    w2i, i2w, bias_init_vector = preProBuildWordVocab(
        word_count_threshold=word_count_threshold)
    word_vector = KeyedVectors.load_word2vec_format(pre_train_emb, binary=True)

    '''
    build model, summary and saver
    '''
    # build model
    chat_model = model_proto(
        dim_wordvec=dim_wordvec,
        n_words=len(w2i),
        dim_hidden=dim_hidden,
        batch_size=batch_size,
        n_encode_lstm_step=n_encode_lstm_step,
        n_decode_lstm_step=n_decode_lstm_step,
        bias_init_vector=bias_init_vector,
        lr=learning_rate)

    if count_kw():
        chat_model.build_generator(b_simulate=False)
        chat_model.kw_stc_sim()
        loss, true_loss = chat_model.build_model()

        train_entropy_summary = tf.summary.scalar('train_entropy', true_loss)
        valid_entropy_summary = tf.summary.scalar('valid_entropy', true_loss)
        train_loss_summary = tf.summary.scalar('train_loss', loss)
        valid_loss_summary = tf.summary.scalar('valid_loss', loss)
        train_summary = tf.summary.merge([train_loss_summary, train_entropy_summary])
        valid_summary = tf.summary.merge([valid_loss_summary, valid_entropy_summary])
    elif model_type == 'kw_sample':
        entropy, kw_entropy, loss = chat_model.build_model()

        train_loss_summary = tf.summary.scalar('train_loss', loss)
        valid_loss_summary = tf.summary.scalar('valid_loss', loss)
        train_entropy_summary = tf.summary.scalar('train_entropy', entropy)
        valid_entropy_summary = tf.summary.scalar('valid_entropy', entropy)
        train_kw_entropy_summary = tf.summary.scalar('train_kw_entropy', kw_entropy)
        valid_kw_entropy_summary = tf.summary.scalar('valid_kw_entropy', kw_entropy)
        train_summary = tf.summary.merge([train_loss_summary, train_entropy_summary, train_kw_entropy_summary])
        valid_summary = tf.summary.merge([valid_loss_summary, valid_entropy_summary, valid_kw_entropy_summary])
    else:
        loss = chat_model.build_model()

        train_summary = tf.summary.scalar('train_loss', loss)
        valid_summary = tf.summary.scalar('valid_loss', loss)

    # build summary
    writer = tf.summary.FileWriter(summary_dir, SESS.graph)

    # build saver
    saver = tf.train.Saver(max_to_keep=100)

    # check trainable print variables
    print('\n------- VARS -------')
    variables = [v for v in tf.trainable_variables()]
    [print(v) for v in variables]

    '''
    reload chat_model or restart training
    '''
    print('\n------- Reloading / Initialization -------')
    if checkpoint >= 0:
        # reload
        saver.restore(SESS, os.path.join(model_path, 'model-{}'.format(checkpoint)))
        print("previous chat_model {} in {} restored, continue training".format(checkpoint, model_path))
    else:
        # init
        tf.global_variables_initializer().run()
        if count_kw():
            restorer = tf.train.Saver()
            restorer.restore(SESS, previous_model)
            print('pretrained chat_model {} restored'.format(previous_model))
        elif 'kw_sample' in model_type:
            SESS.run(chat_model.op_kw_emb_init)
            print('kw embeddings inited')

            restorer = tf.train.Saver(variables[:5] + variables[8:12])
            restorer.restore(SESS, previous_model)
            print('pretrained chat_model {} restored'.format(previous_model))
        print("Start brand new training...")

    '''
    load data
    '''
    dr = Data_Reader(training_data_path)
    valid_dr = Data_Reader(valid_data_path)

    ''' running epochs '''
    n_batch = dr.get_batch_num(batch_size)
    v_n_batch = valid_dr.get_batch_num(batch_size)
    print('data prepared.\ntraining batch number is {}\nvalid batch number is {}'.format(n_batch, v_n_batch))

    for epoch in range(checkpoint + 1, max_epochs + 1):
        '''
        validation. begin at epoch -1
        '''
        print('\n------- validation epoch: {} -------\n'.format(epoch - 1))

        total_valid_loss = 0
        start_time = time.time()

        for batch in range(v_n_batch):
            start_time2 = time.time()
            loss_result, v_summary = run_model(
                valid_dr, chat_model.valid, valid_summary, word_vector, w2i, i2w, kw_index_counter,
                output=batch % print_every == 0)

            writer.add_summary(v_summary, v_n_batch * epoch + batch)
            total_valid_loss += loss_result
            if batch % print_every == 0:
                print("Valid Epoch: {}, batch: {}, total_loss: {}, Elapsed time: {}"
                      .format(epoch - 1, batch, loss_result, time.time() - start_time2))

        print("valid epoch: {} average total_loss: {}, Elapsed time: {}"
              .format(epoch - 1, total_valid_loss / v_n_batch, time.time() - start_time))

        if count_kw():
            # output keyword selection situation
            print('*****************')
            for ix in kw_index_counter.most_common(20):
                print('kw: {} cnt: {} prob: {}'.format(
                    ix[0], ix[1], float(ix[1]) / ((n_step + 1) * v_n_batch * batch_size)))
            print('*****************')
            kw_index_counter = Counter()  # reset keyword counter for next operation

        # valid comes first, and valid-times = train-times + 1
        if epoch >= max_epochs:
            break

        '''
        training. begin at epoch 0
        '''
        print('\n-------\ntraining epoch: {}\n-------\n'.format(epoch))

        for batch in range(n_batch):
            start_time = time.time()
            loss_result, t_summary = run_model(
                dr, chat_model.run, train_summary, word_vector, w2i, i2w, kw_index_counter,
                output=batch % print_every == 0)

            writer.add_summary(t_summary, n_batch * epoch + batch)
            if batch % print_every == 0:
                print("Train Epoch: {}, batch: {}, total_loss: {}, Elapsed time: {}"
                      .format(epoch, batch, loss_result, time.time() - start_time))

        print("Epoch {} is done. Saving the chat_model ...".format(epoch))
        saver.save(SESS, os.path.join(model_path, 'model'), global_step=epoch)

        if count_kw():
            # output keyword selection situation
            print('*****************')
            for ix in kw_index_counter.most_common(33):
                print('kw: {} cnt: {} prob: {}'.format(
                    ix[0], ix[1], float(ix[1]) / ((n_step + 1) * n_batch * batch_size)))
            print('*****************')
            kw_index_counter = Counter()


if __name__ == "__main__":
    train()
