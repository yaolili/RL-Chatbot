# -*- coding: utf-8 -*-

"""
single turn conversation test procedure for BLUE [main]
checked by xurj
2018/5/31
"""

from __future__ import print_function

# get graphic card first
from sess_factory import SESS

# import python library
import os
import tensorflow as tf
from gensim.models import KeyedVectors
import time

# import our utils, config and model
from utils import make_batch_X, preProBuildWordVocab
from data_reader import Data_Reader
from config import *


def test():
    """
    Test procedure
    :return: no return
    """
    '''
    load dictionary
    '''
    word_vector = KeyedVectors.load_word2vec_format(pre_train_emb, binary=True)
    w2i, i2w, bias_init_vector = preProBuildWordVocab(word_count_threshold=word_count_threshold)

    '''
    build tf model, saver
    '''
    if TYPE_OF_OUR_MODEL in model_type:
        chat_model = model_proto(
            dim_wordvec=dim_wordvec,
            n_words=len(i2w),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_encode_lstm_step=n_encode_lstm_step,
            n_decode_lstm_step=n_decode_lstm_step,
            bias_init_vector=bias_init_vector,
            ex_batch=1)
    else:
        chat_model = model_proto(
            dim_wordvec=dim_wordvec,
            n_words=len(i2w),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_encode_lstm_step=n_encode_lstm_step,
            n_decode_lstm_step=n_decode_lstm_step,
            bias_init_vector=bias_init_vector)
    chat_model.build_generator()

    # build saver and restore checkpoint
    saver = tf.train.Saver()
    model_position = os.path.join(model_path, 'model-{}'.format(checkpoint))
    print('\n=== Use model', model_position, '===\n')
    saver.restore(SESS, model_position)

    '''
    open read and write path
    '''
    dr = Data_Reader(test_data_path)
    n_batch = dr.get_batch_num(batch_size)

    query_out = open(test_out_path + '.query', 'w')
    reference_out = open(test_out_path + '.reference', 'w')
    generate_out = open(test_out_path + '.generate', 'w')

    '''
    running batches
    '''
    start_time = time.time()
    print('total batch:', n_batch)

    for batch in range(n_batch):

        '''
        build data
        '''
        batch_x, batch_y, _, kw = dr.generate_training_batch_with_former(batch_size)
        if model_type == 'rev':
            tmp = batch_x
            batch_x = batch_y
            batch_y = tmp
        '''
        generation procedure for this batch
        '''
        feats = make_batch_X(batch_x, n_encode_lstm_step, dim_wordvec, word_vector)
        generated_words_index, _ = chat_model.test(SESS, (feats, kw), word_vector)

        # 提取信息，以供输出
        for idx, (x, y, gw) in enumerate(zip(batch_x, batch_y, generated_words_index)):

            words = []
            for index in gw:
                if index == 2:
                    break
                words.append(i2w[index].decode("utf-8"))
            sent = ' '.join(words)
            assert type(sent) == unicode

            query_out.write(x)
            query_out.write('\n')

            reference_out.write(y)
            reference_out.write('\n')

            generate_out.write(sent.encode('utf-8'))
            generate_out.write('\n')

        # 输出计数，让运行程序的人把握时间
        if batch % print_every == 0:
            print(batch)

    query_out.close()
    reference_out.close()
    generate_out.close()

    print('cost time: {}'.format(time.time() - start_time))


if __name__ == "__main__":
    test()
