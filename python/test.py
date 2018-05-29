# -*- coding: utf-8 -*-

"""
    single turn conversation test
"""


from __future__ import print_function

print('testing')

''' get graphic card first '''
from sess_factory import SESS

''' import python library '''
import os
import tensorflow as tf
from gensim.models import KeyedVectors
import time

''' import our utils, config and model '''
from utils import make_batch_X
import data_parser
from data_reader import Data_Reader
import config

# =====================================================
# Global Parameters
# =====================================================
model_path = os.path.join(config.model_path, config.checkpoint)
testing_data_path = config.test_data_path
output_path = config.test_out_path
word_count_threshold = config.WC_threshold

# =====================================================
# Train Parameters
# =====================================================
dim_wordvec = config.dim_wordvec
dim_hidden = config.dim_hidden

n_encode_lstm_step = config.n_encode_lstm_step
n_decode_lstm_step = config.n_decode_lstm_step

batch_size = config.batch_size


def test():
    """

    :return: no return
    """
    ''' load dictionary '''
    word_vector = KeyedVectors.load_word2vec_format(config.pretrain_emb, binary=True)
    w2i, i2w, bias_init_vector = data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)

    ''' build tf model, saver '''
    if 'alpha' in config.model_type:
        model = config.model_proto(
            dim_wordvec=dim_wordvec,
            n_words=len(i2w),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_encode_lstm_step=n_encode_lstm_step,
            n_decode_lstm_step=n_decode_lstm_step,
            bias_init_vector=bias_init_vector,
            ex_batch=1)
    else:
        model = config.model_proto(
            dim_wordvec=dim_wordvec,
            n_words=len(i2w),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_encode_lstm_step=n_encode_lstm_step,
            n_decode_lstm_step=n_decode_lstm_step,
            bias_init_vector=bias_init_vector)
    model.build_generator()

    saver = tf.train.Saver()
    print('\n=== Use model', model_path, '===\n')
    saver.restore(SESS, model_path)

    ''' open read and write path '''
    dr = Data_Reader(testing_data_path)
    n_batch = dr.get_batch_num(batch_size)

    query_out = open(output_path + '.query', 'w')
    reference_out = open(output_path + '.reference', 'w')
    generate_out = open(output_path + '.generate', 'w')

    ''' running batches '''
    start_time = time.time()

    for batch in range(n_batch):

        ''' build data '''
        batch_x, batch_y, _, kw = dr.generate_training_batch_with_former(batch_size)
        if config.model_type == 'rev':
            tmp = batch_x
            batch_x = batch_y
            batch_y = tmp
        feats = make_batch_X(batch_x, n_encode_lstm_step, dim_wordvec, word_vector)

        generated_words_index, _ = model.test(SESS, (feats, kw), word_vector)

        for idx, (x, y, gw) in enumerate(zip(batch_x, batch_y, generated_words_index)):

            words = []
            for index in gw:
                if index == 2:
                    break
                words.append(i2w[index].decode("utf-8"))
            sent = ' '.join(words)

            query_out.write(x)  # print with \n, but file is not, maybe wb and w different
            query_out.write("\n")

            reference_out.write(y)
            reference_out.write("\n")

            generate_out.write(sent.encode('utf-8'))
            generate_out.write("\n")

            cnt = batch_size * batch + idx
            if cnt % 1000 == 0:
                print(cnt)

    query_out.close()
    reference_out.close()
    generate_out.close()

    print('cost time: {}'.format(time.time() - start_time))


if __name__ == "__main__":
    test()
