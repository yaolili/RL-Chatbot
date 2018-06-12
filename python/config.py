# coding=utf-8

"""
Configuration module
checked by xurj
2018/5/30
"""

from __future__ import print_function
import argparse

TYPE_OF_OUR_MODEL = 'rlcw'

'''
parse command argument
'''
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', help='size of batch',
                    type=int, default=128)
parser.add_argument('-c', '--checkpoint', help='use which checkpoint for test or continue training? (usually 0-14)',
                    type=int, default=-1)
parser.add_argument('-d', '--debug_on', help='turn on to use debug mode (使输出数据位置和运行模式完全隔离，避免混淆)',
                    action='store_true')
parser.add_argument('-m', '--model_type', help='type of model',
                    choices=['env', 'seq2seq', 'rev', 'kw_sample', TYPE_OF_OUR_MODEL], default='env')
parser.add_argument('-p', '--print_every', help='output message every N batch',
                    type=int, default=50)
parser.add_argument('-s', '--n_step', help=TYPE_OF_OUR_MODEL + '\'s generation turns (config会自行修正，不用手动减1了)',
                    type=int, default=3)
args = parser.parse_args()

batch_size = args.batch_size
checkpoint = args.checkpoint
debug_on = args.debug_on
model_type = args.model_type
print_every = args.print_every
n_step = args.n_step - 1

'''
control
'''
# training parameters
learning_rate = 0.0001
max_epochs = 15

n_encode_lstm_step = 22 + 22
n_decode_lstm_step = 22

# simulation parameters
simulate_length = 10
overlapping_threshold = 0.8

# word count threshold
word_count_threshold = 10

# dims
dim_wordvec = 600
dim_hidden = 1000

# rl
decay_factor = 0.9

# kw
n_kw = 1000  # number of keyword
empty_kw = '<empty>'.decode('utf-8')  # representation of no keyword

'''
model import selection
'''
if model_type == 'seq2seq' or model_type == 'rev':
    import model

    model_proto = model.Seq2Seq_chatbot
elif model_type == 'kw_sample':
    import kw_sample_model

    model_proto = kw_sample_model.Kw_sample_chatbot
    kw_path = 'data/kw1000_vector600.pkl'
    previous_model = '/local/WIP/xurj/RL-Chatbot-save/model/seq2seq/model-14'
elif TYPE_OF_OUR_MODEL in model_type:
    import rlcw_model

    model_proto = rlcw_model.Chatbot
    kw_path = 'data/kw1000_vector600.pkl'
    previous_model = '/local/WIP/xurj/RL-Chatbot-save/model/kw_sample/model-14'
else:
    kw_path = 'data/kw1000_vector600.pkl'
    print('env mode')

'''
input path
'''
# train
training_raw_data_path = 'data/train_origin.txt.kw.sf.unique.pkl'
training_data_path = 'data/train.pkl'

# valid
valid_data_path = 'data/valid.pkl'
valid_raw_data_path = 'data/valid_origin.txt.kw.sf.unique.pkl'

# test
test_raw_data_path = 'data/test_origin.txt.kw.pkl'
test_data_path = 'data/test.pkl'
single_test_data_path = 'data/test_origin.first.kw.pkl'

# data reader shuffle index list
index_list_file = 'data/weibo_data/shuffle_index_list'

# word and embedding
pre_train_emb = 'data/weibo_data/word_vector' + str(dim_wordvec) + '.bin'
all_words_path = 'data/weibo_data/all_words.txt'
all_nouns_path = 'data/weibo_data/keywords_dict.1w.tok'
# pmi_dict_path = 'data/weibo_data/pmi_dict.pkl'

'''
output path
'''
LOCAL_DIR = '/local/WIP/xurj/RL-Chatbot-save/'

summary_dir = LOCAL_DIR + 'summary/' + model_type
model_path = LOCAL_DIR + 'model/' + model_type
test_out_path = LOCAL_DIR + 'test/' + model_type
simulate_out_path = LOCAL_DIR + 'simulate/' + model_type

# debug模式下隔离输出数据
if debug_on:
    summary_dir += '_debug'
    model_path += '_debug'
    test_out_path += '_debug'
    simulate_out_path += '_debug'

test_out_path += '/model-' + str(checkpoint)
simulate_out_path += '/model-' + str(checkpoint)

'''
confirm configuration
'''
print('\n------- CONF -------\n'
      'model type: {}\nbatch size: {}\nfrom checkpoint: {}\ndebug: {}\nprint every: {}'
      .format(model_type, batch_size, checkpoint, debug_on, print_every))
