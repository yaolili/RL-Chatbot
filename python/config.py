# coding=utf-8

from __future__ import print_function
import sys

print('\n------- configuration begin -------\n')

''' sys argument '''
try:
    model_type = sys.argv[1]
except:
    model_type = 'env'
try:
    start_epoch = int(sys.argv[2])
except:
    start_epoch = 0
try:
    debug_run = sys.argv[3]  # for test and train and simulate
except:
    debug_run = 'debug'
print('model type: {}\nstart at epoch: {}\ndebug or run: {}'
      .format(model_type, start_epoch, debug_run))
try:
    batch_size = int(sys.argv[4])
except:
    batch_size = 64
print('batch size: {}'.format(batch_size))

if start_epoch > 0:
    B_RELOAD = True
else:
    B_RELOAD = False
try:
    checkpoint = 'model-' + str(start_epoch - 1 if start_epoch > 0 else 14)  # for test and train and simulate
except:
    checkpoint = 'model-14'
print('reload: {}\ncheckpoint: {}'.format(B_RELOAD, checkpoint))

''' important argument '''

# control
print_every = 50
special = False

# vector size
dim_wordvec = 600
dim_hidden = 1000

# rl
n_step = 2
gama = 0.9

# kw
n_kw = 1000
empty_kw = '<empty>'.decode('utf-8')

''' output file location '''
local_dir = '/local/WIP/xurj/RL-Chatbot-save/'
summary_dir = local_dir + 'summary/' + model_type
model_path = local_dir + 'model/' + model_type
test_out_path = local_dir + 'test/' + model_type + '/' + checkpoint
simulate_out_path = local_dir + 'simulate/' + model_type + '/' + checkpoint

''' model import selection '''

if model_type == 'seq2seq' or model_type == 'rl' or model_type == 'rev':
    import model
    model_proto = model.Seq2Seq_chatbot
elif model_type == 'kw':
    import kw_model
    model_proto = kw_model.Kw_chatbot
elif model_type == 'kw_sample':
    import kw_sample_model
    model_proto = kw_sample_model.Kw_sample_chatbot
    kw_path = 'data/kw1000_vector600.pkl'
elif 'alpha' in model_type:
    import alpha_model
    model_proto = alpha_model.Chatbot
    kw_path = 'data/kw1000_vector600.pkl'
else:
    kw_path = 'data/kw1000_vector600.pkl'
    print('env mode')

''' path to dataset '''
test_raw_data_path = 'data/test_origin.txt.kw.pkl'
test_data_path = 'data/test.pkl'
single_test_data_path = 'data/test_origin.first.kw.pkl'

training_raw_data_path = 'data/train_origin.txt.kw.sf.unique.pkl'
valid_raw_data_path = 'data/valid_origin.txt.kw.sf.unique.pkl'

training_data_path = 'data/train.pkl'
valid_data_path = 'data/valid.pkl'

''' 附加情况 '''
if special:
    test_data_path = 'data/special/query.kw.pkl'
    test_out_path = 'test/special/' + model_type + '/' + checkpoint
if debug_run == 'debug':
    summary_dir = summary_dir + '_debug'
    model_path = model_path + '_debug'
    test_out_path = test_out_path + '_debug'
    simulate_out_path = simulate_out_path + '_debug'

''' no so changeable below '''

# path to pretrain embedding for query
# Notice: if you change the pre_train_emb size, the dim_word_vec should be changed too.
pretrain_emb = 'data/weibo_data/word_vector' + str(dim_wordvec) + '.bin'
print('load word embeddings {}'.format(pretrain_emb))

all_words_path = 'data/weibo_data/all_words.txt'
all_nouns_path = 'data/weibo_data/keywords_dict.1w.tok'
pmi_dict_path = 'data/weibo_data/pmi_dict.pkl'

start_batch = 0
max_epochs = 15
rl_epochs = 2  # for each depth, iterate N epochs

# training parameters
learning_rate = 0.0001

n_encode_lstm_step = 22 + 22
n_decode_lstm_step = 22

r_n_encode_lstm_step = 22
r_n_decode_lstm_step = 22

# data reader shuffle index list
load_list = False
index_list_file = 'data/weibo_data/shuffle_index_list'
cur_train_index = start_batch * batch_size

# word count threshold
WC_threshold = 10
reversed_WC_threshold = 10

# dialog simulation turns
MAX_TURNS = 5

# reward coefficient
last_two_sentence = True
alpha1 = 0.25
alpha2 = 0.25
alpha3 = 0.5

print('\n------- configuration end -------\n')
