import model


last_two_sentence = True

valid_every = 5000
print_every = 1000
test_model_proto = model.Seq2Seq_chatbot
test_model_path = 'model/Seq2Seq/model-14'
test_out_path = 'test/seq2seq-14'

# path to training data
training_data_path = 'data2/train_origin.txt.kw.pkl'
reverseed_data_path = 'data2/train_origin.txt.kw.pkl'

# path to test data
test_data_path = 'data2/test_origin.txt.kw.pkl'
valid_data_path = 'data2/valid_origin.txt.kw.pkl'

# path to pretrain embedding for query
# Notice: if you change the pretrain_emb size, the dim_wordvec should be changed too.
pretrain_emb = 'data/weibo_data/word_vector600.bin'


all_words_path = 'data/weibo_data/all_words.txt'
all_nouns_path = 'data/weibo_data/keywords_dict.1w.tok'
pmi_dict_path = 'data/weibo_data/pmi_dict.pkl'


CHECKPOINT = False  # reload 
training_type = 'normal' # 'normal' for seq2seq training, 'pg' for policy gradient
train_model_path = 'model/Seq2Seq/'
train_model_name = 'model-14'

reversed_model_path = 'model/Reversed/' # only uesed for Li's baseline
reversed_model_name = 'model-14'

kw_model_path = 'model/Kw/'
kw_model_name = 'model-0'

rl_model_path = 'model/RL/'
rl_model_name = 'model-0'

start_epoch = 0
start_batch = 0
max_epochs = 15
rl_epochs = 2  # for each depth, iterate N epochs
batch_size = 32

# training parameters
learning_rate = 0.0001
dim_wordvec = 600
dim_hidden = 1000

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
alpha1 = 0.25
alpha2 = 0.25
alpha3 = 0.5
