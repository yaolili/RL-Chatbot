last_two_sentence = True

# path to training data
if last_two_sentence:
    training_data_path = 'data/weibo_data/lenmax22_formersents2.pkl'
else: 
    training_data_path = 'data/weibo_data/lenmax22_formersents1.pkl'

# path to pretrain embedding for query
# Notice: if you change the pretrain_emb size, the dim_wordvec should be changed too.
pretrain_emb = 'data/weibo_data/word_vector600.bin'

# path to all_words
all_words_path = 'data/weibo_data/all_words.txt'



CHECKPOINT = False  # reload 
training_type = 'normal' # 'normal' for seq2seq training, 'pg' for policy gradient
train_model_path = 'model/seq2seq/'
train_model_name = 'model-0'

reversed_model_path = 'model/Reversed/' # only uesed for Li's baseline
reversed_model_name = 'model-63'

start_epoch = 0
start_batch = 0
max_epochs = 10
batch_size = 100

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
WC_threshold = 5
reversed_WC_threshold = 5

# dialog simulation turns
MAX_TURNS = 10

# reward coefficient
alpha = 0.5
