# coding=utf-8

import tensorflow as tf
import numpy as np
import time
import cPickle as pkl
import config


class Kw_sample_chatbot(object):
    def __init__(self, dim_wordvec, n_words, dim_hidden, batch_size, n_encode_lstm_step, n_decode_lstm_step,
                 bias_init_vector=None, lr=0.0001):
        self.dim_wordvec = dim_wordvec
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_words = n_words
        self.n_encode_lstm_step = n_encode_lstm_step
        self.n_decode_lstm_step = n_decode_lstm_step
        self.lr = lr

        # with tf.device("/cpu:0"):
        self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')
        self.lstm1 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        self.lstm2 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        self.encode_vector_W = tf.Variable(tf.random_uniform([dim_wordvec, dim_hidden], -0.1, 0.1),
                                           name='encode_vector_W')
        self.encode_vector_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_vector_b')
        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

        # keywords params
        self.keywords_W = tf.Variable(tf.random_uniform([dim_wordvec, dim_hidden], -0.1, 0.1), name='keywords_W')
        self.keywords_b = tf.Variable(tf.zeros([dim_hidden]), name='keywords_b')
        # for key word selection
        self.h_n_kw = config.n_kw
        self.v_KWemb = tf.get_variable('keyword_emb', shape=(self.h_n_kw, dim_wordvec),
                                       dtype=tf.float32, trainable=True)  # NOTICE trainable
        self.h_i2kw = {}
        self.h_kw2i = {}
        i2emb = []

        with open(config.kw_path, 'rb') as f:
            kw_emb = pkl.load(f)
        kws = kw_emb.keys()
        for kw in kws[:20]:
            assert type(kw) == unicode
            print(kw.encode('utf-8'))
        for i, kw in enumerate(kws):
            assert type(kw) == unicode
            self.h_i2kw[i] = kw
            self.h_kw2i[kw] = i
            i2emb.append(kw_emb[kw])
        assert len(i2emb) == self.h_n_kw
        self.op_kw_emb_init = self.v_KWemb.assign(np.array(i2emb))


    def build_model(self):
        word_vectors = tf.placeholder(tf.float32, [self.batch_size, self.n_encode_lstm_step, self.dim_wordvec])
        kw_index = tf.placeholder(tf.int32, [self.batch_size])
        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_decode_lstm_step + 1])
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_decode_lstm_step + 1])

        ''' re-construct context+query information '''
        word_vectors_flat = tf.reshape(word_vectors, [-1, self.dim_wordvec])
        wordvec_emb = tf.nn.xw_plus_b(word_vectors_flat, self.encode_vector_W,
                                      self.encode_vector_b)  # (batch_size*n_encode_lstm_step, dim_hidden)
        wordvec_emb = tf.reshape(wordvec_emb, [self.batch_size, self.n_encode_lstm_step, self.dim_hidden])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        loss = 0.0

        '''  Encoding Stage '''
        for i in range(0, self.n_encode_lstm_step):
            with tf.variable_scope("LSTM1", reuse=i > 0):
                output1, state1 = self.lstm1(wordvec_emb[:, i, :], state1)
            with tf.variable_scope("LSTM2", reuse=i > 0):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

        ''' Keyword Stage '''
        kw_info, kw_entropy = self.get_kw(output1, kw_index, 'model')

        ''' Decoding Stage '''
        for i in range(0, self.n_decode_lstm_step):
            # with tf.device("/cpu:0"):
            current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])
            with tf.variable_scope("LSTM1", reuse=True):
                output1, state1 = self.lstm1(kw_info, state1)
            with tf.variable_scope("LSTM2", reuse=True):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

            ''' label replies '''
            labels = tf.expand_dims(caption[:, i + 1], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat([indices, labels], 1)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

            ''' calculate cross entropy '''
            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
            cross_entropy = cross_entropy * caption_mask[:, i]
            current_loss = tf.reduce_sum(cross_entropy) / self.batch_size
            loss = loss + current_loss

        kw_entropy = tf.reduce_sum(kw_entropy) / self.batch_size
        total_loss = loss + kw_entropy

        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            # train_op = tf.train.RMSPropOptimizer(self.lr).minimize(total_loss)
            # train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(total_loss)
            optimizer = tf.train.AdamOptimizer(self.lr)
            train_op = optimizer.minimize(total_loss)
            '''
            grads8vars = optimizer.compute_gradients(total_loss)
            for g, v in grads8vars:
                print(g)
                print(v)
            '''

        self.train_op = train_op
        self.loss = loss
        self.kw_entropy = kw_entropy
        self.total_loss = total_loss

        self.word_vectors = word_vectors
        self.mp_kw_index = kw_index
        self.caption = caption
        self.caption_mask = caption_mask
        return loss, kw_entropy, total_loss

    def build_generator(self):
        word_vectors = tf.placeholder(tf.float32, [self.batch_size, self.n_encode_lstm_step, self.dim_wordvec])
        kw_index = tf.placeholder(tf.int32, [self.batch_size])

        ''' re-construct context+query information '''
        word_vectors_flat = tf.reshape(word_vectors, [-1, self.dim_wordvec])
        wordvec_emb = tf.nn.xw_plus_b(word_vectors_flat, self.encode_vector_W, self.encode_vector_b)
        wordvec_emb = tf.reshape(wordvec_emb, [self.batch_size, self.n_encode_lstm_step, self.dim_hidden])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        generated_words = []

        '''  Encoding Stage '''
        for i in range(0, self.n_encode_lstm_step):
            with tf.variable_scope("LSTM1", reuse=i > 0):
                output1, state1 = self.lstm1(wordvec_emb[:, i, :], state1)

            with tf.variable_scope("LSTM2", reuse=i > 0):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

        ''' Keyword Stage '''
        kw_info, _ = self.get_kw(output1, kw_index, 'generator')

        ''' Decoding Stage '''
        for i in range(0, self.n_decode_lstm_step):
            if i == 0:
                # with tf.device('/cpu:0'):
                current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([self.batch_size], dtype=tf.int64))
            with tf.variable_scope("LSTM1", reuse=True):
                output1, state1 = self.lstm1(kw_info, state1)
            with tf.variable_scope("LSTM2", reuse=True):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

            ''' label replies '''
            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)
            generated_words.append(max_prob_index)
            # with tf.device("/cpu:0"):
            current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)

        self.word_vectors = word_vectors
        self.gp_kw_index = kw_index
        self.generated_words = tf.stack(generated_words, axis=1)

    '''
    keyword generation
    '''

    def get_kw(self, output1, kw_index, father):

        with tf.variable_scope("get_kw"):
            logits = tf.layers.dense(output1,
                                     self.h_n_kw,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='kw_prob_dense')

        if father == 'generator':
            kw_index = tf.argmax(logits, axis=1)
            self.kw_index = kw_index
            kw_entropy = None
        else:
            kw_onehot = tf.one_hot(kw_index, self.h_n_kw)
            kw_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=kw_onehot)

        kw_emb = tf.nn.embedding_lookup(self.v_KWemb, kw_index)
        kw_info = tf.nn.xw_plus_b(kw_emb, self.keywords_W, self.keywords_b)

        return kw_info, kw_entropy

    def deal_with_kw(self, kw):
        assert type(kw) is list
        assert len(kw) == self.batch_size
        assert type(kw[0]) == unicode
        kw_index_batch = [self.h_kw2i[k] for k in kw]
        return kw_index_batch

    def run(self, sess, summary, args_tuple, _):

        feats, kw, caption_matrix, caption_masks = args_tuple
        kw_index_batch = self.deal_with_kw(kw)
        _, loss_val, t_summary = sess.run(
            [self.train_op, self.total_loss, summary],
            feed_dict={
                self.word_vectors: feats,
                self.mp_kw_index: np.array(kw_index_batch, dtype=np.int32),
                self.caption: caption_matrix,
                self.caption_mask: caption_masks
            })
        return loss_val, t_summary

    def valid(self, sess, summary, args_tuple, _):

        feats, kw, caption_matrix, caption_masks = args_tuple
        kw_index_batch = self.deal_with_kw(kw)
        loss_val, v_summary = sess.run(
            [self.total_loss, summary],
            feed_dict={
                self.word_vectors: feats,
                self.mp_kw_index: np.array(kw_index_batch, dtype=np.int32),
                self.caption: caption_matrix,
                self.caption_mask: caption_masks
            })
        return loss_val, v_summary

    def test(self, sess, args_tuple, _):

        feats, _ = args_tuple
        generated_words_index, kw_id = sess.run([self.generated_words, self.kw_index],
                                                feed_dict={self.word_vectors: feats})
        return generated_words_index, [self.h_i2kw[i].encode('utf-8') for i in kw_id]
