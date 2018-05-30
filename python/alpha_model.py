# coding=utf-8

from __future__ import print_function
import tensorflow as tf
import numpy as np
import cPickle as pkl
import copy
import time

import utils
import config

print('[IMPORT MODEL] MODEL version ALPHA')


class Chatbot(object):
    """
    keyword based planned conversation model
    """

    def __init__(self, dim_wordvec, n_words, dim_hidden, batch_size,
                 n_encode_lstm_step, n_decode_lstm_step, bias_init_vector=None,
                 lr=0.0001, n_step=config.n_step, ex_batch=3):
        self.h_dim_wordvec = dim_wordvec
        self.h_dim_hidden = dim_hidden
        self.h_batch_size = batch_size
        self.h_ex_batch_size = ex_batch * batch_size
        self.h_n_words = n_words
        self.h_n_encode_lstm_step = n_encode_lstm_step
        self.h_n_decode_lstm_step = n_decode_lstm_step
        self.h_lr = lr
        self.h_n_step = n_step

        # with tf.device("/cpu:0"):
        self.v_word2vector = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        self.v_lstm1 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        self.v_lstm2 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False)

        self.v_encode_vector_W = tf.Variable(tf.random_uniform([dim_wordvec, dim_hidden], -0.1, 0.1),
                                             name='encode_vector_W')
        self.v_encode_vector_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_vector_b')

        self.v_embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.v_embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.v_embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

        # keywords params
        self.v_keywords_W = tf.Variable(tf.random_uniform([dim_wordvec, dim_hidden], -0.1, 0.1), name='keywords_W')
        self.v_keywords_b = tf.Variable(tf.zeros([dim_hidden]), name='keywords_b')
        self.v_kw_lstm = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False)

        # for key word selection
        self.h_n_kw = config.n_kw
        self.v_keyword2vector = tf.get_variable(
            'keyword_emb', shape=(self.h_n_kw, dim_wordvec), dtype=tf.float32)  # NOTICE trainable
        self.h_i2kw = {}
        self.h_kw2i = {}
        # i2emb = []

        with open(config.kw_path, 'rb') as f:
            kw_emb = pkl.load(f)
        kws = kw_emb.keys()  # pickle不同次load顺序一致
        for kw in kws[:20]:
            assert type(kw) == unicode
            print(kw.encode('utf-8'))
        for i, kw in enumerate(kws):
            assert type(kw) == unicode
            self.h_i2kw[i] = kw
            self.h_kw2i[kw] = i
            # i2emb.append(kw_emb[kw])
        # assert len(i2emb) == self.h_n_kw
        # self.op_kw_emb_init = self.v_keyword2vector.assign(np.array(i2emb))

    def build_model(self):
        """
        build training part of this model
        responsible for probability gathering
        calculate entropy without gradient as part of reward
        calculate total_loss
        act as a single graph (can be run by SESS.run)
        use sampled keyword by build_generator
        :return: total_loss and ordinary cross entropy for tensorboard summary
        """
        self.mp_batchx = tf.placeholder(tf.float32,
                                        [self.h_ex_batch_size, self.h_n_encode_lstm_step, self.h_dim_wordvec])
        self.mp_rewards = tf.placeholder(tf.float32, [self.h_ex_batch_size])
        self.mp_caption = tf.placeholder(tf.int32, [self.h_ex_batch_size, self.h_n_decode_lstm_step + 1])
        self.mp_caption_mask = tf.placeholder(tf.float32, [self.h_ex_batch_size, self.h_n_decode_lstm_step + 1])

        word_vectors_flat = tf.reshape(self.mp_batchx, [-1, self.h_dim_wordvec])
        wordvec_emb = tf.nn.xw_plus_b(word_vectors_flat, self.v_encode_vector_W,
                                      self.v_encode_vector_b)  # (batch_size*n_encode_lstm_step, dim_hidden)
        wordvec_emb = tf.reshape(wordvec_emb, [self.h_ex_batch_size, self.h_n_encode_lstm_step, self.h_dim_hidden])

        state1 = tf.zeros([self.h_ex_batch_size, self.v_lstm1.state_size])
        state2 = tf.zeros([self.h_ex_batch_size, self.v_lstm2.state_size])
        padding = tf.zeros([self.h_ex_batch_size, self.h_dim_hidden])

        '''  Encoding Stage '''
        for i in range(0, self.h_n_encode_lstm_step):
            with tf.variable_scope("LSTM1", reuse=True):
                output1, state1 = self.v_lstm1(wordvec_emb[:, i, :], state1)
            with tf.variable_scope("LSTM2", reuse=True):
                output2, state2 = self.v_lstm2(tf.concat([padding, output1], 1), state2)

        ''' Keyword Stage '''
        kw_info, self.mp_kw_index, _, self.mr_kw_probs \
            , self.mp_kw_emb_history, self.mr_kw_avg_sim_log = self.get_kw(output1, True)

        ''' Decoding Stage '''
        entropies = tf.zeros([self.h_ex_batch_size])
        for i in range(0, self.h_n_decode_lstm_step):
            # with tf.device("/cpu:0"):
            current_embed = tf.nn.embedding_lookup(self.v_word2vector, self.mp_caption[:, i])
            with tf.variable_scope("LSTM1", reuse=True):
                output1, state1 = self.v_lstm1(kw_info, state1)
            with tf.variable_scope("LSTM2", reuse=True):
                output2, state2 = self.v_lstm2(tf.concat([current_embed, output1], 1), state2)
            labels = tf.expand_dims(self.mp_caption[:, i + 1], 1)
            indices = tf.expand_dims(tf.range(0, self.h_ex_batch_size), 1)
            concat_indices = tf.concat([indices, labels], 1)
            onehot_labels = tf.sparse_to_dense(concat_indices, tf.stack([self.h_ex_batch_size, self.h_n_words]), 1.0,
                                               0.0)
            logit_words = tf.nn.xw_plus_b(output2, self.v_embed_word_W, self.v_embed_word_b)

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
            cross_entropy = cross_entropy * self.mp_caption_mask[:, i]
            entropies += cross_entropy

        ''' Loss and Training Stage '''
        length = tf.reduce_sum(self.mp_caption_mask, axis=-1)
        self.mr_entropy = entropies / length
        self.mr_entropy = tf.stop_gradient(self.mr_entropy)
        # [ex_batch_size]
        if 'noent' in config.model_type:
            print('no entropy here!')
            self.multi_reward = self.mp_rewards
        else:
            self.multi_reward = self.mp_rewards - self.mr_entropy
        reward_avg = tf.reduce_mean(tf.reshape(self.multi_reward, [3, self.h_batch_size]), axis=0)
        self.reward_avg = tf.concat([reward_avg, reward_avg, reward_avg], axis=0)
        self.reward = self.multi_reward - self.reward_avg
        loss = tf.reduce_sum(-tf.log(self.mr_kw_probs) * self.reward) / self.h_ex_batch_size
        true_loss = tf.reduce_sum(entropies) / self.h_ex_batch_size

        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            # train_op = tf.train.RMSPropOptimizer(self.lr).minimize(total_loss)
            # train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(total_loss)
            optimizer = tf.train.AdamOptimizer(self.h_lr)
            train_op = optimizer.minimize(loss)

            grads8vars = optimizer.compute_gradients(loss)
            print('[INFORMATION: gradient of defined total_loss]')
            [print(g, v) for g, v in grads8vars]
            print('[INFORMATION END]')

        self.train_op = train_op
        self.loss = loss
        self.true_loss = true_loss
        return loss, true_loss

    def build_generator(self, b_simulate=True):
        """
        build generating part of this model
        calculate keyword similarity as part of reward
        do not calculate total_loss
        act as a single graph (can be run by SESS.run)
        sample keyword
        :return: no return
        """
        self.gp_batchx = tf.placeholder(tf.float32,
                                        [self.h_ex_batch_size, self.h_n_encode_lstm_step, self.h_dim_wordvec])
        word_vectors_flat = tf.reshape(self.gp_batchx, [-1, self.h_dim_wordvec])
        wordvec_emb = tf.nn.xw_plus_b(word_vectors_flat, self.v_encode_vector_W, self.v_encode_vector_b)
        wordvec_emb = tf.reshape(wordvec_emb, [self.h_ex_batch_size, self.h_n_encode_lstm_step, self.h_dim_hidden])

        state1 = tf.zeros([self.h_ex_batch_size, self.v_lstm1.state_size])
        state2 = tf.zeros([self.h_ex_batch_size, self.v_lstm2.state_size])
        padding = tf.zeros([self.h_ex_batch_size, self.h_dim_hidden])

        '''  Encoding Stage '''
        for i in range(0, self.h_n_encode_lstm_step):
            with tf.variable_scope("LSTM1", reuse=i > 0):
                output1, state1 = self.v_lstm1(wordvec_emb[:, i, :], state1)
            with tf.variable_scope("LSTM2", reuse=i > 0):
                output2, state2 = self.v_lstm2(tf.concat([padding, output1], 1), state2)

        ''' Keyword Stage '''
        self.gr_kw_info, self.gr_kw_index, self.gr_kw_emb, self.gr_probs \
            , self.gp_kw_emb_history, self.gr_kw_avg_sim_log = self.get_kw(output1, reuse=None,
                                                                           b_simulate=b_simulate)

        ''' Decoding Stage '''
        generated_words = []
        for i in range(0, self.h_n_decode_lstm_step):
            if i == 0:
                # with tf.device('/cpu:0'):
                current_embed = tf.nn.embedding_lookup(self.v_word2vector,
                                                       tf.ones([self.h_ex_batch_size], dtype=tf.int64))
            with tf.variable_scope("LSTM1", reuse=True):
                # here I replace "placeholder" with "kw_info"
                output1, state1 = self.v_lstm1(self.gr_kw_info, state1)
            with tf.variable_scope("LSTM2", reuse=True):
                output2, state2 = self.v_lstm2(tf.concat([current_embed, output1], 1), state2)
            logit_words = tf.nn.xw_plus_b(output2, self.v_embed_word_W, self.v_embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)

            # with tf.device("/cpu:0"):
            current_embed = tf.nn.embedding_lookup(self.v_word2vector, max_prob_index)

            ''' generated sentence sample gathering '''
            generated_words.append(max_prob_index)

        self.gr_words = tf.stack(generated_words, axis=1)

    '''
    keyword generation
    '''

    def get_kw(self, output1, reuse=None, b_simulate=False):
        """
        keyword getting process
        act as part of graph in build_model and build_generator
        1 placeholders here: kw_index (only for build_model because only sample once)
        :param output1: last hidden state after context+query passing through first encoder layer
        :param reuse: tf variable unify reuse management, True for build_model, None for generator
                        (because we build generator first)
        :param b_simulate: set true when simulate.
        :return: keyword related information.
                    note that we do not create class attribute (self....) here in case of naming conflict
        """
        output1 = tf.stop_gradient(output1)  # 遮蔽梯度
        """ choose new keyword and update kw lstm """
        with tf.variable_scope("get_kw", reuse=reuse):
            logits = tf.layers.dense(output1,
                                     self.h_n_kw,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='kw_prob_dense',
                                     activation=tf.nn.tanh)
            prob = tf.nn.softmax(logits)
            if reuse:  # build_model set reuse=True
                kw_index = tf.placeholder(dtype=tf.int64, shape=[self.h_ex_batch_size, 1])
            else:  # build_generator set reuse=False
                if b_simulate:
                    kw_index_1d = tf.argmax(logits, axis=1)
                    kw_index = tf.expand_dims(kw_index_1d, axis=1)
                    # kw_index = tf.multinomial(logits, 1)
                else:
                    kw_index = tf.multinomial(logits, 1)
            kw_emb = tf.nn.embedding_lookup(self.v_keyword2vector, tf.reshape(kw_index, [self.h_ex_batch_size]))
            # [self.h_ex_batch_size, 1]
            kw_batch_loc = tf.constant([[e] for e in range(self.h_ex_batch_size)], dtype=tf.int64)
            # [self.h_ex_batch_size, 2]
            kw_indices = tf.concat([kw_batch_loc, kw_index], axis=-1)
            # [self.h_ex_batch_size]
            kw_probs = tf.gather_nd(prob, kw_indices)

        kw_info = tf.nn.xw_plus_b(kw_emb, self.v_keywords_W, self.v_keywords_b)

        kw_emb_history = tf.placeholder(tf.float32, [None, self.h_ex_batch_size, self.h_dim_wordvec])
        kw_emb_3d = tf.expand_dims(kw_emb, axis=0)
        kw_sims = self.cosine_similarity(kw_emb_3d, kw_emb_history)
        kw_avg_sim = tf.reduce_mean(kw_sims, axis=0)
        kw_avg_sim_log = tf.log((kw_avg_sim + 1.) / 2 + 1e-33)

        return kw_info, kw_index, kw_emb, kw_probs, kw_emb_history, kw_avg_sim_log

    '''
    similarity part
    '''

    @staticmethod
    def cosine_similarity(batch_a, batch_b):
        """
        calculate cosine similarity. it can work in batches. only last dimension will be reduced.
        only act as part of graph in any places if you like
        :param batch_a: tensor whose shape with shape [(..., batch_size,) dim_size], at least 1 dims
        :param batch_b: tensor whose shape with shape [(..., batch_size,) dim_size], must have same dimension as batch_a
        :return: cosine similarity result with shape [(..., batch_size)]
        """
        numerator = tf.reduce_sum(batch_a * batch_b, axis=-1)
        denominator1 = tf.reduce_sum(batch_a * batch_a, axis=-1)
        denominator2 = tf.reduce_sum(batch_b * batch_b, axis=-1)
        result = numerator / tf.sqrt(denominator1 * denominator2 + 1e-20)
        return result

    def kw_stc_sim(self):
        """
        calculate similarity between keyword and sentences
        3 place holder here
        act as a single graph without any variables (can be run by SESS.run)
        :return: no return
        """
        self.sp_kw_embs = tf.placeholder(
            tf.float32, [self.h_n_step + 1, self.h_ex_batch_size, self.h_dim_wordvec])
        self.sp_queries = tf.placeholder(
            tf.float32, [self.h_n_step + 2, self.h_ex_batch_size, self.h_n_encode_lstm_step / 2, self.h_dim_wordvec])
        self.sp_queries_mask = tf.placeholder(
            tf.float32, [self.h_n_step + 2, self.h_ex_batch_size, self.h_n_encode_lstm_step / 2])

        result = tf.zeros([self.h_ex_batch_size])
        mag = 1.
        for i in range(self.h_n_step + 1):
            # [batch, words, dim]
            query_embeddings = self.sp_queries[i, :, :, :]
            # [batch, words]
            query_mask = self.sp_queries_mask[i, :, :]
            # [batch, words, dim]
            response_embeddings = self.sp_queries[i + 1, :, :, :]
            # [batch, words]
            response_mask = self.sp_queries_mask[i + 1, :, :]
            # [batch, dim]
            kw_emb = self.sp_kw_embs[i, :, :]
            q_sims = []
            r_sims = []
            for j in range(self.h_n_encode_lstm_step / 2):
                # [batch, dim]
                q_emb = query_embeddings[:, j, :]
                # [batch]
                q_emb_mask = query_mask[:, j]
                # [batch, dim]
                r_emb = response_embeddings[:, j, :]
                # [batch]
                r_emb_mask = response_mask[:, j]
                # [batch]
                q_sims.append(tf.nn.relu(self.cosine_similarity(q_emb, kw_emb) * q_emb_mask))
                # [batch]
                r_sims.append(tf.nn.relu(self.cosine_similarity(r_emb, kw_emb) * r_emb_mask))

            q_sims = tf.stack(q_sims, axis=1)
            r_sims = tf.stack(r_sims, axis=1)
            q_sim = tf.reduce_max(q_sims, axis=1)
            r_sim = tf.reduce_max(r_sims, axis=1)
            result += (tf.log(q_sim + 1e-33) + tf.log(r_sim + 1e-33)) * mag
            mag *= config.gama

        self.sr_kw_sim_result = result / (self.h_n_step + 1)

    '''
    for training and validation
    '''

    def run(self, sess, summary, dr, i2w, w2i, word_vector, output, _is_train=True):

        # 冲淡无意义关键词的影响
        sim_mag = config.gama

        ''' make init batch '''
        batch_x, batch_y, query, _ = dr.generate_training_batch_with_former(self.h_batch_size)

        # for multi sample
        batch_x = batch_x + copy.deepcopy(batch_x) + copy.deepcopy(batch_x)
        batch_y = batch_y + copy.deepcopy(batch_y) + copy.deepcopy(batch_y)
        query = query + copy.deepcopy(query) + copy.deepcopy(query)

        feats = utils.make_batch_X(batch_x, self.h_n_encode_lstm_step, self.h_dim_wordvec, word_vector)
        caption_matrix, caption_masks = utils.make_batch_Y(batch_y, w2i, self.h_n_decode_lstm_step)

        ''' reserve sample and first batch x embeddings for real run '''
        samples = [[x.encode('utf-8'), q.encode('utf-8')] for x, q in zip(batch_x, query)]
        ori_feats = copy.deepcopy(feats)

        generated_words_index, kw_ix, kw_emb, g_kw_probs = sess.run(
            [self.gr_words, self.gr_kw_index, self.gr_kw_emb, self.gr_probs],
            feed_dict={
                self.gp_batchx: feats
            })
        kw_history_list = [kw_emb]
        kw_ixs = copy.deepcopy(kw_ix)
        f_kw_ix = copy.deepcopy(kw_ix)
        kw_avg_sim = np.zeros([self.h_ex_batch_size])

        for i in range(self.h_n_step + 1):

            ''' generate next batch and get sample '''
            new_sentences = []

            for idx, gw in enumerate(generated_words_index):

                words = []
                for index in gw:
                    if index == 2:
                        break
                    word = i2w[index]
                    words.append(word)

                sent = ' '.join(words)
                new_sentences.append(sent)
                if type(query[idx]) == unicode:
                    query[idx] = query[idx].encode('utf-8')
                assert type(sent) == str
                batch_x[idx] = query[idx] + ' ' + sent

            for s, q in zip(samples, new_sentences):
                s.append(q)

            query = new_sentences

            ''' above code at last cycle just get last sentence out '''
            if i >= self.h_n_step:
                break

            ''' scheduling '''
            feats = utils.make_batch_X(batch_x, self.h_n_encode_lstm_step, self.h_dim_wordvec, word_vector)
            generated_words_index, kw_ix, kw_emb, kw_sim_log = sess.run(
                [self.gr_words, self.gr_kw_index, self.gr_kw_emb, self.gr_kw_avg_sim_log],
                feed_dict={
                    self.gp_batchx: feats,
                    self.gp_kw_emb_history: kw_history_list
                })
            assert type(kw_sim_log) == np.ndarray
            assert len(kw_sim_log) == self.h_ex_batch_size
            kw_avg_sim += kw_sim_log
            kw_history_list.append(kw_emb)
            kw_ixs = np.concatenate((kw_ixs, kw_ix), axis=1)
            sim_mag *= config.gama

        ''' calculate keyword-sentence & keyword-keyword similarity '''
        all_queries = [['' for _ in range(self.h_ex_batch_size)] for _ in range(self.h_n_step + 2)]
        assert len(samples) == self.h_ex_batch_size
        for i, abatch in enumerate(samples):
            assert len(abatch) == self.h_n_step + 3
            for j, aquery in enumerate(abatch[1:]):
                assert type(aquery) == str
                all_queries[j][i] = aquery
        qs_emb_batches = [[] for _ in range(self.h_n_step + 2)]
        qs_emb_batches_mask = [[] for _ in range(self.h_n_step + 2)]
        assert len(all_queries) == self.h_n_step + 2
        for i, query_batch in enumerate(all_queries):
            assert len(query_batch) == self.h_ex_batch_size
            qs_emb_batches[i] = utils.make_batch_X(query_batch,
                                                   self.h_n_encode_lstm_step / 2, self.h_dim_wordvec, word_vector)
            q_emb_mask = [[] for _ in range(self.h_ex_batch_size)]
            for j, aquery in enumerate(query_batch):
                assert type(aquery) == str
                mask_len = len(aquery.split())
                q_emb_mask[j] = [1. if k < mask_len else 0. for k in range(self.h_n_encode_lstm_step / 2)]
                assert len(q_emb_mask[j]) == self.h_n_encode_lstm_step / 2
            assert len(q_emb_mask) == self.h_ex_batch_size
            qs_emb_batches_mask[i] = q_emb_mask
        f_qs_emb_batches = np.asarray(qs_emb_batches, np.float32)
        f_qs_emb_batches_mask = np.array(qs_emb_batches_mask, np.float32)
        f_kw_embs = np.asarray(kw_history_list, np.float32)
        stc_kw_sims = sess.run(self.sr_kw_sim_result,
                               feed_dict={
                                   self.sp_kw_embs: f_kw_embs,
                                   self.sp_queries: f_qs_emb_batches,
                                   self.sp_queries_mask: f_qs_emb_batches_mask
                               })
        assert type(stc_kw_sims) == np.ndarray
        assert len(stc_kw_sims) == self.h_ex_batch_size
        kw_avg_sim /= self.h_n_step + 1.
        reward = stc_kw_sims + kw_avg_sim

        ''' real run '''
        if _is_train:
            _, loss_val, t_summary, true_loss_val, entropies, kw_probs, m_kw_index \
                , final_reward = sess.run(
                [self.train_op, self.loss, summary, self.true_loss, self.mr_entropy, self.mr_kw_probs,
                 self.mp_kw_index, self.reward],
                feed_dict={
                    self.mp_batchx: ori_feats,
                    self.mp_caption: caption_matrix,
                    self.mp_caption_mask: caption_masks,
                    self.mp_rewards: reward,
                    self.mp_kw_index: f_kw_ix
                })
        else:
            loss_val, t_summary, true_loss_val, entropies, kw_probs, m_kw_index \
                , final_reward = sess.run(
                [self.loss, summary, self.true_loss, self.mr_entropy, self.mr_kw_probs,
                 self.mp_kw_index, self.reward],
                feed_dict={
                    self.mp_batchx: ori_feats,
                    self.mp_caption: caption_matrix,
                    self.mp_caption_mask: caption_masks,
                    self.mp_rewards: reward,
                    self.mp_kw_index: f_kw_ix
                })

        ''' output debug message '''
        if output:
            sample_id = 0
            print('=====')
            for e in samples[sample_id]:
                print(e)
            for e in kw_ixs[sample_id]:
                print('kw: {}'.format(self.h_i2kw[e].encode('utf-8')))
            print('stc_kw_sim: {}'.format(stc_kw_sims[sample_id]))
            print('entropy: {}'.format(entropies[sample_id]))
            print('kw prob: {}'.format(kw_probs[sample_id]))
            print('stc_kw_sim: {}'.format(stc_kw_sims[sample_id]))
            print('avg_kw_sim: {}'.format(kw_avg_sim[sample_id]))
            print('final_reward: {}'.format(final_reward[sample_id]))

            print('->true_loss: {}'.format(true_loss_val))

        ''' return result to train.py '''
        ret_kw_strs = [[self.h_i2kw[kw].encode('utf-8') for kw in case_kw_ixs] for case_kw_ixs in kw_ixs]
        return loss_val, t_summary, ret_kw_strs

    def valid(self, sess, summary, dr, ixtoword, wordtoix, word_vector, output):

        return self.run(sess, summary, dr, ixtoword, wordtoix, word_vector, output, _is_train=False)

    def test(self, sess, args_tuple, _):

        feats, kw = args_tuple
        generated_words_index, kw_index = sess.run(
            [self.gr_words, self.gr_kw_index],
            feed_dict={
                self.gp_batchx: feats
            })
        return generated_words_index, [self.h_i2kw[k[0]].encode('utf-8') for k in kw_index]
