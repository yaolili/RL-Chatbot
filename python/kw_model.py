# coding=utf-8

import tensorflow as tf
import numpy as np

class Kw_chatbot():
    def __init__(self, dim_wordvec, n_words, dim_hidden, batch_size, n_encode_lstm_step, n_decode_lstm_step, bias_init_vector=None, lr=0.0001):
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

        self.encode_vector_W = tf.Variable(tf.random_uniform([dim_wordvec, dim_hidden], -0.1, 0.1), name='encode_vector_W')
        self.encode_vector_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_vector_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

        # keywords params
        self.keywords_W = tf.Variable(tf.random_uniform([dim_wordvec, dim_hidden], -0.1, 0.1), name='keywords_W')
        self.keywords_b = tf.Variable(tf.zeros([dim_hidden]), name='keywords_b')


    def build_model(self):
        word_vectors = tf.placeholder(tf.float32, [self.batch_size, self.n_encode_lstm_step, self.dim_wordvec])

        # add keyword embedding from pretrain X input
        kw_vectors = tf.placeholder(tf.float32, [self.batch_size, self.dim_wordvec])
        kw_info = tf.nn.xw_plus_b(kw_vectors, self.keywords_W, self.keywords_b)


        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_decode_lstm_step+1])
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_decode_lstm_step+1])

        word_vectors_flat = tf.reshape(word_vectors, [-1, self.dim_wordvec])
        wordvec_emb = tf.nn.xw_plus_b(word_vectors_flat, self.encode_vector_W, self.encode_vector_b ) # (batch_size*n_encode_lstm_step, dim_hidden)
        wordvec_emb = tf.reshape(wordvec_emb, [self.batch_size, self.n_encode_lstm_step, self.dim_hidden])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        probs = []
        entropies = []
        loss = 0.0

        ##############################  Encoding Stage ##################################
        for i in range(0, self.n_encode_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(wordvec_emb[:, i, :], state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

        ############################# Decoding Stage ######################################
        for i in range(0, self.n_decode_lstm_step):
            # with tf.device("/cpu:0"):
            current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])

            tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                # here I replace "padding" with "kw_info"
                # FIXME: whether the first time step should be padding or not?
                # while at model.py, we just use padding
                output1, state1 = self.lstm1(kw_info, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

            labels = tf.expand_dims(caption[:, i+1], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat([indices, labels], 1)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
            cross_entropy = cross_entropy * caption_mask[:, i]
            entropies.append(cross_entropy)
            probs.append(logit_words)

            current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
            loss = loss + current_loss

        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            # train_op = tf.train.RMSPropOptimizer(self.lr).minimize(total_loss)
            # train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(total_loss)
            train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        inter_value = {
            'probs': probs,
            'entropies': entropies
        }

        self.train_op = train_op
        self.loss = loss
        self.word_vectors = word_vectors
        self.kw_vectors = kw_vectors
        self.caption = caption
        self.caption_mask = caption_mask
        self.inter_value = inter_value
        return loss

    def build_generator(self):
        word_vectors = tf.placeholder(tf.float32, [self.batch_size, self.n_encode_lstm_step, self.dim_wordvec])

        word_vectors_flat = tf.reshape(word_vectors, [-1, self.dim_wordvec])
        wordvec_emb = tf.nn.xw_plus_b(word_vectors_flat, self.encode_vector_W, self.encode_vector_b)
        wordvec_emb = tf.reshape(wordvec_emb, [self.batch_size, self.n_encode_lstm_step, self.dim_hidden])

        # add keyword embedding from pretrain X input
        kw_vectors = tf.placeholder(tf.float32, [self.batch_size, self.dim_wordvec])
        kw_info = tf.nn.xw_plus_b(kw_vectors, self.keywords_W, self.keywords_b)


        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []

        for i in range(0, self.n_encode_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(wordvec_emb[:, i, :], state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

        for i in range(0, self.n_decode_lstm_step):
            tf.get_variable_scope().reuse_variables()

            if i == 0:
                # with tf.device('/cpu:0'):
                current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([self.batch_size], dtype=tf.int64))

            with tf.variable_scope("LSTM1"):
                # here I replace "placeholder" with "kw_info"
                output1, state1 = self.lstm1(kw_info, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            # with tf.device("/cpu:0"):
            current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)

            embeds.append(current_embed)

        self.word_vectors = word_vectors
        self.kw_vectors = kw_vectors
        self.generated_words = tf.stack(generated_words, axis=1)
        self.probs = tf.stack(probs, axis=1)
        self.embeds = tf.stack(embeds, axis=1)

        
    def deal_with_kw(self, kw, word_vector):
        assert(type(kw) is list)
        for i in range(len(kw)):
            # Notice: original kw[i] is string, but len(kw[i]) is calculated by bytes
            # use split() to make sure one word
            # after decode('utf-8'), kw[i] is unicode
            try:
                kw[i] = kw[i].decode('utf-8')
            except:
                print(kw)
                print(kw[i])
                assert(False)
            kw[i] = kw[i].strip().split(" ")
            assert len(kw[i]) == 1, 'Length of kw is not 1'
            kw[i] = kw[i][0]
            
            # FIXME: cannot hash kw[i] directly
            a = kw[i]
            kw[i] = word_vector[a] if a in word_vector else np.zeros(self.dim_wordvec)
        return np.array(kw)
        
    def run(self, sess, summary, args_tuple, word_vector):
    
        feats, kw, caption_matrix, caption_masks = args_tuple
        current_kw = self.deal_with_kw(kw, word_vector)
        _, loss_val, t_summary = sess.run(
            [self.train_op, self.loss, summary],
            feed_dict={
                self.word_vectors: feats,
                self.kw_vectors: current_kw,
                self.caption: caption_matrix,
                self.caption_mask: caption_masks
            })
        return loss_val, t_summary
        
    def valid(self, sess, summary, args_tuple, word_vector):
    
        feats, kw, caption_matrix, caption_masks = args_tuple
        current_kw = self.deal_with_kw(kw, word_vector)
        loss_val, v_summary = sess.run(
            [self.loss, summary],
            feed_dict={
                self.word_vectors: feats,
                self.kw_vectors: current_kw,
                self.caption: caption_matrix,
                self.caption_mask: caption_masks
            })
        return loss_val, v_summary
        
    def test(self, sess, args_tuple, word_vector):
    
        feats, kw = args_tuple
        current_kw = self.deal_with_kw(kw, word_vector)
        generated_words_index = sess.run(self.generated_words,
            feed_dict={self.word_vectors: feats, self.kw_vectors: current_kw})
        return generated_words_index, None