# coding=utf-8

from __future__ import print_function


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def make_batch_X(batch_X, n_encode_lstm_step, dim_wordvec, word_vector, noise=False):
    for i in range(len(batch_X)):
        batch_X[i] = [word_vector[w] if w in word_vector else np.zeros(dim_wordvec) for w in batch_X[i]]
        if noise:
            batch_X[i].insert(0, np.random.normal(size=(dim_wordvec,))) # insert random normal at the first step

        if len(batch_X[i]) > n_encode_lstm_step:
            batch_X[i] = batch_X[i][:n_encode_lstm_step]
        else:
            for _ in range(len(batch_X[i]), n_encode_lstm_step):
                batch_X[i].append(np.zeros(dim_wordvec))

    current_feats = np.array(batch_X)
    return current_feats # current_feats is word embedding sequence

def make_batch_Y(batch_Y, wordtoix, n_decode_lstm_step):
    current_captions = batch_Y
    current_captions = map(lambda x: '<bos> ' + x, current_captions)

    current_caption_ind = []
    for idx, each_cap in enumerate(current_captions):
        current_words_ind = []
        words = each_cap.lower().split(' ')
        for word in words:
            if len(current_words_ind) == n_decode_lstm_step - 1:
                break
            if word in wordtoix:
                current_words_ind.append(wordtoix[word])
            else:
                current_words_ind.append(wordtoix['<unk>'])
        current_words_ind.append(wordtoix['<eos>'])
        current_caption_ind.append(current_words_ind)


    current_caption_matrix = pad_sequences(current_caption_ind, padding='post', maxlen=n_decode_lstm_step)
    # add a column of zero, means the end of a sentence
    current_caption_matrix = np.hstack([current_caption_matrix, np.zeros([len(current_caption_matrix), 1])]).astype(int)
    current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
    nonzeros = np.array(map(lambda x: (x != 0).sum() + 1, current_caption_matrix))

    for ind, row in enumerate(current_caption_masks):
        row[:nonzeros[ind]] = 1

    return current_caption_matrix, current_caption_masks