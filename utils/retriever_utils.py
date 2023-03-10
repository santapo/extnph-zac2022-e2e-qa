#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Various retriever utilities."""

import unicodedata

import numpy as np
import regex as re
import scipy.sparse as sp
from sklearn.utils import murmurhash3_32

from utils.utils import stopwords

# ------------------------------------------------------------------------------
# Sparse matrix saving/loading helpers.
# ------------------------------------------------------------------------------


def save_sparse_csr(filename, matrix, metadata=None):
    data = {
        'data': matrix.data,
        'indices': matrix.indices,
        'indptr': matrix.indptr,
        'shape': matrix.shape,
        'metadata': metadata,
    }
    np.savez(filename, **data)


def load_sparse_csr(filename):
    loader = np.load(filename, allow_pickle=True)
    matrix = sp.csr_matrix((loader['data'], loader['indices'],
                            loader['indptr']), shape=loader['shape'])
    return matrix, loader['metadata'].item(0) if 'metadata' in loader else None


# ------------------------------------------------------------------------------
# Token hashing.
# ------------------------------------------------------------------------------


def hash(token, num_buckets):
    """Unsigned 32 bit murmurhash for feature hashing."""
    return murmurhash3_32(token, positive=True) % num_buckets


def check_unique(dictionary):
    num_key = len(set(dictionary.keys()))
    num_value = len(set(dictionary.values()))
    if num_key == num_value :
        return True
    else:
        print(f"num keys= {num_key} not equal {num_value} = num values")
        return False
# ------------------------------------------------------------------------------
# Text cleaning.
# ------------------------------------------------------------------------------



def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFKC', text)

def is_punctuation(text):
    return re.match(r'^\p{P}+$', text)

def filter_word(text):
    """Take out stopwords, punctuation, and compound endings."""
    text = normalize(text)
    if is_punctuation(text):
        return True
    if text.lower() in stopwords:
        return True
    return False


def filter_ngram(gram, mode='any'):
    """Decide whether to keep or discard an n-gram.
    Args:
        gram: list of tokens (length N)
        mode: Option to throw out ngram if
          'any': any single token passes filter_word
          'all': all tokens pass filter_word
          'ends': book-ended by filterable tokens
    """
    filtered = [filter_word(w) for w in gram]
    if mode == 'any':
        return any(filtered)
    elif mode == 'all':
        return all(filtered)
    elif mode == 'ends':
        return filtered[0] or filtered[-1]
    else:
        raise ValueError('Invalid mode: %s' % mode)

def get_field(d, field_list):
    """get the subfield associated to a list of elastic fields 
        E.g. ['file', 'filename'] to d['file']['filename']
    """
    if isinstance(field_list, str):
        return d[field_list]
    else:
        idx = d.copy()
        for field in field_list:
            idx = idx[field]
        return 