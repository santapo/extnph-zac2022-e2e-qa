"""A script to build the tf-idf document matrices for retrieval."""

import argparse
import logging
import math
import os
from collections import Counter
from functools import partial
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize

import numpy as np
import scipy.sparse as sp
from base import retriever, tokenizers
from scripts.utils.retriever_utils import filter_ngram, save_sparse_csr,check_unique
from scripts.utils.util import normalize_unicode

from interruptingcow import timeout

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Multiprocessing functions
# ------------------------------------------------------------------------------

PROCESS_TOK = None
PROCESS_DB = None
TOKEN2ID = None

def init(tokenizer_class, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)

def init1(tokenizer_class, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)

def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


# ------------------------------------------------------------------------------
# Build article --> word count sparse matrix.
# ------------------------------------------------------------------------------


def get_ngrams( ngram, doc_id):
    text = fetch_text(doc_id)
    try:
        with timeout(5, exception=RuntimeError): 
            tokens = tokenize(normalize_unicode(text))

            # Get ngrams from tokens, with stopword/punctuation filtering.
            ngrams = tokens.ngrams(
                n=ngram, uncased=True, filter_fn=filter_ngram
            )
    except RuntimeError:
        pass
    except:
        text = " ".join(text.split()[:1000])
        tokens = tokenize(normalize_unicode(text))
        ngrams = tokens.ngrams(
                n=ngram, uncased=True, filter_fn=filter_ngram
            )
    return ngrams

def create_token_dict(tok_class, db_class, db_opts,num_workers,ngram, doc_ids):
    workers = ProcessPool(
        num_workers,
        initializer=init1,
        initargs=(tok_class, db_class, db_opts)
    )

    logger.info('Creating dictionary of tokens')
    token2id = {}
    sub_func = partial(get_ngrams,ngram)
    step = max(int(len(doc_ids) / 10), 1)
    batches = [doc_ids[i:i + step] for i in range(0, len(doc_ids), step)]
    for i, batch in enumerate(batches):
        logger.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        for grams in workers.imap(sub_func, batch):
            for gram in grams:
                if gram not in token2id:
                    hash_value = len(token2id)
                    token2id[gram] = hash_value
                else:
                    hash_value = token2id[gram]
    workers.close()
    workers.join()
    return token2id

def count(doc2idx, token2id, ngram,  doc_id):
    """Fetch the text of a document and compute hashed ngrams counts."""
    row, col, data, len_term = [], [], [], []
    ngrams = get_ngrams(ngram, doc_id)
    # Hash ngrams and count occurences
    tmp = []
    hash2gram = {}
    try:
        for gram in ngrams:
            hash_value = token2id[gram]
            hash2gram[hash_value] = gram
            tmp.append(hash_value)
    except KeyError:
        pass
    counts = Counter(tmp)
    # Return in sparse matrix data format.
    print(len(counts))
    row.extend(counts.keys())
    col.extend([doc2idx[doc_id]] * len(counts))
    data.extend(counts.values())
    len_term = [len(hash2gram[hash_value].split()) for hash_value in counts.keys()]

    assert len(data) == len(row) == len(col) == len(len_term)
    return row, col, data, len_term


def get_count_matrix(args, db, db_opts):
    """Form a sparse word to document count matrix (inverted index).
    M[i, j] = # times word i appears in document j.
    count_matrix type : (hash_index_token_value, index_doc)___#time appears in index_doc
    """
    # Map doc_ids to indexes
    db_class = retriever.get_class(db)
    with db_class(**db_opts) as doc_db:
        doc_ids = doc_db.get_doc_ids()

    tok_class = tokenizers.get_class(args.tokenizer)
    
    
    doc2idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    token2id = create_token_dict(tok_class, db_class, db_opts, 10,args.ngram, doc_ids)

    workers = ProcessPool(
        args.num_workers,
        initializer=init,
        initargs=(tok_class, db_class, db_opts)
    )

    # Compute the count matrix in steps (to keep in memory)
    logger.info('Mapping...')
    row, col, data, len_term = [], [], [], []

    step = max(int(len(doc_ids) / 10), 1)
    batches = [doc_ids[i:i + step] for i in range(0, len(doc_ids), step)]
    _count = partial(count, doc2idx, token2id, args.ngram)
    logging.info("num workers = " + str(workers._processes))
    for i, batch in enumerate(batches):
        logger.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        for b_row, b_col, b_data, b_len_term in workers.imap_unordered(_count, batch):
            row.extend(b_row)
            col.extend(b_col)
            data.extend(b_data)
            len_term.extend(b_len_term)
    workers.close()
    workers.join()

    logger.info('Creating sparse matrix...')
    count_matrix = sp.csr_matrix(
        (data, (row, col)), shape=(len(token2id) ,len(doc_ids))
    )
    length_matrix = sp.csr_matrix(
        (len_term, (row, col)), shape=(len(token2id), len(doc_ids))
    )
    count_matrix.sum_duplicates()
    return count_matrix, length_matrix, token2id, (doc2idx, doc_ids)


# ------------------------------------------------------------------------------
# Transform count matrix to different forms.
# ------------------------------------------------------------------------------

def avgFieldLength(count_matrix):
    "return the average length of all documents"
    num_docs = count_matrix.shape[1]
    num_tokens_appeared = count_matrix.sum()
    return num_tokens_appeared / num_docs


def get_tfidf_matrix(count_matrix,length_matrix,k=1.2,b=0.75):

    """Convert the word count matrix into bm25 one.
    IDF = log(1+(N - Nt + 0.5) / (Nt + 0.5))
    L = avg(sum(len(document))
    tfidf = IDF * ((k + 1) * freq) / (k * (1.0 - b + b * len(term)/L) + freq)
    * tf = term frequency in document
    * N = number of documents
    * Nt = number of occurences of term in all documents
    """
    Ns = get_doc_freqs(count_matrix)
    idfs = np.log(1+(count_matrix.shape[1] - Ns + 0.5) / (Ns + 0.5))

    idfs = sp.diags(idfs, 0)
    L = avgFieldLength(count_matrix)
    print("L:",L)

    #(k + 1) * freq
    numerator = count_matrix.multiply(k+1)

    #(k * (1.0 - b + b * len(term)/L)
    d = length_matrix.multiply(b /L)
    d.data += 1.0 - b
    tmp = d.multiply(k)
    denominator = count_matrix + tmp
    denominator = denominator.power(-1)
    tfs = numerator.multiply(denominator)
    
    tfidfs = idfs.dot(tfs)
    return tfidfs,L

def get_doc_freqs(count_matrix):
    """Return hash_token_value --> # of docs it appears in. shape: (max_hash_value_appeared,)"""
    binary = (count_matrix > 0).astype(int)
    freqs = np.array(binary.sum(1)).squeeze()
    return freqs


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default="data/retriever_data/wikipedia_main_docs.db",
                        help='Path to sqlite db holding document texts')
    parser.add_argument('--out_dir', type=str, default="data/retriever_data/",
                        help='Directory for saving output files')
    parser.add_argument('--ngram', type=int, default=2,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    parser.add_argument('--tokenizer', type=str, default='corenlp',
                        help=("String option specifying tokenizer type to use "
                              "(e.g. 'corenlp')"))
    parser.add_argument('--num-workers', type=int, default=10,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    logging.info('Counting words...')
    count_matrix,length_matrix, token2id, doc_dict = get_count_matrix(
        args, 'sqlite', {'db_path': args.db_path}
    )

    logger.info('Making tfidf vectors...')
    tfidf,L = get_tfidf_matrix(count_matrix,length_matrix)
    logger.info('Getting word-doc frequencies...')
    freqs = get_doc_freqs(count_matrix)

    basename = os.path.splitext(os.path.basename(args.db_path))[0]
    basename += ('-tfidf-ngram=%d-words=%d-tokenizer=%s' %
                 (args.ngram, len(token2id), args.tokenizer))
    filename = os.path.join(args.out_dir, basename)

    logger.info('Saving to %s.npz' % filename)
    metadata = {
        'doc_freqs': freqs,
        'tokenizer': args.tokenizer,
        'token_ids':token2id,
        'ngram': args.ngram,
        'doc_dict': doc_dict,
        'average_length': L
    }
    is_unique = check_unique(token2id)
    if is_unique:
        logger.info("All tokens are unique")
    else:
        logger.warning("Some tokens are non-unique")
    save_sparse_csr(filename, tfidf, metadata)