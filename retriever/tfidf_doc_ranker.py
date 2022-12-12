"""Rank documents with TF-IDF scores"""

import logging
from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np
import scipy.sparse as sp
from scripts.utils import retriever_utils

from .. import tokenizers
from . import DEFAULTS

logger = logging.getLogger(__name__)


class TfidfDocRanker(object):
    """Loads a pre-weighted inverted index of token/document terms.
    Scores new queries by taking sparse dot products.
    """

    def __init__(self, tfidf_path=None, strict=True):
        """
        Args:
            tfidf_path: path to saved model file
            strict: fail on empty queries or continue (and return empty result)
        """
        # Load from disk
        tfidf_path = tfidf_path or DEFAULTS['tfidf_path']
        logger.info('Loading %s' % tfidf_path)
        matrix, metadata = retriever_utils.load_sparse_csr(tfidf_path)
        self.doc_mat = matrix
        self.ngrams = metadata['ngram']
        self.token2id = metadata['token_ids']
        self.tokenizer = tokenizers.get_class(metadata['tokenizer'])()
        self.doc_freqs = metadata['doc_freqs'].squeeze()
        self.average_length = metadata['average_length']
        ### doc_dict = [{index in db: order},[]]
        self.doc_dict = metadata['doc_dict']
        self.num_docs = len(self.doc_dict[0])
        self.strict = strict
    def get_doc_index(self, doc_id):
        """Convert doc_id --> doc_index"""
        return self.doc_dict[0][doc_id]

    def get_doc_id(self, doc_index):
        """Convert doc_index --> doc_id"""
        return self.doc_dict[1][doc_index]

    def closest_docs(self, query, k=1):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
        spvec = self.text2spvec(query)
        res = spvec * self.doc_mat
        if len(res.data) <= k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_scores = res.data[o_sort]
        doc_ids = [self.get_doc_id(i) for i in res.indices[o_sort]]
        return doc_ids, doc_scores

    def batch_closest_docs(self, queries, k=1, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results

    def parse(self, query):
        """Parse the query into tokens (either ngrams or tokens)."""
        tokens = self.tokenizer.tokenize(query)
        return tokens.ngrams(n=self.ngrams, uncased=True,
                             filter_fn=retriever_utils.filter_ngram)

    def text2spvec(self, query,k=1.2,b=0.75):
        """Create a sparse tfidf-weighted word vector from query.
        IDF = log(1+(N - Nt + 0.5) / (Nt + 0.5))
        L = avg(sum(len(document))
        tfidf = IDF * ((k + 1) * freq) / (k * (1.0 - b + b * len(term)/L) + freq)
        * tf = term frequency in document
        * N = number of documents
        * Nt = number of occurences of term in all documents
        """
        # Get hashed ngrams
        words = self.parse(retriever_utils.normalize(query))
        # print("87: ",words)
        """
         words_hash = {'word_id':len(word.split())}
         """
        words_hash = {}
        wids = []
        for w in words:
            # wid = retriever_utils.hash(w, self.hash_size)
            try:
                wid = self.token2id[w]
                words_hash[str(wid)] = len(w.split())
                wids.append(wid)
            except KeyError:
                pass
                # print(w)
                continue
        if len(words_hash) == 0:
            if self.strict:
                raise RuntimeError('No valid word in: %s' % query)
            else:
                logger.warning('No valid word in: %s' % query)
                return sp.csr_matrix((1, len(self.token2id)))

        # Count TF
        """
        tf was stored in self.doc_mat. It can be loads later
        """
        wids_unique, wids_counts = np.unique(wids, return_counts=True)
        w_len = [words_hash[str(wid)] for wid in wids_unique]
        w_len = np.array(w_len)

        #(k + 1) * freq
        numerator = (k+1) * wids_counts

        #(k * (1.0 - b + b * len(term)/L) + freq
        d = k*(1-b + b * w_len/self.average_length)

        denominator = d + wids_counts
        tfs = numerator / denominator
                    
        #             tfidfs = idfs.dot(tfs)
        # Count IDF
        Ns = self.doc_freqs[wids_unique]
        idfs = np.log(1+(self.num_docs - Ns + 0.5) / (Ns + 0.5))
        # TF-IDF
        data = np.multiply(tfs, idfs)
        # One row, sparse csr matrix
        indptr = np.array([0, len(wids_unique)])

        spvec = sp.csr_matrix(
            (data, wids_unique, indptr), shape=(1, len(self.token2id))
        )
        return spvec
