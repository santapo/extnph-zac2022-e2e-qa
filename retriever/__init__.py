import os
from .. import DATA_DIR

DEFAULTS = {
    'db_path': os.path.join(DATA_DIR, 'retriever_data/wikipedia_docs.db'),
    'tfidf_path': os.path.join(
        DATA_DIR,
        'retriever_data/wikipedia_main_docs-tfidf-ngram=2-hash=16777216-tokenizer=corenlp.npz'
    ),
    'elastic_url': 'localhost:9200'
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


def get_class(name):
    if name == 'tfidf':
        return TfidfDocRanker
    if name == 'sqlite':
        return DocDB
    if name == 'elasticsearch':
        return ElasticDocRanker
    raise RuntimeError('Invalid retriever class: %s' % name)


from .doc_db import DocDB
from .tfidf_doc_ranker import TfidfDocRanker
from .elastic_doc_ranker import ElasticDocRanker