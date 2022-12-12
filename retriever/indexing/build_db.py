#!/usr/bin/env python3
"""A script to read in and store documents in a sqlite database."""

import argparse
import sqlite3
import json
import os
import logging
import importlib.util
from scripts.utils.util import clean
from multiprocessing import Pool,Process,Manager
import itertools 
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Import helper
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------


def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


def get_contents(filename):
    documents = []
    
    """Parse the contents of a file. Each line is a JSON encoded document."""
    # with open(filename, 'r') as lines:
    lines = open(filename,'r').readlines()
    for line in tqdm(lines):
        doc = json.loads(line)
        # Skip if it is empty or None
        if doc:
            # Add the document
            main_content = doc['text'].split("==")[0]
            main_content = clean(main_content)
            documents.append((doc['id'],doc['title'], main_content))
    return documents

def store_contents(data_path, save_path):
    """Preprocess and store a corpus of documents in sqlite.
    Args:
        data_path: Root path to directory (or directory of directories) of files
          containing json encoded documents (must have `id` and `text` fields).
        save_path: Path to output sqlite db.
        
    """
    if os.path.isfile(save_path):
        # os.remove(save_path)
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id PRIMARY KEY, title, text);")

    files = [f for f in iter_files(data_path)]

    count = 0

    for f in files:
        logger.info('Processing %s' % f)
        
        results = get_contents(f)
        count += len(results)
        c.executemany("INSERT INTO documents VALUES (?,?,?)", results)
        results = []

    logger.info('Read %d docs.' % count)
    logger.info('Committing...')
    conn.commit()
    conn.close()

    
# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/wikipedia_20220620_cleaned.jsonl')
    parser.add_argument('--save_path', type=str, default='data/retriever_data/wikipedia_main_docs.db')
    args = parser.parse_args()

    store_contents(args.data_path, args.save_path)
# import tokenizer.corenlp_tokenizer