"""Interactive mode for the tfidf DrQA retriever module."""

import argparse
import code
import prettytable
import logging
from retriever import get_class
from retriever.doc_db import DocDB


# ------------------------------------------------------------------------------
# Drop in to interactive
# ------------------------------------------------------------------------------


def process(query, k=1):
    doc_names, doc_scores = ranker.closest_docs(query, k)
    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id','Doc Title','Doc Text', 'Doc Score']
    )
    for i in range(len(doc_names)):
        table.add_row([i + 1, doc_names[i], db.get_doc_title(doc_names[i]),db.get_doc_text(doc_names[i])[:100], '%.5g' % doc_scores[i]])
    print(table)

if __name__ == '__main__':


    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--database', type=str,default="data/retriever_data/wikipedia_docs.db")
    args = parser.parse_args()

    logger.info('Initializing ranker...')
    ranker = get_class('tfidf')(tfidf_path=args.model)

    db = DocDB(args.database)
    banner = """
    Interactive TF-IDF DrQA Retriever
    >> process(question, k=1)
    >> usage()
    """


    def usage():
        print(banner)


    code.interact(banner=banner, local=locals())
