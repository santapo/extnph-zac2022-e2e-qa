from retriever.indexing.es_client import ES
from retriever.indexing.create_index import create_index
import json
from tqdm import tqdm
from utils.util import clean,write_file,remove_title, clean_entity, split_paragraph
from multiprocessing import Manager,Pool
import logging

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

host = "localhost"
port = "9200"
index_name = "wikipedia"

def create_data(line):
    doc = json.loads(line)
    text = doc['text']
    text_split = text.split()
    sentences = text.split(". ")
    if len(sentences) >1 and "định hướng" not in doc:
        
        # if len(text_split) > max_words:
        #     main_content = " ".join(text_split[:max_words])
        # else:
        #     main_content = text
        # main_content = clean(main_content)
        tmp = {}
        tmp['id'] = doc['id']
        # main_content = remove_title(main_content,doc['title'])
        tmp['title'] = doc['title']
        # tmp['text'] = main_content
        return tmp

if __name__ == "__main__":
    logger.warning("Creating index {} to {}:{}".format(index_name, host, port))
    create_index(host, port, index_name)
    data = []
    count = 0
    lines = open("./data/wikipedia_20220620_cleaned.jsonl",'r').readlines()
    es = ES(host,port,index_name)
    max_words = 10000

    workers = Pool(10)
    step = max(int(len(lines) / 10), 1)
    batches = [lines[i:i + step] for i in range(0, len(lines), step)]
    for i, batch in enumerate(batches):
        logger.warning('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        for tmp in workers.map(create_data, batch):
            if tmp:
                data.append(tmp)
            if len(data) ==2048:
                es.put_data(data)
                data = []

