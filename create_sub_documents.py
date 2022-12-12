import json
import logging
from multiprocessing import Manager, Pool

from tqdm import tqdm

from retriever import ES, create_index
from utils.utils import clean, remove_title, split_paragraph

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

host = "localhost"
port = "9200"
index_name = "paragraph"


def create_data(index):
    global lines
    data = []
    line = lines[index]
    doc = json.loads(line)
    text = doc['text']
    sentences = text.split(". ")
    if len(sentences) >1 and "định hướng" not in doc:
        paragraphs = split_paragraph(text)
        num_paragraph_concat = 2
        for i in range(0,len(paragraphs),num_paragraph_concat):
            tmp = {}
            text = " ".join(paragraphs[i:i+num_paragraph_concat])
            main_content = clean(text)
            tmp['id'] = doc['id'] + "_" + str(i+1)
            main_content = remove_title(main_content,doc['title'])
            tmp['title'] = doc['title']
            tmp['text'] = main_content
            data.extend([tmp])
    return data

if __name__ == "__main__":
    logger.warning("Creating index {} to {}:{}".format(index_name, host, port))
    create_index(host, port, index_name)
    data = []
    count = 0
    lines = open("./data/wikipedia_20220620_cleaned.jsonl",'r').readlines()
    es = ES(host,port,index_name)
    manager = Manager()
    data = []
    workers = Pool(10)
    num_batch = 10 
    step = max(int(len(lines) / num_batch), 1)
    batches = [range(i,i + step) if i + step < len(lines) else range(i,len(lines))  for i in range(0, len(lines), step)]
    for i, batch in enumerate(batches):
        logger.warning('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        for b_data in tqdm(workers.imap_unordered(create_data, batch),total=len(batch)):
            data.extend(b_data)
            if len(data) >=2048:
                es.put_data(data)
                data = []
    es.put_data(data)

