from utils import *
import pandas as pd
import json
import numpy
import re
from tqdm import tqdm
from scripts.utils.util import clean,read_file


if __name__ == "__main__":
    s_date_kw = ["ngày tháng năm","ngày","ngày tháng","năm","tháng","thế kỷ"]
    e_date_kw = [" nào"," bao nhiêu"," mấy"]
    date_kw = [s + e for s in s_date_kw for e in e_date_kw]

    data_path = "./data/e2eqa-train+public_test-v1/zac2022_train_merged_final.json"
    max_wiki = 2000
    data = read_file(data_path)
    data = data['data']
    texts = []
    lbs = []
    count_wiki = 0
    for idx,row in tqdm(enumerate(data)):
        answer = row.get('answer',False)
        text = clean(row['question'])
        lb = None
        if not answer:
            if any([kw in text for kw in date_kw]):
                lb = 1
            elif "mấy" in text or "bao nhiêu" in text:
                lb = 2
        else:
            if "wiki" in answer :
                lb = 0
            elif "ngày" in answer or "tháng" in answer or "năm" in answer or "thế kỷ" in answer or "thế kỉ" in answer:
                lb = 1
            else:
                lb = 2
        if lb in [0,1,2]:
            if lb ==0:
                if count_wiki <= max_wiki:
                    texts.append(text)
                    lbs.append(lb)
                    count_wiki += 1
            else:
                texts.append(text)
                lbs.append(lb)
            
    res = pd.DataFrame(data={'text':texts,'label':lbs})
    res.to_csv('./data/e2eqa-train+public_test-v1/classify_data.csv',index=False)
