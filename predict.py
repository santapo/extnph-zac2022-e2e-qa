import argparse
from tqdm import tqdm

import torch
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer,
                          QuestionAnsweringPipeline, pipeline)

from base import tokenizers
from retriever import ranking
from retriever.indexing.es_client import ES
from scripts.utils.post_processing import (format_date,
                                           get_meta_question_class,
                                           get_question_class,
                                           load_classify_model)
from scripts.utils.retriever_utils import filter_ngram
from scripts.utils.util import *


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path',default="./data/e2eqa-train+public_test-v1/zac2022_testa_only_question.json",type=str)
    parser.add_argument('--output',default="./submission.json",type=str)
    parser.add_argument('--database', type=str,default="data/retriever_data/wikipedia_docs.db")
    parser.add_argument('--scripted_qna_model',default='data/traced_mrc.pt')
    parser.add_argument('--qna_model',default='nguyenvulebinh/vi-mrc-large')
    parser.add_argument('--classify_model', type=str,default="data/model_classify")
    parser.add_argument('--title_file', type=str,default="data/wikipedia_20220620_all_titles.txt")
    parser.add_argument('--map_answer', type=str,default="scripts/utils/map.json")
    parser.add_argument('--es_host', type=str,default="localhost")
    parser.add_argument('--es_port', type=str,default="9200")
    parser.add_argument('--es_title_index', type=str,default="wikipedia")
    parser.add_argument('--es_document_index', type=str,default="paragraph")
    parser.add_argument('--batch_size', type=int,default=32)
    parser.add_argument('--tfidf',default=None)
    args = parser.parse_args()
    return args


class TorchJITQuestionAnsweringPipeline(QuestionAnsweringPipeline):
    def __init__(self, scripted_model, **kwargs):
        super().__init__(**kwargs)
        self.model.to("cpu"); del self.model
        self.model = scripted_model
        self.model.to(kwargs.get("device"))
        self.model.eval()

    def _forward(self, inputs):
        example = inputs["example"]
        model_inputs = {k: inputs[k] for k in self.tokenizer.model_input_names}
        with torch.no_grad():
            start, end = self.model(**model_inputs)[:2]
        return {"start": start, "end": end, "example": example, **inputs}

def load_scripted_qna_model(scripted_model_path, model_path, batch_size, **kwargs):
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    scripted_model = torch.jit.load(scripted_model_path)
    model = TorchJITQuestionAnsweringPipeline(scripted_model,
                                              model=model,
                                              tokenizer=tokenizer,
                                              batch_size=batch_size,
                                              **kwargs)
    return model

def load_qna_model(model_path, batch_size, **kwargs):
    model = pipeline('question-answering',
                     model=model_path,
                     tokenizer=model_path,
                     batch_size=batch_size,
                     **kwargs)
    return model

def get_contexts(query, es, tokenizer, k):
    query = clean_entity(query)
    tokens = tokenizer.tokenize(query)

    ngrams, entities = tokens.ngrams(
                n=1,
                uncased=False,
                filter_fn=filter_ngram,
                return_pos = True
            )

    ngrams.sort(key=lambda x: len(x.split()),reverse=True)
    
    entities.sort(key=lambda x: len(x.split()),reverse=True)

    if len(entities) >0 and len(entities[0].split())>2:
        entity = entities[0]
    else:
        entity = None
    must_have =  list(set(entities))
    question_tokens = [x for x in ngrams if x not in must_have]

    if entity and len(entity.split()) > 2:

        texts = es.query_exact( origin_question= query,
                        question_tokens = ngrams,
                        must_have = None,
                        entity = entity,
                        re_ranking= None, 
                        k = k
                    )

        if len(texts) == 0 :
            texts = es.query_exact( origin_question= query,
                        question_tokens = question_tokens,
                        must_have = must_have,
                        entity = None,
                        re_ranking= ranking.re_ranking, 
                        k = k
                    )

        if len(texts) == 0 :
            texts = es.query_exact( origin_question= query,
                        question_tokens = ngrams,
                        must_have = None,
                        entity = None,
                        re_ranking= ranking.re_ranking, 
                        k = k
                    )
    else:

        texts = es.query_exact( origin_question= query,
                        question_tokens = question_tokens,
                        must_have = must_have,
                        entity = None,
                        re_ranking= ranking.re_ranking, 
                        k = k
                    )
        # (2.2)
        if len(texts) == 0 :
            texts = es.query_exact( origin_question= query,
                        question_tokens = ngrams,
                        must_have = None,
                        entity = None,
                        re_ranking= ranking.re_ranking, 
                        k = k
                    )
    #b (3.1)
    if len(texts) == 0:
        texts = es.query_context(" ".join(ngrams),re_ranking= ranking.re_ranking, k =k)

    return texts

def get_answers(model, question_clean, contexts):
    if contexts == "" or contexts is None:
        return None, 1
    else:
        QA_input = {
            'question': question_clean,
            'context': contexts
            }
        inference = model(QA_input)
        answers, score = inference['answer'], inference['score']
        if score < 0.01:
            answers = ""
        answers = None if answers == "" else answers
        return answers, score

def correct_answers(es, orginal_question, answers, classifier, mapper):
    answers = remove_all_punc(answers)
    answers = answers.strip()
    answers = mapper.get(answers,answers)
    question_class = get_question_class(orginal_question, classifier)
    meta_question_class = get_meta_question_class(question_class)
    if meta_question_class == "number":
        matches = re.findall("(\d+)", answers)
        answers = matches[0] if bool(matches) else "null"
    elif meta_question_class == "date":
        answers = format_date(answers, question_class)
    elif meta_question_class == "wiki":
        answers = es.query_title(answers)
        answers = "wiki/" + answers.replace(" ","_")
    return answers

if __name__ == "__main__":
    args = load_args()

    model_1 = load_scripted_qna_model(args.scripted_qna_model,
                                    args.qna_model,
                                    args.batch_size,
                                    device=0,
                                    max_seq_len=128,
                                    doc_stride=10,
                                    handle_impossible_answer=False)
    model_2 = load_scripted_qna_model(args.scripted_qna_model,
                                    args.qna_model,
                                    args.batch_size,
                                    device=0,
                                    max_seq_len=384,
                                    doc_stride=50,
                                    handle_impossible_answer=False)
    # model = load_qna_model(args.qna_model,
    #                        args.batch_size,
    #                        device=0,
    #                        doc_stride=50,
    #                        max_seq_len=384,
    #                        handle_impossible_answer=False)
    question_classifier = load_classify_model(args.classify_model)
    es_context = ES(args.es_host, args.es_port, args.es_document_index)
    es_map_title = ES(args.es_host, args.es_port, args.es_title_index)
    mapper = read_file(args.map_answer)
    data = read_file(args.test_path)
    data = data['data']

    tok_class = tokenizers.get_class('corenlp')
    tokenizer = tok_class()
    results = []
    debugs = []
    for i, d in tqdm(enumerate(data)):
        tmp = {}
        _id = d['id']
        question = d['question']
        question = clean_question(question)
        contexts = get_contexts(question, es_context, tokenizer, k=3)
        contexts = " ".join(contexts)
        answers_1, score_1 = get_answers(model_1, question, contexts)
        answers_2, score_2 = get_answers(model_2, question, contexts)
        answers = answers_1
        if score_2 > score_1:
            answers = answers_2
        tmp['id'] = _id
        tmp['question'] = question
        tmp['answer'] = answers
        if answers:
            tmp['answer'] =  correct_answers(es_map_title, question, answers, question_classifier, mapper)
        results.append(tmp)

    write_file(args.output,{"data": results})
    tokenizer.shutdown()