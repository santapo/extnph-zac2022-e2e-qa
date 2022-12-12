from retriever import ES
from tokenizers import get_tokenizer

from utils.inference_utils import get_answer, get_contexts, load_scripted_qna_model
from utils.post_processing import correct_answers, load_classify_model
from utils.utils import *


class ExtNphQA:
    def __init__(self,
                 es_host,
                 es_port,
                 es_title_index,
                 es_document_index,
                 classifier_model,
                 qna_model,
                 qna_tokenizer,
                 batch_size,
                 map_answer,
                 tokenizer):
        self.search_title_engine = ES(es_host, es_port, es_title_index)
        self.search_context_engine = ES(es_host, es_port, es_document_index)
        self.qtype_classifier = load_classify_model(classifier_model)
        self.short_model = load_scripted_qna_model(qna_model,
                                                   qna_tokenizer,
                                                   batch_size,
                                                   device=0,
                                                   max_seq_len=128,
                                                   doc_stride=10,
                                                   handle_impossible_answer=False)
        self.long_model = load_scripted_qna_model(qna_model,
                                                  qna_tokenizer,
                                                  batch_size,
                                                  device=0,
                                                  max_seq_len=384,
                                                  doc_stride=50,
                                                  handle_impossible_answer=False)
        self.tokenizer = get_tokenizer(tokenizer)
        self.answer_mapper = read_file(map_answer)

    def predict(self, question: str, k: int = 3):
        question = clean_question(question)
        contexts = get_contexts(question, self.search_context_engine, self.tokenizer, k=k)
        contexts = " ".join(contexts)
        short_answer, short_score = get_answer(self.short_model, question, contexts)
        long_answer, long_score = get_answer(self.long_model, question, contexts)

        answer = short_answer
        score = short_score
        if long_score > short_score:
            answer = long_answer
            score = long_score
        if answer:
            answer =  correct_answers(self.search_title_engine,
                                      question,
                                      answer,
                                      self.qtype_classifier,
                                      self.answer_mapper)
        return {
            "question": question,
            "answer": answer,
            "score": score,
        }