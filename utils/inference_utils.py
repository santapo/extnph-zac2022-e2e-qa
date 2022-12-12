import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, QuestionAnsweringPipeline

from .retriever_utils import filter_ngram
from .utils import *


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

def load_scripted_qna_model(qna_model, tokenizer_path, batch_size, **kwargs):
    model = AutoModelForQuestionAnswering.from_pretrained(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    scripted_model = torch.jit.load(qna_model)
    model = TorchJITQuestionAnsweringPipeline(scripted_model,
                                              model=model,
                                              tokenizer=tokenizer,
                                              batch_size=batch_size,
                                              **kwargs)
    return model

def get_contexts(query, es, tokenizer, k, rank_method):
    query = clean_entity(query)
    tokens = tokenizer.tokenize(query)
    ngrams, entities = tokens.ngrams(n=1,
                                     uncased=False,
                                     filter_fn=filter_ngram,
                                     return_pos = True)
    ngrams.sort(key=lambda x: len(x.split()),reverse=True)
    entities.sort(key=lambda x: len(x.split()),reverse=True)
    if len(entities)>0 and len(entities[0].split())>2:
        entity = entities[0]
    else:
        entity = None
    must_have =  list(set(entities))
    question_tokens = [x for x in ngrams if x not in must_have]

    if entity and len(entity.split())>2:
        texts = es.query_exact(origin_question=query,
                               question_tokens=ngrams,
                               must_have=None,
                               entity=entity,
                               re_ranking=None, 
                               k=k)
        if len(texts) == 0 :
            texts = es.query_exact(origin_question=query,
                                   question_tokens=question_tokens,
                                   must_have=must_have,
                                   entity=None,
                                   re_ranking=rank_method, 
                                   k=k)
        if len(texts) == 0 :
            texts = es.query_exact(origin_question=query,
                                   question_tokens=ngrams,
                                   must_have=None,
                                   entity=None,
                                   re_ranking=rank_method, 
                                   k=k)
    else:
        texts = es.query_exact(origin_question=query,
                               question_tokens=question_tokens,
                               must_have=must_have,
                               entity=None,
                               re_ranking=rank_method, 
                               k=k)
        # (2.2)
        if len(texts) == 0 :
            texts = es.query_exact(origin_question=query,
                                   question_tokens=ngrams,
                                   must_have=None,
                                   entity=None,
                                   re_ranking=rank_method, 
                                   k=k)
    #b (3.1)
    if len(texts) == 0:
        texts = es.query_context(" ".join(ngrams), re_ranking=rank_method, k=k)
    return texts

def get_answer(model, question_clean, contexts):
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