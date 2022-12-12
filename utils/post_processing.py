import re

from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

META_QUESTION_CLASSES = {
    "full_date": "date",
    "month_year": "date",
    "date_month": "date",
    "undetected_date": "date",
    "undetected_month": "date",
    "year": "date",
    "century": "date",
    "number": "number",
    "wiki": "wiki"
}

QUESTION_FORMAT_DICT = {
    "ngày tháng năm nào": "full_date",
    "ngày tháng năm bao nhiêu": "full_date",
    "ngày tháng năm mấy": "full_date",
    "tháng năm nào": "month_year",
    "tháng năm mấy": "month_year",
    "tháng năm bao nhiêu": "month_year",
    "ngày tháng nào": "date_month",
    "ngày tháng mấy": "date_month",
    "ngày tháng bao nhiêu": "date_month",
    "ngày bao nhiêu": "undetected_date",
    "ngày nào": "undetected_date",
    "ngày mấy": "undetected_date",
    "thời gian nào": "undetected_date",
    "thời điểm nào": "undetected_date",
    "bao giờ": "undetected_date",
    "tháng bao nhiêu": "undetected_month",
    "tháng nào": "undetected_month",
    "tháng mấy": "undetected_month",
    "năm bao nhiêu": "year",
    "năm nào": "year",
    "năm mấy": "year",
    "thế kỷ nào": "century",
    "thế kỷ mấy": "century",
    "thế kỷ bao nhiêu": "century",
    "mấy": "number",
    "bao nhiêu": "number",
    "ngày gì": "wiki",
    "là gì": "wiki",
    "gì": "wiki",
    "là ai": "wiki",
    "ai là": "wiki",
    "đâu là": "wiki",
    "ở đâu": "wiki"
}

DATE_SUB_PATTERNS = {
    "full_date": [
                    (r"ngày (\d+)(\/|-|\.)(\d+)(\/|-|\.)(\d+)", r"ngày \1 tháng \3 năm \5"),
                    (r"(\d+)(\/|-|\.)(\d+)(\/|-|\.)(\d+)", r"ngày \1 tháng \3 năm \5"),
                    (r"(\d+) tháng (\d+) năm (\d+)", r"ngày \1 tháng \2 năm \3"),
                    (r"ngày (\d+) tháng (\d+) năm (\d+)", r"ngày \1 tháng \2 năm \3")
                ],
    "month_year": [
                    (r"ngày (\d+)(\/|-|\.)(\d+)(\/|-|\.)(\d+)", r"tháng \3 năm \5"),
                    (r"(\d+)(\/|-|\.)(\d+)(\/|-|\.)(\d+)", r"tháng \3 năm \5"),
                    (r"(\d+) tháng (\d+) năm (\d+)", r"tháng \2 năm \3"),
                    (r"ngày (\d+) tháng (\d+) năm (\d+)", r"tháng \2 năm \3"),
                    (r"tháng (\d+)(\/|-|\.)(\d+)", r"tháng \1 năm \3"),
                    (r"(\d+)(\/|-|\.)(\d+)", r"tháng \1 năm \3"),
                    (r"(\d+) năm (\d+)", r"tháng \1 năm \2"),
                    (r"tháng (\d+) năm (\d+)", r"tháng \1 năm \2")
                ],
    "date_month": [
                    (r"ngày (\d+)(\/|-|\.)(\d+)(\/|-|\.)(\d+)", r"ngày \1 tháng \3"),
                    (r"(\d+)(\/|-|\.)(\d+)(\/|-|\.)(\d+)", r"ngày \1 tháng \3"),
                    (r"ngày (\d+) tháng (\d+) năm (\d+)", r"ngày \1 tháng \2"),
                    (r"(\d+) tháng (\d+) năm (\d+)", r"ngày \1 tháng \2"),
                    (r"ngày (\d+)(\/|-|\.)(\d+)", r"ngày \1 tháng \3"),
                    (r"(\d+)(\/|-|\.)(\d+)", r"ngày \1 tháng \3"),
                    (r"$(\d+) tháng (\d+)", r"ngày \1 tháng \2"),
                    (r"ngày (\d+) tháng (\d+)", r"ngày \1 tháng \2"),
                ],
    "year": [
                (r"ngày (\d+)(\/|-|\.)(\d+)(\/|-|\.)(\d+)", r"năm \5"),
                (r"(\d+)(\/|-|\.)(\d+)(\/|-|\.)(\d+)", r"năm \5"),
                (r"ngày (\d+) tháng (\d+) năm (\d+)", r"năm \3"),
                (r"(\d+) tháng (\d+) năm (\d+)", r"năm \3"),
                (r"tháng (\d+)(\/|-|\.)(\d+)", r"năm \3"),
                (r"(\d+)(\/|-|\.)(\d+)", r"năm \3"),
                (r"tháng (\d+) năm (\d+)", r"năm \2"),
                (r"(\d+) năm (\d+)", r"năm \2"),
                (r"(\d+)", r"năm \1"),
                (r"năm (\d+)", r"năm \1")
            ],
    "century": [
                (r"$(\d+)", r"thế kỷ \1"),
                (r"thế kỷ (\d+)", r"thế kỷ \1")
            ]

}

DATE_SYNONYMS = {
    "mùng": "ngày",
    "mồng": "ngày",
    "tháng chạp": "tháng 12",
    "chạp": "12",
    "tết": "tháng 1",
    "rằm": "15",
    "ngày rằm": "ngày 15",
    "tháng giêng": "tháng 1",
    "giêng": "1",
    "hai": "2",
    "ba": "3",
    "tư": "4",
    # "năm": "5",
    "sáu": "6",
    "bẩy": "7",
    "bảy": "7",
    "tám": "8",
    "chín": "9",
    "mười": "10",
    "mười một": "11"
}


def get_question_class(question, classifier):
    question_class = get_question_class_by_dict(question)
    if question_class is not None: return question_class
    question_class, score = get_question_class_by_model(classifier, question)
    return question_class

def get_meta_question_class(question_class):
    return META_QUESTION_CLASSES[question_class]

def get_question_class_by_dict(question: str):
    question = question.lower()
    for key, value in QUESTION_FORMAT_DICT.items():
        if key in question:
            return value
    return None

def format_date(answer, question_class):
    answer = answer.lower()
    for key, sym in DATE_SYNONYMS.items():
        if key in answer:
            answer = answer.replace(key, sym)

    formatted_answer = answer
    if question_class in DATE_SUB_PATTERNS:
        sub_patterns = DATE_SUB_PATTERNS[question_class]
        flags = False
        for pattern, repl in sub_patterns:
            re_answer = re.sub(pattern, repl, formatted_answer)
            if re_answer != formatted_answer:
                flags= True
                formatted_answer = re_answer
                break
        if flags:
            matches = re.findall(sub_patterns[-1][0], formatted_answer)
            if question_class == "full_date":
                formatted_answer = "ngày {} tháng {} năm {}".format(*matches[0])
            elif question_class == "month_year":
                formatted_answer = "tháng {} năm {}".format(*matches[0])
            elif question_class == "date_month":
                formatted_answer = "ngày {} tháng {}".format(*matches[0])
            elif question_class == "year":
                formatted_answer = "năm {}".format(matches[0])
    elif question_class in ["undetected_month", "undetected_date"]:
        for date_format, sub_patterns in DATE_SUB_PATTERNS.items():
            for pattern, _ in sub_patterns:
                matches = re.findall(pattern, formatted_answer)
                flags = bool(matches)
                if flags: break
            if flags: break
        if date_format == "full_date":
            formatted_answer = "ngày {} tháng {} năm {}".format(*matches[0])
        elif date_format == "month_year":
            formatted_answer = "tháng {} năm {}".format(*matches[0])
        elif date_format == "date_month":
            formatted_answer = "ngày {} tháng {}".format(*matches[0])
    return formatted_answer

def load_classify_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=3,
            output_attentions=False,
            output_hidden_states=False,
        ).to('cuda')
    tokenizer =  AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    classifier = pipeline("sentiment-analysis",
                            model=model,
                            tokenizer=tokenizer,
                            device=0,
                            framework='pt')
    return classifier

def get_question_class_by_model(classifier, question):
    q_class = classifier(question)[0]
    label = q_class['label']
    score = q_class['score']
    if label == 'LABEL_0':    return ('wiki', score)
    elif label == 'LABEL_1':  return ('undetected_date', score)
    elif label == 'LABEL_2':  return ('number', score)