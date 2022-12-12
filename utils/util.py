import json
import regex as re
import unicodedata
import pickle as pkl
import os
import pkgutil
import lxml.html
import lxml.html.clean
import string

def _json_loader(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data

def _jsonl_loader(path):
    data = []
    with open(path,'r') as lines:
        for line in lines:
            data.append(json.load(line))
    return data

def _txt_reader(path):
    data = []
    with open(path,'r') as lines:
        for line in lines:
            data.append(line.replace("\n",""))
    return data

def _pkl_reader(path):
    with open(path,'rb') as f:
        data = pkl.load(f)
    return data

def read_file(filepath,extension_allow=["jsonl","json","txt",'pkl']):
    extension = filepath.split(".")[-1]
    if extension not in extension_allow:
        raise ValueError(f"only {extension_allow} were allowed")
    else:
        if filepath.endswith('.jsonl'):
            data =_jsonl_loader(filepath)
        elif filepath.endswith('.json'):
            data = _json_loader(filepath)
        elif filepath.endswith('.txt'):
            data = _txt_reader(filepath) 
        elif filepath.endswith('.pkl'):
            data = _pkl_reader(filepath)    
    return data

root_path = __package__.replace(".","/")
accent_norm = read_file(os.path.join(root_path,'accent_norm.json'))
stopwords = read_file(os.path.join(root_path,'stopwords.txt'))

def write_file(filepath,data):
    if isinstance(data,object):
        with open(filepath,'wb+') as f:
            pkl.dump(data,f,pkl.HIGHEST_PROTOCOL)
    with open(filepath,'w+') as f:
        if isinstance(data,dict):
            json.dump(data,f,ensure_ascii=False)
        elif isinstance(data,list):
            f.write('\n'.join(data))

def normalize_accent(text,accent_norm = accent_norm):
    for k,v in accent_norm.items():
        text = text.replace(k,v)
    return text

def clean_with_regex(text,allow_punc = [',','.','/', '%', '-',"(",")"]):
    #r"\((.*?)\)"
    patterns = {
        r"\n+": " ",
        r"\.+":". ",
        r"(\.\s)+":". ",
        r'\s+': " "
    }
    punctuation = r"\?\!"
    replacements = {
        "BULLET::::" : " ",
        "•":" ",
        "\"": "",
        "\'":""

    }
    remove_punctuation = "".join([x for x in string.punctuation if x not in allow_punc])
    #replace text
    for k,v in replacements.items():
        text = text.replace(k,v)
    #convert to . end of sentence
    text = re.sub(re.compile(punctuation),'.',text)
    text = text.translate(str.maketrans('/', ' ', remove_punctuation))
    for pattern,replace in patterns.items():
        cleanr = re.compile(pattern)
        text = re.sub(cleanr,replace,text)
    return text

def _date_month_year(match):
    if "ngày" in match.group(0):
        return match.group(2) + " tháng " + match.group(3) + " năm " + match.group(4)
    else:
        return 'ngày ' + match.group(2) + " tháng " + match.group(3) + " năm " + match.group(4)

def _date_month(match):
    if "ngày" in match.group(0):
        return match.group(2) + " tháng " + match.group(3)
    return 'ngày ' + match.group(2) + " tháng " + match.group(3)

def _month_year(match):
    if "tháng" in match.group(0):
        return match.group(2) + " năm " + match.group(3)
    return 'tháng ' + match.group(2) + " năm " + match.group(3)

def _remove_sticky_year(match):
    return "năm " + match.group(1) + " đến ngày " + match.group(2)

def normalize_datetime(text):
    patterns = {
        r'(ngày )?(\d{1,2})\/(\d{1,2})\/(\d{1,4})': _date_month_year,
        r'(ngày )?(\d{1,2})[-–](\d{1,2})[-–](\d{1,4})': _date_month_year,
        r'(ngày )?(\d{1,2})\/(\d{1,2})': _date_month ,
        r'(ngày )?(\d{1,2})[-–](\d{1,2})': _date_month,
        r'(tháng )?(\d{1,2})\/(\d{1,4})': _month_year,
        r'(tháng )?(\d{1,2})[-–](\d{1,4})': _month_year,
        r'năm (\d{4})(\d{1,2})': _remove_sticky_year
    }

    for pattern,function in patterns.items():
        text = re.sub(pattern, function, text)
    return text

def clean_html(text):
    try:
        doc = lxml.html.fromstring(text)
        cleaner = lxml.html.clean.Cleaner(style=True)
        doc = cleaner.clean_html(doc)
        return doc.text_content()
    except:
        return text

def tuple2dict(items):
    res = {}
    for k,v in items:
        data = res.get(k,False)
        if data is not None and data is not False:
            res[k] = res[k].append(str(v))
        else:
            res[k] = [str(v)]
    return res

# def get_type_answer(answer):
#     return any(char.isdigit() for char in answer)

def question_keyword():
    return read_file(os.path.join(root_path,'keyword_answer.json'),
                        extension_allow=["json"])

def format_answer(question,ans):
    keyword = question_keyword()
    answer_type = None
    for k,v in keyword.items():
        if k in question:
            answer_type = v
            break
    if answer_type == 'year' and "năm" not in ans:
        ans = "năm " + ans
    elif answer_type == 'month' and "tháng" not in ans:
        ans = "tháng " + ans
    return ans.lower()

def normalize_unicode(text):
    return unicodedata.normalize('NFKC',text)

def clean(text):
    text = normalize_datetime(text)
    text = clean_html(text)
    text = normalize_unicode(text)
    text = normalize_accent(text,accent_norm)
    text = remove_section(text)
    text = replace_stroke(text)
    text = clean_with_regex(text)
    text = replace_stroke(text)
    text = text.strip()
    return " ".join(text.split())

def remove_stopword(text_split:list):
    res = []
    for phrase in text_split:
        if phrase not in stopwords:
            phrase = phrase.replace("_"," ")
            res.append(phrase)
    return " ".join(res).replace("?","")


def remove_title(text,title):
    title = clean(title)
    if text.startswith(title):
        text = text[len(title):]
    return text.strip()

def remove_parentheses(text):
    pattern = r"\([^)]*\)"
    text = re.sub(pattern,'',text)
    return text.strip()

def remove_all_punc(text):
    text = text.replace("-"," - ")
    pattern = r"\p{P}+"
    return re.sub(pattern,"",text)

def remove_section(text):
    pattern = r"^==[^)]*==$"
    return re.sub(pattern," ",text)

def clean_question(question):
    text = normalize_accent(question)
    text = remove_all_punc(text)
    return text

def get_entity(text):
    lower_text = text.lower()
    lower_text_split = lower_text.split()
    text_split = text.split()
    tmp = ""
    entity = []

    for i,w in enumerate(text_split):
        if w != lower_text_split[i]:
            tmp += w + " "
            if i< len(text_split)-1 and text_split[i+1][0].islower():
                entity.append(tmp.strip())
                tmp = ""
            elif i == len(text_split)-1:
                entity.append(tmp.strip())
                tmp = ""
    entity.sort(key= lambda x: len(x.split()),reverse=True)
    entity = [e for e in entity if e.lower() not in stopwords]
    if len(entity)> 0 :
        if len(entity[0].split()) > 2:
            return [entity[0]]
        else:
            return entity
    else:
        return []

def _replace_stroke_1(match):
    return(match.group(0).replace("-"," "))

def _replace_stroke_2(match):
    return match.group(1).capitalize() + " " + match.group(2).capitalize() + " "

def replace_stroke(text):
    patterns = {
        r"(([^\s]+)-){2,6}": _replace_stroke_1,
        r"([^\d\s]+)-([^\d\s]+)\s": _replace_stroke_2,
        r"([^\d\s]+)-([^\d\s]+)$": _replace_stroke_2,
        r"\s+": " "
        # r"([^\d\s]+)-(\d+)$ | ([^\d\s]+)-(\d+)\s": _replace_stroke_2
    }
    for pattern, repl_func in patterns.items():
        text = re.sub(pattern,repl_func,text)
    return text.strip()


def clean_entity(title):
    title = normalize_accent(title)
    title = remove_parentheses(title)
    title = title.split(",")[0]
    if "-" in title:
        title = replace_stroke(title)    
    return title

def get_heading(text):
    ignore_heading = ["xem thêm","liên kết ngoài","tham khảo"]
    pattern = r"(\n== .+ ==\n)"
    res = []
    has_ignore = False
    for m in re.finditer(pattern, text):
        has_ignore = any([ignore in m.group(0).lower() for ignore in ignore_heading])
        if has_ignore:
            res.append((m.start(0), m.end(0)))
            break
        else:
            res.append((m.start(0), m.end(0)))
    return res, has_ignore

def split_paragraph(text):
    heading, has_ignore = get_heading(text)
    paragraphs = []
    current_index = 0
    if has_ignore:
        for i in range(len(heading)):
            paragraphs.append(text[current_index:heading[i][0]])
            current_index = heading[i][1]
    else:
        for i in range(len(heading)+1):
            if i == len(heading):
                paragraphs.append(text[current_index:])
            else:
                paragraphs.append(text[current_index:heading[i][0]])
                current_index = heading[i][1]
    return paragraphs