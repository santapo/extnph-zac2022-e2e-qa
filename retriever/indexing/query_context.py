import json
def count_intersection(text_1:list,text_2:list):
    return len(set(text_1) & set(text_2))

def gen_permutation(text):
    words = text.split()
    p = []
    for len_permutation in range(1,len(words)+1):
        for i in range(len(words)):
            if i+len_permutation <= len(words):
                word = " ".join(words[i:i+len_permutation])
                p.append(word.strip())
    return p

def max_len_common(context:str,question_permutation:list):
    max_len = 0 
    for phrase in question_permutation:
        if phrase in context:
            max_len = len(phrase.split())
            break
    return max_len

def get_contexts(question,lines):
    question_split = question.split()
    question_permutation = gen_permutation(question)[::-1]
    best_common = 0
    longest_common = 0
    contexts = []

    for line in lines:
        line = line.replace("\n","")
        num_common = count_intersection(question_split,line.split())
        if num_common > 0:
            length_common = max_len_common(line,question_permutation)
            if length_common > longest_common:
                longest_common = length_common
                best_common = num_common
                contexts = [line]
            elif length_common == longest_common:
                if num_common > best_common:
                    best_common = num_common
                    contexts = [line]
                elif num_common == best_common:
                    contexts.append(line)
    return contexts

# def  get_contexts(question_kws,knowledge_dict,max_context=3):
#     context_ids = []
#     for k,v in knowledge_dict.items():
#         num_common = count_intersection(question_kws,v)
#         if num_common == len(question_kws):
#             context_ids = [(k,num_common)]
#             break
#         elif num_common > 0:
#             context_ids.append((k,num_common))
#             print(question_kws,v)
#     context_ids.sort(key=lambda x: x[1],reverse=True)
#     context_ids = [x[0] for x in context_ids][:max_context]
#     return context_ids

