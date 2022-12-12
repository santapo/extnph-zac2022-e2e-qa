import numpy as np

def count_intersection(text_1:list,text_2:list):
    tmp = len(set(text_1) & set(text_2))
    return tmp if tmp > 0 else 0.7

def normalize_score(scores):
    max_score = max(scores)
    min_score = min(scores)
    return (np.array(scores) - min_score) / (max_score - min_score)

# def re_ranking(db,query,doc_ids,tfidf_scores):
#     query = query.lower()
#     if len(doc_ids) == 1:
#         return doc_ids
#     else:
#         scores = normalize_score(tfidf_scores).tolist()
#         # print("query: ",query)
#         query_split = query.split()
#         re_index = []
#         for i,doc_id in enumerate(doc_ids):
#             title = db.get_doc_title(doc_id).lower()
#             # print(title)
#             title_split = title.split()
#             num_intersect = count_intersection(query_split,title_split)
#             new_score = scores[i] * num_intersect / len(title_split)
#             re_index.append((doc_id,new_score))
#         re_index.sort(key = lambda x:x[1], reverse = True)

#         # print('Context: ',db.get_doc_text(re_index[0][0]))
#         return [i[0] for i in re_index ]
def re_ranking(query,titles,scores,do_normalize_score = True):
    query = query.lower()
    if len(titles) == 1:
        return [0]
    else:
        if do_normalize_score:
            scores = normalize_score(scores).tolist()
        else:
            scores = [1] * len(titles)
        query_split = query.split()
        re_index = []
        for i,title in enumerate(titles):
            title = title.lower()
            # print(title)
            title_split = title.split()
            num_intersect = count_intersection(query_split,title_split)
            new_score = scores[i] * num_intersect / len(title_split)
            re_index.append((i,new_score))
        re_index.sort(key = lambda x:x[1], reverse = True)

        # print('Context: ',db.get_doc_text(re_index[0][0]))
        return [i[0] for i in re_index ]