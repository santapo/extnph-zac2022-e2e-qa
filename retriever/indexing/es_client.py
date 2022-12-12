from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

class ES:
    def __init__(self,host,port,index_name):
        self.es = Elasticsearch(host+":"+port)
        self.index = index_name

    def put_data(self,data):
        requests = []
        """
            data: [{"id":"1123","title":"title","text":"text"}]
        """
        
        for d in data:
            request = d
            request["_op_type"] = "index"
            request['_id'] = d["id"]
            request["_index"] = self.index
            requests.append(request)
        bulk(self.es, requests)
        return 

    def query_context(self,question, re_ranking= None, k =5):
        script_query = {
            "size":k,
            "query":{
                "match":{
                    'text': question
                }
            }
        }
        response = self.es.search(
            index = self.index,
            body=script_query,
            request_timeout =3000
        )
        texts = []
        if response["hits"]["hits"]==[]:
            return []
        else:
            if re_ranking:
                titles = []
                texts = []
                scores = []
                for i,hit in enumerate(response["hits"]["hits"]):
                    if i >= k:break
                    else:
                        titles.append(hit['_source']['title'])
                        texts.append(hit['_source']['text'] )
                        scores.append((hit['_score'] ))
                id_sorted = re_ranking(question,titles,scores)
                return [texts[int(index)] for index in id_sorted]
            else:
                for i,hit in enumerate(response["hits"]["hits"]):
                    if i >= k:break
                    texts.append(hit['_source']['text'] )
                return texts

    def _script_base(self,list_must, list_should,k):
        res = {"size":k,
                "query" : {
                    "bool": {
                        "must":list_must,
                        "should": list_should,
                        "minimum_should_match" : 1,
                        }
                    }
            }
        return res

    def query_exact(self,origin_question,
                    question_tokens:list,
                    must_have = None, 
                    entity = None,
                    re_ranking= None,
                    k =5
                ):
        list_must = []
        list_should = []
        if must_have:
            for token in must_have:
                tmp = {"text" : token}
                list_must.append({"match_phrase":tmp})

        if entity:
            tmp = {"title" : entity}
            list_must.append({"match_phrase":tmp})

        for i,token in enumerate(question_tokens):
            tmp = {"text" : token}
            list_should.append({"match":tmp})

        list_should.append({"match":{'title':origin_question}})
        script_query = self._script_base(list_must, list_should, k)
        response = self.es.search(
            index = self.index,
            body=script_query,
            request_timeout =3000
            )
        texts = []
        if response["hits"]["hits"]==[]:
            return ""
        else:
            if re_ranking:
                titles = []
                texts = []
                scores = []
                for i,hit in enumerate(response["hits"]["hits"]):
                    if i >= k:break
                    else:
                        titles.append(hit['_source']['title'])
                        texts.append(hit['_source']['text'] )
                        scores.append((hit['_score'] ))
                id_sorted = re_ranking(origin_question,titles,scores)
                return [texts[int(index)] for index in id_sorted]
            else:
                for i,hit in enumerate(response["hits"]["hits"]):
                    if i >= k: break
                    texts.append(hit['_source']['text'] )
                return texts

    def query_title(self,query):
        script_query = {
            "size":5,
            "query":{
                "match_phrase":{
                    'title': query
                }
            }
        }
        response = self.es.search(
            index = self.index,
            body=script_query,
            request_timeout =3000
        )
        hits = response["hits"]["hits"]
        if hits==[]:
            return self.query_title_2(query)
        else:
            texts = response["hits"]["hits"][0]['_source']['title']
            return texts

    def query_title_2(self,query):
        script_query = {
            "size":5,
            "query":{
                "match":{
                    'title': query
                }
            }
        }
        response = self.es.search(
            index = self.index,
            body=script_query,
            request_timeout =3000
        )
        hits = response["hits"]["hits"]
        if hits==[]:
            return query
        else:
            texts = response["hits"]["hits"][0]['_source']['title']
            return texts

if __name__ == "__main__":
    query = "Việt Nam"
    es = ES("192.168.0.90","9200","wikipedia")
    print(es.query_title(query))
