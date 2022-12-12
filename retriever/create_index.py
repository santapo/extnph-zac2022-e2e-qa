from elasticsearch import Elasticsearch

INDEX_FILE = {
        "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 1
        },
        "mappings": {
            "dynamic": "true",
            "_source": {
                    "enabled": "true"
                },
            "properties": {
                "id": {
                    "type": "keyword"
                },
                "title": {
                    "type":"text"
                },
                "text": {
                    "type": "text"
                },
                "title_key":{
                    "type": "keyword"
                }
            }
        }
}

def create_index(host, port, index_name):
    client = Elasticsearch(":".join([host,port]))
    client.indices.delete(index=index_name, ignore=[404])
    client.indices.create(index=index_name, body=INDEX_FILE)


if __name__ == "__main__":
    client = Elasticsearch("192.168.0.90:9200")
    index_name = "wikipedia"


    create_index(index_name,client)
    client.indices.put_settings(index=index_name,
                        body= {"index" : {
                                "max_result_window" : 10000000
                              }})