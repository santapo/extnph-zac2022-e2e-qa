#!/bin/bash

echo Downloading support materials...
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar -O data/VnCoreNLP-1.1.1.jar
wget https://dl-challenge.zalo.ai/e2e-question-answering/wikipedia_20220620_cleaned.zip -O data/wikipedia_20220620_cleaned.zip
unzip -qq data/wikipedia_20220620_cleaned.zip -d data/ 

echo Installing Elasticsearch...
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.13.4-amd64.deb
dpkg -i elasticsearch-7.13.4-amd64.deb
service elasticsearch start
rm -r elasticsearch-7.13.4-amd64.deb

echo Installing python\'s dependencies...
pip3 install -r requirements.txt

echo Indexing...
python3 create_documents.py
python3 create_sub_documents.py