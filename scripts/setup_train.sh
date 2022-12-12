#!/bin/bash

echo Downloading dataset...
mkdir -p data/
wget https://dumps.wikimedia.org/viwiki/20220620/viwiki-20220620-pages-articles.xml.bz2 -O data/viwiki-20220620-pages-articles.xml.bz2
wget https://dl-challenge.zalo.ai/e2e-question-answering/wikipedia_20220620_cleaned.zip -O data/wikipedia_20220620_cleaned.zip
unzip -qq data/wikipedia_20220620_cleaned.zip -d data/ 
wget https://dl-challenge.zalo.ai/e2e-question-answering/e2eqa-train+public_test-v1.zip -O data/e2eqa-train+public_test-v1.zip
unzip -qq data/e2eqa-train+public_test-v1.zip -d data/
wget https://dl-challenge.zalo.ai/e2e-question-answering/e2eqa-private_test_v3.zip -O data/e2eqa-private_test_v3.zip
unzip -qq data/e2eqa-private_test_v3.zip -d data/

echo Installing python\'s dependencies...
pip3 install -r requirements.txt