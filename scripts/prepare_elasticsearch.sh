#!/bin/bash

wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.13.4-amd64.deb
dpkg -i elasticsearch-7.13.4-amd64.deb
service elasticsearch start
rm -r elasticsearch-7.13.4-amd64.deb

python3 create_documents.py
python3 create_sub_documents.py