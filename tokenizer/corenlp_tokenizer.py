#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Simple wrapper around the Stanford CoreNLP pipeline.
Serves commands to a java subprocess running the jar. Requires java 8.
"""

import copy
import os
from vncorenlp import VnCoreNLP
from scripts.utils.util import normalize_accent
from .tokenizer import Tokens, Tokenizer
from . import DEFAULTS


def remove_punc(word,accept_punc = [',','.']):
    res = []
    for p in accept_punc:
        if p in word:
            word = word.replace(p,'')
    return word

class CoreNLPTokenizer(Tokenizer):

    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, wseg.
            classpath: Path to the corenlp directory of jars
            mem: Java heap memory
        """
        self.classpath = (kwargs.get('classpath') or
                          DEFAULTS['corenlp_classpath'])
        # self.classpath = "/data/truongnx/zalo_ai/VnCoreNLP"
        self.vncore_path = os.path.join(self.classpath,"VnCoreNLP-1.1.1.jar")
        self.annotators = copy.deepcopy(kwargs.get('annotators', set(['wseg','pos'])))
        self.annotators = ",".join(self.annotators)
        self.mem = kwargs.get('mem', '4g')
        self.corenlp = VnCoreNLP(self.vncore_path,annotators=self.annotators, max_heap_size='-Xmx'+self.mem)

    def join_token(self,tokens_dict:list,join_pos = ['Np']):
        res = [tokens_dict]
        for allow_pos in join_pos:
            tmp = ""
            latest_res = []
            for i,t in enumerate(res[-1]):
                t_form = t['form']
                t_tag = t['posTag']
                if t_tag != allow_pos:
                    is_end = True
                else:
                    if i < len(res[-1])-1:
                        is_end = False
                        tmp += t_form + " "
                    else:
                        
                        is_end = True
                        tmp += t_form + " "
                if is_end:
                    if tmp != "":
                        latest_res.append({'form':tmp.strip(),'posTag':allow_pos})
                        latest_res.append({'form':t_form.strip(),'posTag':t['posTag']})
                        tmp = ""
                    else:
                        latest_res.append({'form':t_form.strip(),'posTag':t['posTag']})
                    
            res.append(latest_res)
        return res[-1]

    def tokenize(self, text, accept_punc = [',','.']):

        text = text.strip()
        text_split = text.split()
        text_split = [remove_punc(x,accept_punc) for x in text_split]
        output = self.corenlp.annotate(text)
        sentences = []
        for s in output['sentences']:
            s = self.join_token(s)
            sentences += s
        data = []
        tmp = ""
        original_word_index = 0
        entities = []
        for i in range(len(sentences)):
            
            token = sentences[i]['form'].replace("_"," ")
            token = normalize_accent(token)
            len_token = len(token.split())
            original_word = " ".join(text_split[original_word_index:original_word_index+len_token])
            if sentences[i]['posTag'] == 'Np':
                entities.append(token)
            if token not in accept_punc:
                tmp += token
            if tmp == original_word:
                original_word_index += len_token
                if tmp:
                    data.append((
                                tmp,
                                original_word
                            ))
                    tmp = ""
            
            else:
                continue
        return Tokens(data, entities, self.annotators)
    
    def shutdown(self):
        self.corenlp.close()
