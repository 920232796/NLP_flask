import torch 
import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_seq2seq")
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
import os
import json
import time
import bert_seq2seq
from bert_seq2seq.utils import load_bert

def math_ques(model, text):
    with torch.no_grad():
        return model.generate(text, beam_size=4)