import torch 
import torch.nn as nn 
import sys
sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_seq2seq")
import numpy as np
import os
import json
import time
import bert_seq2seq
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from bert_seq2seq.utils import load_bert


def poem(model, text):
    with torch.no_grad():
        text = text + "##对联"
        return model.generate(text, beam_size=4, is_poem=False)



