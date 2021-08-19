import torch 
import sys
from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
import os
import json
import time
import bert_seq2seq
from bert_seq2seq.utils import load_bert

target = ["财经", "彩票", "房产", "股票", "家居", "教育", "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐"]

def text_classification(model, text, tokenizer, device="cpu"):
    with torch.no_grad():
        text, text_ids = tokenizer.encode(text)
        text = torch.tensor(text, device=device).view(1, -1)
        return target[torch.argmax(model(text)).item()]
        


