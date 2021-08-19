
import torch 
from bert_seq2seq import load_gpt
from bert_seq2seq import load_chinese_base_vocab

def get_article(model, text, out_max_length=300, top_k=30):

    return model.sample_generate(text, out_max_length=out_max_length, top_k=top_k)