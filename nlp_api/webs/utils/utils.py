
from bert_seq2seq import load_bert, load_gpt
import torch

def load_model(word2idx, model_path, model_name=None, model_class=None,
               target_size=None, is_gpt=False, is_all_params=True, device=torch.device("cpu")):
    if is_gpt:
        model = load_gpt(word2idx)
        model.eval()
        model.set_device(device)
        if is_all_params:
            model.load_all_params(model_path, device=device)
        else :
            model.load_pretrain_params(model_path)
        return model
    if target_size is not None:
        model = load_bert(word2idx, model_name=model_name, model_class=model_class, target_size=target_size)
        model.set_device(device)
        model.eval()
        if is_all_params:
            model.load_all_params(model_path, device=device)
        else :
            model.load_pretrain_params(model_path)
    else :
        model = load_bert(word2idx, model_name=model_name, model_class=model_class)
        model.set_device(device)
        model.eval()
        if is_all_params:
            model.load_all_params(model_path, device=device)
        else :
            model.load_pretrain_params(model_path)
    return model