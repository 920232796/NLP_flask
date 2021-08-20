
from bert_seq2seq import load_bert, load_gpt
import torch
import redis


class RedisCache:
    def __init__(self, host, port, decode_responses=True):
        self.pool = redis.ConnectionPool(host=host, port=port, decode_responses=decode_responses) # redis 连接池
        print("构建redis连接池成功")

    def get_cache(self, text, prefix):
        r = redis.Redis(connection_pool=self.pool)
        cache_content = r.get(text + "##" + prefix)
        print(f"获取到缓存数据{cache_content}")
        if cache_content is None :
            return ""
        return cache_content

    def set_cache(self, text, prefix, res):
        r = redis.Redis(connection_pool=self.pool)
        k = text + "##" + prefix
        r.set(k, res)  #  # key是"food" value是"mutton" 将键值对存入redis缓存
        print(f"set数据k:{k}, v: {res}")

    def keys(self):
        r = redis.Redis(connection_pool=self.pool)
        return r.keys()

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