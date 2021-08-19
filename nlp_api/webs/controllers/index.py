from re import U
from flask import Blueprint, request
from flask.json import jsonify
from flask import make_response
# from application import db
import jieba
from test.classification import text_classification
from test.auto_title_test import title
from test.poem_test import poem
from test.math_test import math_ques
from test.ner_test import ner_print
from test.gpt_article_continued import get_article
from test.relation_extract_test import relation_extract, predicate2id


from bert_seq2seq import load_chinese_base_vocab, Tokenizer
from bert_seq2seq import load_bert, load_gpt
from datetime import datetime
import torch
## 引入缓存
from application import cache

router_index = Blueprint("index_page", __name__)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
classify_target = ["财经", "彩票", "房产", "股票", "家居", "教育", "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐"]
ner_target = ["other", "address", "book", "company", "game", "government", "movie", "name", "organization", "position", "scene"]

model_dir = "../state_dict/"
vocab_path = model_dir + "roberta_wwm_vocab.txt"  # roberta模型字典的位置
model_name = "roberta"
nezha_name = "nezha"

# 模型位置
cls_model_path = model_dir + "bert_multi_classify_model.bin"
couplet_model_path = model_dir + "bert_model_poem_ci_duilian.bin"
math_model_path = model_dir + "bert_math_ques_model.bin"
ner_model_path = model_dir + "bert_ner_model_crf.bin"
auto_title_model_path = model_dir + "nezha_auto_title.bin"
gpt_article_model_path = model_dir + "gpt2_article_continued/pytorch_model.bin"
auto_relation_extract_model_path = model_dir + "nezha_relation_extract.bin"
## 文本分类
word2idx = load_chinese_base_vocab(vocab_path, simplfied=False)
tokenizer = Tokenizer(word2idx)
bert_classify = load_bert(word2idx, model_name=model_name, model_class="cls", target_size=len(classify_target))
bert_classify.set_device(device)
bert_classify.eval()
bert_classify.load_all_params(model_path=cls_model_path, device=device)
## 对联
bert_couplet = load_bert(word2idx, model_name=model_name, model_class="seq2seq")
bert_couplet.set_device(device)
bert_couplet.eval()
bert_couplet.load_all_params(model_path=couplet_model_path, device=device)
## 小学数学题
bert_math = load_bert(word2idx, model_name=model_name, model_class="seq2seq")
bert_math.set_device(device)
bert_math.eval()
bert_math.load_all_params(model_path=math_model_path, device=device)
## ner
bert_ner = load_bert(word2idx, model_name=model_name, model_class="sequence_labeling_crf", target_size=len(ner_target))
bert_ner.set_device(device)
bert_ner.eval()
bert_ner.load_all_params(model_path=ner_model_path, device=device)
## auto title
bert_auto_title = load_bert(word2idx, model_name=nezha_name, model_class="seq2seq")
bert_auto_title.set_device(device)
bert_auto_title.eval()
bert_auto_title.load_all_params(model_path=auto_title_model_path, device=device)
## gpt2 article continued
gpt_article = load_gpt(word2idx)
gpt_article.eval()
gpt_article.set_device(device)
gpt_article.load_pretrain_params(gpt_article_model_path)
## auto_relation_extract
auto_relation_extract_model = load_bert(word2idx, model_class="relation_extrac", model_name=nezha_name, target_size=len(predicate2id))
auto_relation_extract_model.eval()
auto_relation_extract_model.set_device(device)
auto_relation_extract_model.load_all_params(auto_relation_extract_model_path, device=device)


print("all models have beed loaded .")

@router_index.route("/")
def index():

    return "hello world test install"

# 请求拦截器 可以配置一个全局的拦截，当1s内请求次数超过5次时候，那么则暂停一段时间
@router_index.before_request
def before_user():
    global flag
    ## 在这里做一下缓存功能
    num = cache.get("number")
    if num is not None and num > 4:
        return "网站流量过大，请稍后访问。"
    if num is None :
        num = 0
    num = num + 1
    cache.set("number", num, timeout=3)
    return None

@router_index.route("/auto_fenci", methods=["GET", "POST"])
def auto_fenci():
    resp = {"code": 200, "msg": "success", "data": [], "ret": 1}
    req = request.values
    word = req["text"] if "text" in req else ""
    if word == "":
        resp["ret"] = 0
        resp["msg"] = "请重新输入句子"
        return jsonify(resp)
    word_split = jieba.cut(word)
    word_split = "\t".join(word_split)
    print(f"输出结果: {word_split}")
    resp["data"] = word_split

    return jsonify(resp)

@router_index.route("/text_classify", methods=["GET", "POST"])
def text_classify():
    resp = {"code": 200, "msg": "success", "data": [], "ret": 1}
    req = request.values
    text = req["text"] if "text" in req else ""
    if text == "":
        resp["ret"] = 0
        resp["msg"] = "请重新输入句子"
        return jsonify(resp)
    res = text_classification(bert_classify, text, tokenizer, device=device)
    print(f"输出结果: {res}")
    resp["data"] = res

    return jsonify(resp)

@router_index.route("/auto_title", methods=["GET", "POST"])
def auto_title():
    resp = {"code": 200, "msg": "success", "data": [], "ret": 1}
    req = request.values
    text = req["text"] if "text" in req else ""
    if text == "":
        resp["ret"] = 0
        resp["msg"] = "请重新输入句子"
        return jsonify(resp)
    res = title(bert_auto_title, text)
    print(f"输出结果: {res}")
    resp["data"] = res

    return jsonify(resp)

@router_index.route("/auto_couplet", methods=["GET", "POST"])
def auto_couplet():
    resp = {"code": 200, "msg": "success", "data": [], "ret": 1}
    req = request.values
    text = req["text"] if "text" in req else ""
    if text == "":
        resp["ret"] = 0
        resp["msg"] = "请重新输入句子"
        return jsonify(resp)
    res = poem(bert_couplet, text)
    print(f"输出结果: {res}")
    resp["data"] = res
    return jsonify(resp)

@router_index.route("/auto_math", methods=["GET", "POST"])
def auto_math():
    resp = {"code": 200, "msg": "success", "data": [], "ret": 1}
    req = request.values
    text = req["text"] if "text" in req else ""
    if text == "":
        resp["ret"] = 0
        resp["msg"] = "请重新输入句子"
        return jsonify(resp)
    res = math_ques(bert_math, text)
    try:
        res_res = eval(res)
        resp["data"] = res + " = " + str(res_res)

    except Exception as e :
        # print(e)
        resp["data"] = "no result"
        return jsonify(resp)
    print(f"输出结果: { resp['data']}")
    return jsonify(resp)

@router_index.route("/auto_ner", methods=["GET", "POST"])
def auto_ner():
    resp = {"code": 200, "msg": "success", "data": [], "ret": 1}
    req = request.values
    text = req["text"] if "text" in req else ""
    if text == "":
        resp["ret"] = 0
        resp["msg"] = "请重新输入句子"
        return jsonify(resp)
    res = ner_print(bert_ner, text, word2idx, tokenizer, ner_target, device=device)
    resp["data"] = res

    print(f"输出结果: { resp['data']}")
    return jsonify(resp)

@router_index.route("/auto_article", methods=["GET", "POST"])
def auto_article():
    resp = {"code": 200, "msg": "success", "data": [], "ret": 1}
    req = request.values
    text = req["text"] if "text" in req else ""
    if text == "":
        resp["ret"] = 0
        resp["msg"] = "请重新输入句子开头"
        return jsonify(resp)
    res = get_article(gpt_article, text, out_max_length=300)
    resp["data"] = res

    print(f"输出结果: { resp['data']}")
    return jsonify(resp)

@router_index.route("/auto_relation_extract", methods=["GET", "POST"])
def auto_relation_extract():
    resp = {"code": 200, "msg": "success", "data": [], "ret": 1}
    req = request.values
    text = req["text"] if "text" in req else ""
    if text == "":
        resp["ret"] = 0
        resp["msg"] = "请重新输入"
        return jsonify(resp)
    res = relation_extract(auto_relation_extract_model, text, word2idx, tokenizer, device=device)
    resp["data"] = res

    print(f"输出结果: { resp['data']}")
    return jsonify(resp)