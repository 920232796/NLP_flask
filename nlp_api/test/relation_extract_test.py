import torch 
import numpy as np
import json
from bert_seq2seq import Tokenizer, load_chinese_base_vocab
from bert_seq2seq import load_bert


predicate2id, id2predicate = {}, {}
with open('../state_dict/all_50_schemas', encoding="utf-8") as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

def search_subject(token_ids, subject_labels, idx2word):
    # subject_labels: (lens, 2)
    if type(subject_labels) is torch.Tensor:
        subject_labels = subject_labels.numpy()
    if type(token_ids) is torch.Tensor:
        token_ids = token_ids.cpu().numpy()
    subjects = []
    subject_ids = []
    start = -1
    end = -1
    for i in range(len(token_ids)):
        if subject_labels[i, 0] > 0.5:
            start = i
            for j in range(len(token_ids)):
                if subject_labels[j, 1] > 0.5:
                    subject_labels[j, 1] = 0
                    end = j
                    break
            if start == -1 or end == -1:
                continue
            subject = ""
            for k in range(start, end + 1):
                subject += idx2word[token_ids[k]]
            # print(subject)
            subject_ids.append([start, end])
            start = -1
            end = -1
            subjects.append(subject)

    return subjects, subject_ids

def search_object(token_ids, object_labels, idx2word):
    objects = []
    if type(object_labels) is torch.Tensor:
        object_labels = object_labels.numpy()
    if type(token_ids) is torch.Tensor:
        token_ids = token_ids.cpu().numpy()
    start = np.where(object_labels[:, :, 0] > 0.5)
    end = np.where(object_labels[:, :, 1] > 0.5)
    for _start, predicate1 in zip(*start):
        for _end, predicate2 in zip(*end):
            if _start <= _end and predicate1 == predicate2:
                object_text = ""
                for k in range(_start, _end + 1):
                    # print(token_ids(k))
                    object_text += idx2word[token_ids[k]]
                objects.append(
                   (id2predicate[predicate1], object_text)
                )
                break 
    
    return objects

def relation_extract(model, text, word2idx, tokenizer, device="cpu"):
    idx2word = {v: k for k , v in word2idx.items()}
    with torch.no_grad():
        token_ids_test, segment_ids = tokenizer.encode(text, max_length=256)
        token_ids_test = torch.tensor(token_ids_test, device=device).view(1, -1)
        # 先预测subject
        pred_subject = model.predict_subject(token_ids_test)
        pred_subject = pred_subject.squeeze(0)
        subject_texts, subject_idss = search_subject(token_ids_test[0], pred_subject.cpu(), idx2word)
        if len(subject_texts) == 0:
            return "没有预测出任何信息"
        result_info = ""
        for sub_text, sub_ids in zip(subject_texts, subject_idss):
            result_info += "s is " + str(sub_text) + "\n"
            sub_ids = torch.tensor(sub_ids, device=device).view(1, -1)
            # print("sub_ids shape is " + str(sub_ids))
            object_p_pred = model.predict_object_predicate(token_ids_test, sub_ids)
            res = search_object(token_ids_test[0], object_p_pred.squeeze(0).cpu(), idx2word)
            result_info += "p and o is " + str(res) + "\n"
        return result_info





