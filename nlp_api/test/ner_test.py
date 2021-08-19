import torch



def viterbi_decode(nodes, trans):
    """
    维特比算法 解码
    nodes: (seq_len, target_size)
    trans: (target_size, target_size)
    """
    with torch.no_grad():
        scores = nodes[0]
        scores[1:] -= 100000 # 刚开始标签肯定是"O"
        target_size = nodes.shape[1]
        seq_len = nodes.shape[0]
        labels = torch.arange(0, target_size).view(1, -1)
        path = labels
        for l in range(1, seq_len):
            scores = scores.view(-1, 1)
            M = scores + trans + nodes[l].view(1, -1)
            scores, ids = M.max(0)
            path = torch.cat((path[:, ids], labels), dim=0)
            # print(scores)
        # print(scores)
        return path[:, scores.argmax()]

def ner_print(model, test_data, word2idx, tokenizer, target, device="cpu"):
    model.eval()
    idxtword = {v: k for k, v in word2idx.items()}
    trans = model.state_dict()["crf_layer.trans"]

    decode = []
    text_encode, text_ids = tokenizer.encode(test_data)
    text_tensor = torch.tensor(text_encode, device=device).view(1, -1)
    out = model(text_tensor).squeeze(0) # 其实是nodes
    labels = viterbi_decode(out, trans)
    starting = False
    for l in labels:
        if l > 0:
            label = target[l.item()]
            decode.append(label)
        else :
            decode.append("other")
    flag = 0
    res = {}

    decode_text = [idxtword[i] for i in text_encode]
    for index, each_entity in enumerate(decode):
        if each_entity != "other":
            if flag != each_entity:
                cur_text = decode_text[index]
                if each_entity in res.keys():
                    res[each_entity].append(cur_text)
                else :
                    res[each_entity] = [cur_text]
                flag = each_entity
            elif flag == each_entity:
                res[each_entity][-1] += decode_text[index]
        else :
            flag = 0
    return res
