import jieba
token2id, id2token = {}, {}
with open("//data/vocabs/token2id", mode='r', encoding='utf-8') as infile:
    for row in infile:
        row = row.strip()
        token, token_id = row.split('¤')[0], int(row.split('¤')[1])
        token2id[token] = token_id
        id2token[token_id] = token
dep = "林徽因什么理由 拒绝了 徐志摩而选择梁思成为终身伴侣？"
print(token2id[" "])
