import json
import spacy
from collections import Counter
from tqdm import tqdm
import json
import numpy as np


def get_token_idx_span(text: str, tokens: list([str])):
    '''
    For each word in tokens, get its index span in the text (begin position, end position).

    @param text(str): input text of paragraph.

    @param tokens(list[str]): input token list.
    
    @return spans(list): token index span in the text.
    '''
    current = 0
    spans = []
    for token in tokens:
        # 由于原文中可能包含其他字符，使用 find 是准确方法。
        current = text.find(token, current)
        assert (current >= 0)
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def word_tokenize(nlp, sentence):
    # 分词
    doc = nlp(sentence)
    return [token.text for token in doc]


def get_examples(word_counter: Counter, char_counter: Counter, path: str):
    '''
    @return examples(list): list of example, each contains
    {
        "context_tokens": paragraph context represented in words,
        "context_chars": paragraph context represented in characters of each word,
        "question_tokens": question represented in words,
        "question_chars": question represented in characters of each word,
        "y1s": start index of answer in the context,
        "y2s": end index of answer in the context,
        "id": question id (start from 1)
    }

    @return eval_examples(list): list of example used for evaluation, each contains
    {
        "context": context of paragraph,
        "question": question (string),
        "spans": start position and end position for each word in the context,
        "answers": given answers,
        "uuid": raw question id
    }
    '''
    nlp = spacy.blank("en")
    total = 0
    examples = []
    eval_examples = {}

    with open(path, "r") as f:
        source = json.load(f)
        for article in source['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context'].replace(
                    "''", '" ').replace("``", '" ')
                # print(context)
                context_tokens = word_tokenize(nlp, context)
                # print(context_tokens)
                context_chars = [list(token) for token in context_tokens]
                # print(context_chars)
                spans = get_token_idx_span(context, context_tokens)
                print(len(paragraph['qas']))
                for token in context_tokens:
                    # 给每个词频都增加问题的数量（一种加权）？
                    word_counter[token] += len(paragraph['qas'])
                    for ch in token:
                        char_counter[ch] += len(paragraph['qas'])

                for qa in paragraph['qas']:
                    total += 1
                    question = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    questions_tokens = word_tokenize(nlp, question)
                    questions_chars = [list(token)
                                       for token in questions_tokens]
                    for token in questions_tokens:
                        word_counter[token] += 1
                        for ch in token:
                            char_counter[ch] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa['answers']:
                        text = answer['text']
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(text)
                        answer_texts.append(text)
                        answer_span = []
                        for index, span in enumerate(spans):
                            #  把答案范围内的词的索引全部添加到 answer_span 中
                            if span[0] < answer_end and span[1] > answer_start:
                                answer_span.append(index)
                            # 分别是开始词的索引和结束词的索引（文章中的第几个词）
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {
                        "context_tokens": context_tokens,
                        "context_chars": context_chars,
                        "questions_tokens": questions_tokens,
                        "questions_chars": questions_chars,
                        "y1s": y1s,
                        "y2s": y2s,
                        "id": total
                    }
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context,
                        "question": question,
                        "spans": spans,
                        "answers": answer_texts,
                        "uuid": qa['id']
                    }
    print("Process {} questions in total.".format(total))

    return examples, eval_examples


def get_embedding(counter: Counter, data_type, limit=-1, emb_file=None, vec_size=None, num_vectors=None):
    '''
    :param counter:
    :param data_type:
    :param limit:
    :param emb_file:
    :param vec_size:
    :param num_vectors:
    :return emb_mat: shape(num_token, vec_size) get an embedding by index
    :return token2idx: dict get a token's index
    '''
    print("Getting word embedding...")

    # 丢弃出现次数低于 limit 的单词
    filtered_elements = [k for k, v in counter.items() if v > limit]

    # Building embedding dictionary...
    # key: word -> value: vector
    embedding_dict = {}
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=num_vectors):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                # same as "if word in counter and counter[word] > limit"
                if word in filtered_elements:
                    embedding_dict[word] = vector
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]

    print("Finish getting embedding dictionary...")

    NULL = "--NULL--"
    # Out of vocabulary
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx,
                      token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]

    idx2emb_dict = {idx: embedding_dict[token]
                    for idx, token in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]

    # emb_mat: token -> index -> embedding

    return emb_mat, token2idx_dict


def is_answerable(example):
    return len(example["y1s"]) > 0 and len(example["y2s"]) > 0


def build_features(args, examples, data_type, out_file, word2idx_dict: dict, char2idx_dict: dict, is_test=False):
    '''
    Save context indexes, context char indexes, ques indexes, ques char indexes,
    ques start, ques end and ques ids to outfile.
    :param args:
    :param examples:
    :param data_type:
    :param out_file:
    :param word2idx_dict:
    :param char2idx_dict:
    :param is_test:
    :return: meta
    '''
    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    ans_limit = args.ans_limit
    char_limit = args.char_limit

    total = 0
    context_indexes = []
    context_char_indexes = []
    ques_indexes = []
    ques_char_indexes = []
    y1s = []
    y2s = []
    ids = []
    meta = {}

    def drop_example(example, is_test_):
        '''
        Decide whether to drop the example.
        :param example:
        :param is_test_:
        :return:
        '''
        if is_test_:
            return False
        else:
            if len(example["context_tokens"]) > para_limit \
                    or len(example["questions_tokens"]) > ques_limit:
                return True
            if is_answerable(example):
                for i in range(len(example["y2s"])):
                    if abs(example["y2s"][i] - example["y1s"][i]) <= ans_limit:
                        return False
                return True
        return False

    for i, example in tqdm(enumerate(examples)):

        if drop_example(example, is_test):
            continue

        total += 1

        def _get_word(word: str):
            tmp = (word, word.lower(), word.capitalize(), word.upper())
            for item in tmp:
                if item in word2idx_dict:
                    return word2idx_dict[item]

        def _get_char(char: str):
            if char in char2idx_dict:
                return char2idx_dict[char]

        context_tokens = example["context_tokens"]
        context_idx = np.zeros((para_limit), dtype=np.int32)
        for i, token in enumerate(context_tokens):
            context_idx[i] = _get_word(token)
        context_indexes.append(context_idx)

        context_char_index = np.zeros((para_limit, char_limit), dtype=np.int32)
        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_index[i][j] = _get_char(char)
        context_char_indexes.append(context_char_index)

        ques_idx = np.zeros((ques_limit), dtype=np.int32)
        for i, token in enumerate(example["questions_tokens"]):
            ques_idx[i] = _get_word(token)
        ques_indexes.append(ques_idx)

        ques_char_idx = np.zeros((ques_limit, char_limit), dtype=np.int32)
        for i, token in enumerate(example["questions_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idx[i][j] = _get_char(char)
        ques_char_indexes.append(ques_char_idx)

        if is_answerable(example):
            # select one answer.
            start, end = example["y1s"][-1], example["y2s"][-1]
        else:
            start, end = -1, -1

        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])

    np.savez(out_file,
             context_indexes=np.array(context_indexes),
             context_char_indexes=np.array(context_char_indexes),
             ques_indexes=np.array(ques_indexes),
             ques_char_indexes=np.array(ques_char_indexes),
             y1s=np.array(y1s),
             y2s=np.array(y2s),
             ids=np.array(ids))
    meta["total"] = total
    return meta


def save(path, obj, message=None):
    if message is not None:
        print("Saving {}".format(message))
        with open(path, "w") as f:
            json.dump(object, f)


def pre_process(args):
    # 使用训练集建立词典和嵌入矩阵
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = get_examples(
        word_counter, char_counter, "train.json")
    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, 'word', emb_file=args.glove_file, vec_size=args.glove_dim, num_vectors=args.glove_num_vecs)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, 'char', emb_file=None, vec_size=args.char_dim)

    dev_examples, dev_eval = get_examples(
        word_counter, char_counter, "dev.json")
    build_features(args, train_examples, "train",
                   args.train_record_file, word2idx_dict, char2idx_dict)
    dev_meta = build_features(
        args, dev_examples, "dev", args.dev_record_file, word2idx_dict, char2idx_dict)
    if args.include_test_examples:
        test_examples, test_eval = get_examples(
            word_counter, char_counter, "test.json")
        save(args.test_eval_file, test_eval, message="test eval")
        test_meta = build_features(args, test_examples, "test",
                                   args.test_record_file, word2idx_dict, char2idx_dict, is_test=True)
        save(args.test_meta_file, test_meta, message="test meta")

    save(args.word_emb_file, word_emb_mat, message="word embedding")
    save(args.char_emb_file, char_emb_mat, message="char embedding")
    save(args.train_eval_file, train_eval, message="train eval")
    save(args.dev_eval_file, dev_eval, message="dev eval")
    save(args.word2idx_file, word2idx_dict, message="word dictionary")
    save(args.char2idx_file, char2idx_dict, message="char dictionary")
    save(args.dev_meta_file, dev_meta, message="dev meta")


if __name__ == "__main__":
    get_examples()
