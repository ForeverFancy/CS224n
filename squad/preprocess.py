import json
import spacy
from collections import Counter

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

def preprocess():
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
    word_counter = Counter()
    char_counter = Counter()
    path = './data/tmp.json'
    nlp = spacy.blank("en")
    total = 0
    examples = []
    eval_examples = {}
    
    with open(path, "r") as f:
        source = json.load(f)
        for article in source['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context'].replace("''", '" ').replace("``", '" ')
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
                    questions_chars = [list(token) for token in questions_tokens]
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
                        "uuid":qa['id']
                    }
    print("Process {} questions in total.".format(total))
    
    return example, eval_examples
                            

                    




if __name__ == "__main__":
    preprocess()
