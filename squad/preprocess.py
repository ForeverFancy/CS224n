import transformers
from collections import Counter
import json
import spacy
from spacy.lang.en import English
import torch
from tqdm import tqdm
import pickle


def get_token_idx_span(text: str, tokens: list([str])):
    '''
    For each word in tokens, get its index span in the text (begin position, end position).

    @param text (str): input text of paragraph.

    @param tokens (list[str]): input token list.

    @return spans (list): token index span in the text.
    '''
    current = 0
    spans = []
    for token in tokens:
        # 由于原文中可能包含其他字符，使用 find 是准确方法，找到在原文中出现的位置。
        current = text.find(token, current)
        assert (current >= 0)
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def word_tokenize(nlp, sentence):
    # 分词
    doc = nlp(sentence)
    return [token.text for token in doc if token.pos_ != 'PUNCT']


def get_examples(path: str, training: bool):
    '''
    @param path (str): input file path

    @param training (bool): set whether it is training mode

    @return examples (list): list of example, each contains
    {
        "id": qas_id,

        "context": context,

        "question": question,

        "answer": answer_text,

        "start_position": start_position,

        "end_position": end_position,

        "is_impossible": is_impossible
    }
    '''
    total = 0
    examples = []
    eval_examples = {}
    # language = English()
    nlp = spacy.load('en')

    with open(path, "r") as f:
        source = json.load(f)
        for article in source['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context'].replace(
                    "''", '" ').replace("``", '" ')
                # print(context)
                context_tokens = word_tokenize(nlp, context)
                # print(context)
                spans = get_token_idx_span(context, context_tokens)
                # print(spans)
                # print(len(paragraph['qas']))

                for qa in paragraph['qas']:
                    qas_id = qa['id']
                    is_impossible = qa["is_impossible"]
                    total += 1
                    question = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    question_tokens = word_tokenize(nlp, question)

                    start_position = 0               # Index of the answer start position
                    end_position = 0                 # Index of the answer end position
                    answer_text = None

                    if training:
                        if not is_impossible:
                            # when training, we only use the first answer.
                            answer = qa['answers'][0]
                            answer_text = answer['text']
                            start = answer['answer_start']
                            end = start + len(answer_text)

                            answer_span = []
                            for index, span in enumerate(spans):
                                #  把答案范围内的词的索引全部添加到 answer_span 中
                                if span[0] < end and span[1] > start:
                                    answer_span.append(index)
                            # 分别是开始词的索引和结束词的索引（文章中的第几个词）
                            start_position, end_position = answer_span[0], answer_span[-1]

                        example = {
                            "id": qas_id,
                            "context": context_tokens,
                            "question": question_tokens,
                            "answer": answer_text,
                            "start_position": start_position,
                            "end_position": end_position,
                            "is_impossible": is_impossible
                        }
                        examples.append(example)

                    else:
                        y1s, y2s = [], []
                        answer_texts = []
                        # Use all answer when evaluating.
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
                        # TODO: add eval examples.
                        pass

    print("Process {} questions in total.".format(total))
    # print(examples)

    return examples


def build_feature(tokenizer: transformers.BertTokenizer, examples: list, max_length: int = None):
    '''
    @param tokenizer (transformers.BertTokenizer): tokenzier to convert token to ids

    @param examples (list): input examples

    @param maxlength (int): set max length to cut off example sequence

    @return examples (list): new examples with input feature
    '''

    if max_length is not None:
        length = max_length
    else:
        length = 1e3

    for example in examples:
        context = tokenizer.convert_tokens_to_ids(
            example['context'][: min(length, len(example['context']))])
        # print(context)
        question = tokenizer.convert_tokens_to_ids(
            example['question'][: min(length, len(example['question']))])
        # print(question)
        out = tokenizer.prepare_for_model(context, question, return_token_type_ids=True, return_attention_mask=True)
        inputs = out['input_ids']
        token_type_ids = out['token_type_ids']
        attention_mask = out['attention_mask']
        # print(inputs)
        # print(token_type_ids)
        # print(attention_mask)

        example['input_feature'] = inputs
        example['token_type_ids'] = token_type_ids
        example['attention_mask'] = attention_mask

    return examples


def save_examples(examples: list, para_limit: int = None, ques_limit: int = None, is_test: bool = False):

    def _drop_example(example: dict, is_test_: bool):
        if is_test_:
            return False
        else:
            if para_limit is not None and len(example["context_tokens"]) > para_limit \
                    or ques_limit is not None and len(example["questions_tokens"]) > ques_limit:
                return True
        return False

    input_features = []
    token_type_ids = []
    attention_mask = []
    answer_text = []
    start_positions = []
    end_positions = []
    is_impossible = []
    ids = []
    length = []

    # padding input feature to max length.
    for i, example in enumerate(examples):
        length.append(len(example['input_feature']))

    maxlength = max(length)
    print(maxlength)

    for i, example in enumerate(examples):
        input_features.append(example['input_feature'] + [0] * (maxlength - length[i]))
        attention_mask.append(example['attention_mask'] + [0] * (maxlength - length[i]))
        token_type_ids.append(example['token_type_ids'] + [0] * (maxlength - length[i]))


    for i, example in tqdm(enumerate(examples)):
        if _drop_example(example, is_test):
            continue

        ids.append(example['id'])
        is_impossible.append(example['is_impossible'])
        start_positions.append(example['start_position'])
        end_positions.append(example['end_position'])
        answer_text.append(example['answer'])

    torch.save(torch.tensor(input_features, dtype=torch.long),
               './save/input_feature.pt')
    torch.save(torch.tensor(attention_mask, dtype=torch.long),
               './save/attention_mask.pt')
    torch.save(torch.tensor(token_type_ids, dtype=torch.long),
               './save/token_type_ids.pt')
    torch.save(torch.tensor(is_impossible, dtype=torch.long),
               './save/is_impossible.pt')
    torch.save(torch.tensor(start_positions, dtype=torch.long),
               './save/start_positions.pt')
    torch.save(torch.tensor(end_positions, dtype=torch.long),
               './save/end_positions.pt')

    with open('./save/ids.pt', 'wb') as f:
        pickle.dump(ids, f)

    with open('./save/answer_text.pt', "wb") as f:
        pickle.dump(answer_text, f)
    
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(input_features, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(token_type_ids, dtype=torch.long),
        torch.tensor(is_impossible, dtype=torch.long),
        torch.tensor(start_positions, dtype=torch.long),
        torch.tensor(end_positions, dtype=torch.long)
    )

    torch.save(dataset, "./dataset.pt")

    print("Finish saving.")


if __name__ == "__main__":
    examples = get_examples("./data/tmp.json", True)
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    examples = build_feature(tokenizer, examples)
    save_examples(examples)
