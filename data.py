import os
import re
from dataclasses import dataclass
from typing import Union
from functools import reduce, partial
from collections import Counter

from torch.utils.data import DataLoader, default_collate

import pandas as pd
from tqdm import tqdm

from datasets import Dataset
from datasets import load_from_disk

import torch


pd.set_option('future.no_silent_downcasting', True)


def _raw_data(loc):
    if os.path.isfile(loc) and loc.endswith(".csv"):
        return pd.read_csv(loc)
    elif os.path.isfile(loc) and loc.endswith(".pkl"):
        return pd.read_pickle(loc)
    elif os.path.isdir(loc):
        return pd.concat([pd.read_csv(os.path.join(loc, f)) for f in os.listdir(loc) if f.endswith('.csv')])
    else:
        raise ValueError('Data file/directory not found.')


@dataclass
class ColumnSet:
    columns: tuple = None
    condition: int = None
    children: tuple = ()

    def descendent_values(self):
        yield from self.columns
        for child in self.children:
            yield from child.descendent_values()


@dataclass
class Column:
    type: str = None                            # multi-class or multi-label
    map: dict = None                            # mapping from string values to indices (multi-class)
    _values: list = None                        # all possible values that the variable can take (multi-label)
    empty_value: Union[int, float] = -100       # the value to substitute if no valid string value is found
    apply_fn: callable = lambda x: x            # a function to apply before applying the map

    @property
    def values(self):
        if self._values is None:
            return list(self.map.keys())
        else:
            return self._values

    @property
    def reverse_map(self):
        if self.map is not None:
            return {val: key for key, val in self.map.items()}
        else:
            return {i: v for i, v in enumerate(self._values)}


GROUP_MAP = {
    'the author themselves and/or their ingroup': 0,
    'another participant in the conversation and/or the group they belong to': 1,
    'an individual outside of the conversation': 2,
    'another group': 3,
    'none of the above': 4,
}
ORDINALS = {'Very low': 0.1, 'Low': 0.3, 'Medium': 0.5, 'High': 0.7, 'Very high': 0.9}
COLUMNS = {
    'toxicity': Column(type='mc', map={'Yes/Maybe': 1, 'No': 0}),
    'counternarrative': Column(type='mc', map={'Yes': 1, 'No': 0}),
    'justInappropriate': Column(type='mc', map={'Yes': 1, 'No': 0}),
    'hasImplication': Column(type='mc', map={"['_Different kind of toxicity']": 0, "[]": 1}, apply_fn=str),
    'subject': Column(type='mc', map=GROUP_MAP),
    'subjectGroupType': Column(type='ml', _values=[
        '_Sexual orientation', '_Gender', '_Disability', '_Race/Ethnicity', '_Age', '_Religion', '_Famous individual',
        '_Political affiliation', '_Social belief', '_Body image', '_Addiction', '_Socioeconomic status',
        '_Profession', '_Nationality', '_other']),
    'subjectTokens': Column(type='ml-tokens'),
    'hasOther': Column(type='mc', map={"['_No other']": 0, "[]": 1}, apply_fn=str),
    'other': Column(type='mc', map=GROUP_MAP),
    'otherTokens': Column(type='ml-tokens'),
    'implTopic': Column(type='mc', map={'(a)': 0, '(a.1)': 1, '(b)': 2, '(b.1)': 3, '(c)': 4, '(d)': 5, '(e)': 6}),
    'implTopicTokens': Column(type='ml-tokens'),
    'implPolarity': Column(type='mc', map={'Negative': 0, 'Neutral': 1, 'Positive': 2}),
    'implStereotype': Column(type='mc', map={'Yes': 1, 'No': 0}),
    'implSarcasm': Column(type='mc', map={'Yes': 1, 'No': 0}),
    'implTemporality': Column(type='ml', _values=['_Past', '_Present', '_Future']),
    'authorBelief': Column(type='mc', map=ORDINALS, empty_value=0.5),
    'authorPrefer': Column(type='mc', map=ORDINALS, empty_value=0.5),
    'authorAccount': Column(type='mc', map=ORDINALS, empty_value=0.5),
    'typicalBelief': Column(type='mc', map=ORDINALS, empty_value=0.5),
    'typicalPrefer': Column(type='mc', map=ORDINALS, empty_value=0.5),
    'expertBelief': Column(type='mc', map=ORDINALS, empty_value=0.5)
}


ANSWER_HIERARCHY = \
    ColumnSet(columns=('toxicity',), condition=1, children=(
        ColumnSet(columns=('justInappropriate',), condition=0, children=(
            ColumnSet(columns=('hasImplication',), condition=1, children=(
                ColumnSet(columns=('subject', 'subjectGroupType', 'subjectTokens')),  # 'subjectGroup',
                ColumnSet(columns=('hasOther',), condition=1, children=(
                    ColumnSet(columns=('other', 'otherTokens')),  # 'otherGroup',
                )),
                ColumnSet(columns=('implTopic', 'implTopicTokens', 'implPolarity', 'implTemporality', 'implStereotype')),
                ColumnSet(columns=('authorBelief', 'authorPrefer', 'authorAccount',
                                   'typicalBelief', 'typicalPrefer', 'expertBelief')),
            )),
        )),
    ))


def _preprocess(df):
    # add preprocessed version of annotation columns
    for col in df.columns:
        if 'answer' not in col:
            continue

        def process(value):
            if type(value) == dict:
                return [key for key, val in value.items() if val]
            else:
                return value

        df[col.replace('answer', 'answer_pp')] = df[col].apply(process)

    df['answer_pp_implTopic'] = df['answer_pp_implTopic'].str.extract(r'^(?:\.\.\.\s)?(\(..?.?\))')

    def ml_func(values, members):
        return [int(v in members) for v in values] if members is not None else [-100 for _ in values]

    for col in COLUMNS.keys():
        if COLUMNS[col].type == 'mc':
            df['label_' + col] = df[f'answer_pp_{col}'].apply(COLUMNS[col].apply_fn).replace(COLUMNS[col].map)
            df.loc[~df['label_' + col].isin(COLUMNS[col].map.values()), 'label_' + col] = COLUMNS[col].empty_value
        elif COLUMNS[col].type == 'ml':
            df['label_' + col] = df[f'answer_pp_{col}'].apply(partial(ml_func, COLUMNS[col].values))
        elif COLUMNS[col].type == 'ml-tokens':
            df['label_' + col] = df[[f'answer_pp_{col}', 'comment_body_tokens']].apply(
                lambda row: ml_func(row.iloc[1].split(), row.iloc[0]), axis=1
            )

    # zero out columns where appropriate
    stack = [ANSWER_HIERARCHY]
    while stack:
        node = stack.pop()
        for child in node.children:
            stack.append(child)

            empty = reduce(lambda a, b: a | b, [df.loc[:, 'label_' + col] != node.condition for col in node.columns])
            for col in child.descendent_values():
                if COLUMNS[col].type == 'mc':
                    df.loc[empty, 'label_' + col] = -100
                elif COLUMNS[col].type == 'ml':
                    df.loc[empty, 'label_' + col] = pd.Series([[-100] * len(COLUMNS[col].values)]).repeat(empty.sum()).values
                elif COLUMNS[col].type == 'ml-tokens':
                    df.loc[empty, 'label_' + col] = df.loc[empty, 'comment_body_tokens'].apply(lambda x: [-100] * len(x.split()))

    return df


def _create_thread_text(comments_df, tokenizer, comment_token, add_post_text=False, max_length=500):
    nr_long_msgs = 0
    nr_skip_msgs = 0

    result = []
    for (workerid, st_id), st_df in tqdm(comments_df.groupby(by=['workerid', 'st_id'])):
        row = {'st_id': st_id}

        title = st_df['subm_title'].unique().tolist()[0]
        post = st_df['subm_body'].unique().tolist()[0]
        subreddit = st_df['subreddit'].unique().tolist()[0]

        post_title = '`' + title + '`' if not pd.isna(title) else 'EMPTY'
        if add_post_text:
            post_text = '\n```\n' + post + '\n```' if not pd.isna(post) else 'EMPTY'
        else:
            post_text = 'HIDDEN'

        start_str = f"""\
From a thread in r/{subreddit}

Post Title: {post_title}
Post Text: {post_text}

"""
        st_df = st_df.sort_values('st_nr')

        message_counts = [0]
        message_strs = [""]
        cols = ['author_name', 'comment_body']
        for i, (author, comment_body) in enumerate(st_df[cols].itertuples(index=False)):
            msg = comment_token + f"Message {i + 1} (by {author}):\n```\n{comment_body}\n```\n\n"

            if len(tokenizer.encode(start_str + message_strs[-1] + msg)) > max_length:
                # adding the current comment would make the existing message too long, add a new empty message
                message_strs.append("")
                message_counts.append(0)

            message_strs[-1] += msg
            message_counts[-1] += 1

        if len(message_strs) > 1:
            nr_long_msgs += 1

        def count_words(txt, count_zero_length):
            return len(list(filter(
                lambda x: x[0] % 2 == 0 and (count_zero_length or len(x[1]) > 0),
                enumerate(re.split(r'(\s+)', txt))
            )))

        # start_nr_words = sum(1 for i, w in enumerate(start_words) if i % 2 == 0)
        start_nr_words = count_words(start_str, False)

        new_rows = []
        for i, msg_str in enumerate(message_strs):
            if message_counts[i] == 0:
                continue

            new_row = row.copy()
            new_row['text'] = start_str + msg_str

            if len(tokenizer.encode(new_row['text'])) > max_length:
                nr_skip_msgs += 1
                continue

            start = sum(message_counts[:i])
            end = start + message_counts[i]

            for key, col in COLUMNS.items():
                by_comment = st_df.iloc[start:end]['label_' + key].tolist()
                if col.type in ['mc', 'ml']:
                    new_row['label_' + key] = by_comment
                else:
                    for i, c in enumerate(by_comment):
                        assert len(c) == len(st_df.iloc[start+i]['comment_body_tokens'].split())
                        assert len(c) == count_words(st_df.iloc[start + i]['comment_body'], True)
                    # token-level annotations should be merged to thread-level
                    labels = ([-100] * start_nr_words) + sum((([-100] * 5) + c + [-100] for c in by_comment), start=[])
                    # assert len(labels) == count_words(new_row['text'], False)
                    new_row['label_' + key] = labels
            new_rows.append(new_row)
        result.extend(new_rows)

    print(f'Split up {nr_long_msgs} that were too long otherwise.')
    print(f'Skipped {nr_skip_msgs} that were still too long after.')
    return pd.DataFrame(result)


def _collate(batch):
    DEFAUlT_COLS = ['input_ids', 'attention_mask']
    TOKEN_COLS = ['label_subjectTokens', 'label_otherTokens', 'label_implTopicTokens']
    collated = default_collate([{col: torch.tensor(row[col]) for col in DEFAUlT_COLS + TOKEN_COLS} for row in batch])

    COMMENT_COLS = ['label_' + key for key in [
        'toxicity', 'counternarrative', 'justInappropriate', 'hasImplication', 'subject', 'subjectGroupType',
        'hasOther', 'other', 'implTopic', 'implPolarity', 'implStereotype', 'implSarcasm', 'implTemporality',
        'authorBelief', 'authorPrefer', 'authorAccount', 'typicalBelief', 'typicalPrefer', 'expertBelief'
    ]]
    for col in COMMENT_COLS:
        collated[col] = torch.tensor(sum((row[col] for row in batch), start=[]))

    return collated


def load_data(train_data_loc, test_data_loc, cached_data_loc, tokenizer, comment_token, batch_size, sample_cap=-1,
              use_cache=True, write_cache=True, dump_intermediates=False, max_length=500):
    data_dirs = {'train': train_data_loc, 'test': test_data_loc}
    cache_dirs = {key: os.path.join(cached_data_loc, key) for key in data_dirs.keys()}

    if use_cache and all(os.path.exists(cdir) for cdir in cache_dirs.values()):
        tokenized_datasets = {key: load_from_disk(cdir) for key, cdir in cache_dirs.items()}
    else:
        by_comment_data = {key: _raw_data(_dir) for key, _dir in data_dirs.items()}

        if sample_cap > 0:
            by_comment_data = {key: df.sort_values('st_id').iloc[-sample_cap:] for key, df in by_comment_data.items()}

        by_thread_df = {
            key: _create_thread_text(_preprocess(df), tokenizer, comment_token, max_length=max_length)
            for key, df in by_comment_data.items()
        }

        datasets = {key: Dataset.from_pandas(df) for key, df in by_thread_df.items()}

        def tokenize_func(examples):
            tokenized = tokenizer(examples["text"], padding="max_length", max_length=max_length)

            updated_labels = {}
            token_level_cols = {key: examples['label_' + key] for key, col in COLUMNS.items() if col.type == 'ml-tokens'}
            for key, col in token_level_cols.items():
                updated = []
                for enc, text, word_label in zip(tokenized.encodings, examples['text'], col):
                    words = re.split(r'(\s+)', text)
                    char2word = sum(([i // 2 if i % 2 == 0 else -1] * len(w) for i, w in enumerate(words)), start=[])
                    char_label = [word_label[j] if (j := char2word[i]) != -1 else -1 for i, _ in enumerate(text)]
                    char_label[-1] = char_label[-1] if char_label[-1] != -1 else 0
                    char_label = [l if l > -1 else (char_label[i-1] == char_label[i+1] == 1) for i, l in enumerate(char_label)]

                    token_label = []
                    for token_idx, id in enumerate(enc.ids):                # loop over tokens
                        char_idxs = enc.token_to_chars(token_idx)
                        if char_idxs is None:
                            token_label.append(-100)
                            continue
                        char_lbl = [char_label[ci] for ci in range(*char_idxs)]
                        label = int(Counter(char_lbl).most_common(1)[0][0])
                        token_label.append(label)
                    updated.append(token_label)
                updated_labels['label_' + key] = updated

            return {**tokenized, **updated_labels}

        tokenized_datasets = {key: dataset.map(tokenize_func, batched=True) for key, dataset in datasets.items()}

        if dump_intermediates:
            lines = []
            for (i, row1), row2 in zip(by_thread_df['test'].iterrows(), tokenized_datasets['test']):
                words = [w for i, w in enumerate(re.split(r'(\s+)', row1['text'])) if i % 2 == 0]
                final1 = ""
                for l, w in zip(row1['label_subjectTokens'], words):
                    final1 += " "
                    for c in w:
                        final1 += c if l != 1 else c + '\u035F'
                lines.append(final1 + '\n')

                tokens = [tokenizer._convert_id_to_token(id) for id in row2['input_ids']]
                final2 = ""
                for l, t in zip(row2['label_subjectTokens'], tokens):
                    t = ' ' + t[1:] if t[0] == "▁" else t
                    for c in t:
                        final2 += c if l != 1 else c + '\u035F'
                lines.append(final2 + '\n')
                lines.append("\n")
                
            with open('out.txt', 'w') as f:
                f.writelines(lines)

        if write_cache:
            for key, dataset in tokenized_datasets.items():
                dataset.save_to_disk(os.path.join(cached_data_loc, key))

    _train = tokenized_datasets['train'].shuffle(seed=0)
    tokenized_datasets['dev'] = _train.select(range(500))
    tokenized_datasets['train'] = _train.select(range(500, len(_train)))

    dev_dataloader = DataLoader(tokenized_datasets['dev'], batch_size=batch_size, collate_fn=_collate)
    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=batch_size, collate_fn=_collate)
    eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=batch_size, collate_fn=_collate)
    return train_dataloader, dev_dataloader, eval_dataloader


if __name__ == '__main__':
    DATA_DIR = {'test': '../data/temporal/preprocessed_test.pkl', 'train': '../data/temporal/preprocessed_train.pkl'}
    COMMENT_TOKEN = "<COMMENT>"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    tokenizer.add_special_tokens({'additional_special_tokens': [COMMENT_TOKEN]})

    load_data(
        DATA_DIR['train'], DATA_DIR['test'], 'cache/',
        tokenizer, COMMENT_TOKEN,
        8, sample_cap=25, use_cache=False, write_cache=False, dump_intermediates=True
    )