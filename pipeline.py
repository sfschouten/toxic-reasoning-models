from types import MappingProxyType
from functools import partial
from pprint import pprint
import math

import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader, default_collate

import torch

from transformers import Pipeline, AutoTokenizer

from models.model_eurobert import EuroBertForToxicReasoning

from data import COLUMNS, comtok_create_thread_text, comtok_tokenize_func


DEFAULT_COMTOK_THRESHOLDS = MappingProxyType({
    'toxicity': 0.3,
    'justInappropriate': 0.5,
    'hasImplication': 0.3,
    'hasOther': 0.5,
    'implTemporality': 0.5,
    'implStereotype': 0.5,
})


def collate_fn(items):
    return default_collate([{col: torch.tensor(row[col]) for col in row} for row in items])


def convert_comtok_prediction(ids, preds, token_preds, thresholds):
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def rev_map(key):
        return COLUMNS[key].reverse_map

    def binary(key, idx):
        return rev_map(key)[0 if sigmoid(preds[key][idx]) < thresholds[key] else 1]

    def mc(key, idx):
        return rev_map(key)[preds[key][idx].argmax().squeeze().item()]

    def ml(key, idx):
        value = preds[key][idx]
        return [v for p, v in zip(value.tolist(), COLUMNS[key].values) if sigmoid(p) > thresholds[key]]

    def prob(key, idx):
        return sigmoid(preds[key][idx].item())

    result = []
    for i, (st_id, comment_ids) in enumerate(ids):
        nr_prev = sum(len(c_ids) for _, c_ids in ids[:i])
        comment_preds = []
        for j, (st_nr, comment_id) in enumerate(comment_ids):
            idx = nr_prev + j
            comment_preds.append({
                'comment_id': comment_id,
                (k := 'toxicity'): binary(k, idx),
                (k := 'justInappropriate'): binary(k, idx),
                (k := 'hasImplication'): binary(k, idx),
                (k := 'hasOther'): binary(k, idx),
                (k := 'implPolarity'): mc(k, idx),
                (k := 'implTopic'): mc(k, idx),
                (k := 'implTemporality'): ml(k, idx),
                (k := 'implStereotype'): binary(k, idx),
                (k := 'authorBelief'): prob(k, idx),
                (k := 'authorPrefer'): prob(k, idx),
                (k := 'authorAccount'): prob(k, idx),
                (k := 'typicalBelief'): prob(k, idx),
                (k := 'typicalPrefer'): prob(k, idx),
                (k := 'expertBelief'): prob(k, idx),
            })
        result.append({'st_id': st_id, 'preds': comment_preds})

    return result


class ToxicReasoningPipeline(Pipeline):

    REQUIRED_THREAD_FIELDS = ['st_id', 'workerid', 'subreddit', 'subm_title', 'subm_body']
    REQUIRED_COMMENT_FIELDS = ['comment_id', 'st_nr', 'comment_body', 'author_name']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.further_init(self.tokenizer.vocab['<COMMENT>'])

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {}

        for field in self.REQUIRED_COMMENT_FIELDS + self.REQUIRED_THREAD_FIELDS:
            if field in kwargs:
                preprocess_kwargs[field] = kwargs.pop(field)

        if 'comtok_thresholds' in kwargs:
            forward_kwargs['comtok_thresholds'] = kwargs.pop('comtok_thresholds')

        if 'mapping_functions' in kwargs:
            postprocess_kwargs['mapping_functions'] = kwargs.pop('mapping_functions')

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs, subreddit='UNKNOWN', subm_title=None, subm_body=None, author_name='UNKNOWN'):
        """
        Check if anything other than a pandas dataframe is passed, and if so, perform conversion.
        """
        if isinstance(inputs, pd.DataFrame):
            # assume this is a dataframe consisting of comments, with the schema of the toxic reasoning data
            return inputs

        if isinstance(inputs, str):
            # assume this is a single comment (only the text)
            inputs = [{
                'st_id': 'subthread1', 'workerid': '',
                'subreddit': subreddit, 'subm_title': subm_title, 'subm_body': subm_body,
                'comments': [{'comment_id': 'comment1', 'st_nr': 1, 'comment_body': inputs, 'author_name': author_name}]
            }]
        elif isinstance(inputs, dict) and 'comment' in inputs:
            # assume this is a single comment (with metadata)
            assert all(field in inputs for field in self.REQUIRED_COMMENT_FIELDS)
            inputs = [{
                'st_id': 'subthread1', 'workerid': '',
                'subreddit': subreddit, 'subm_title': subm_title, 'subm_body': subm_body,
                'comments': [inputs]
            }]
        elif isinstance(inputs, dict) and 'post_text' in inputs:
            # assume this is a thread of comments
            inputs = [inputs]

        assert isinstance(inputs, list)

        records = [
            comment_dict | {k: thread_dict[k] for k in self.REQUIRED_THREAD_FIELDS}
            for thread_dict in inputs for comment_dict in thread_dict['comments']
        ]
        df = pd.DataFrame.from_records(records)
        return df

    def _forward(self, model_inputs: pd.DataFrame, comtok_thresholds=DEFAULT_COMTOK_THRESHOLDS, max_length=500,
                 **kwargs):
        """
        :param model_inputs: a dataframe with the comments of a number of threads (equal to the batch size)
        :param kwargs:
        :return:
        """

        # TDOO infer toxic_reasoning_style from which model was passed
        toxic_reasoning_style = 'comment_token'

        if toxic_reasoning_style == 'comment_token':  # comtok for short
            # this style uses <COMMENT> tokens to classify all toxic reasoning fields for each comment

            # first do some pre-processing
            test_thread_df = comtok_create_thread_text(
                model_inputs, self.tokenizer, '<COMMENT>', max_length=max_length
            )
            # TODO the comment_token should be stored somewhere attached to the model

            ids: list[list[tuple[int, str]]] = test_thread_df['ids'].tolist()
            st_ids: list[str] = test_thread_df['st_id'].tolist()
            all_ids = list(zip(st_ids, ids))
            by_thread_df = test_thread_df.drop(columns=['ids', 'st_id'])
            dataset = Dataset.from_pandas(by_thread_df)
            tok_func = partial(comtok_tokenize_func,
                               tokenizer=self.tokenizer, max_length=max_length, include_labels=False)
            tokenized = dataset.map(tok_func, batched=True).remove_columns(['text'])
            dataloader = DataLoader(tokenized, batch_size=len(tokenized), collate_fn=collate_fn)
            batches = list(dataloader)
            assert len(batches) == 1

            # now forward model
            batch = {k: v.to(self.model.device) for k, v in batches[0].items() if isinstance(v, torch.Tensor)}
            preds, _ = self.model(**batch)
            token_preds = {k: v for k, v in preds.items() if k.endswith('Tokens')}
            other_preds = {k: v for k, v in preds.items() if not k.endswith('Tokens')}

            # finally convert prediction to unified format and return
            return convert_comtok_prediction(all_ids, other_preds, token_preds, comtok_thresholds)

        elif toxic_reasoning_style == 'decoder':
            raise NotImplementedError

    def postprocess(self, model_outputs, mapping_functions=MappingProxyType({})):
        mapping_outputs = {}
        for name, map_func in mapping_functions.items():
            map_results = []
            for st_model_out in model_outputs:
                mapped_comments = []
                for comment_pred in st_model_out['preds']:
                    mapped_comments.append(map_func(comment_pred))
                map_results.append({'st_id': st_model_out['st_id'], 'classes': mapped_comments})
            mapping_outputs[name] = map_results

        return model_outputs, mapping_outputs


if __name__ == "__main__":
    KEY = "EuroBERT/EuroBERT-610m"
    tokenizer = AutoTokenizer.from_pretrained(KEY)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<COMMENT>']})
    # model = EuroBertForToxicReasoning.from_pretrained('saved_eurobert_apr1_1401')
    model = EuroBertForToxicReasoning.from_pretrained('saved_eurobert_210m_apr9_1249')
    pipeline = ToxicReasoningPipeline(tokenizer=tokenizer, model=model)

    from mappings import latent_hatred

    TEST_CASES = [
        "white revolution is the only solution",
        "yes you are fine in white majority country. how is mexico doing these days? people come here because you couldn't build it",
        "black people only seem to thrive in white countries. how strange",
        "soon white people will not be a majority anywhere. you believe",
        "muslims are always wanting to kill someone! its their duty to kill the infidel.",
        "oh how i wish a man with a gun was there! wake up white people! they hate us",
    ]

    for test_case in TEST_CASES:
        print('\n\n' + test_case)
        model_results, mapping_results = pipeline(
            test_case, mapping_functions={'latent_hatred': latent_hatred}
        )

        pprint(model_results)
        pprint(mapping_results)
