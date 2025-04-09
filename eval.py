import warnings
from functools import partial

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report

from models.data import COLUMNS


def get_random_baseline(class_counts, baseline_type):
    total = sum(class_counts.values())
    if total == 0:
        return []

    # the probability of each class in the label distribution
    p = {cls: count / total for cls, count in class_counts.items()}

    if baseline_type == 'match':        # match the label distribution
        r = p
    elif baseline_type == 'uniform':    # uniform probability
        r = {cls: 1 / len(class_counts) for cls in class_counts}
    elif baseline_type == 'majority':   # always predict majority label
        max_cls = max(class_counts, key=class_counts.get)
        r = {cls: 1 if cls == max_cls else 0 for cls in class_counts}
    else:
        raise ValueError()

    f1 = {cls: 2 * p[cls] * r[cls] / (p[cls] + r[cls]) if p[cls] + r[cls] > 0 else 0 for cls in class_counts}
    return [
        {
            'label': cls,
            'precision': p[cls],
            'recall': r[cls],
            'f1': f1[cls],
            'support': class_counts[cls],
        } for cls in class_counts
    ] + [
        {
            'label': 'weighted avg',
            'precision': sum(p[cls] * p[cls] for cls in class_counts),
            'recall': sum(p[cls] * r[cls] for cls in class_counts),
            'f1': sum(p[cls] * f1[cls] for cls in class_counts),
            'support': sum(class_counts.values())
        }, {
            'label': 'macro avg',
            'precision': sum(p.values()) / len(class_counts),
            'recall': sum(r.values()) / len(class_counts),
            'f1': sum(f1.values()) / len(class_counts),
            'support': sum(class_counts.values())
        }
    ]


def _cls_report(keyvalues, df, predictions, col):
    kwargs = {}

    if COLUMNS[col].type == 'mc':
        references = df[f'answer_pp_{col}']
        idx = ~references.isna()
        kwargs['y_true'] = references[idx].astype(str)
        kwargs['y_pred'] = predictions[idx].astype(str)
        kwargs['labels'] = COLUMNS[col].values
    elif COLUMNS[col].type == 'ml':
        references = df[f'label_{col}']
        idx = references.apply(lambda x: any(y != -100 for y in x))
        kwargs['y_true'] = references[idx].tolist()

        _len = len(references.iat[0])
        predictions = predictions.apply(lambda x: [-1] * _len if x == 'NA' else x)
        kwargs['y_pred'] = predictions[idx].apply((lambda x: [float(y) > 0 for y in x])).tolist()
        kwargs['target_names'] = COLUMNS[col].values
    else:
        raise NotImplementedError

    if len(kwargs['y_pred']) == 0:
        return []

    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = classification_report(output_dict=True, **kwargs)

    results.extend([{**keyvalues, 'label': label} | ({'accuracy': value} if label == 'accuracy' else {
        'precision': value['precision'],
        'recall': value['recall'],
        'f1': value['f1-score'],
        'support': value['support'],
    }) for label, value in report.items()])

    results.extend([
        {**keyvalues, 'annotator': keyvalues['annotator'] + '_baseline'} | d
        for d in get_random_baseline(
            class_counts={label: value['support'] for label, value in report.items()
                          if label not in ['accuracy', 'micro avg', 'macro avg', 'weighted avg', 'samples avg']},
            baseline_type='match',
        )
    ])

    return results


JACCARD_NONE = 'no_reference'
JACCARD_EMPTY_REF = 'empty_reference'
JACCARD_EMPTY_PRED = 'empty_prediction'
JACCARD_EMPTY_BOTH = 'empty_both'


def jaccard_index(pred, answer, label):
    """ This should not return NA because of a missing prediction, but may return NA for missing answers."""
    if type(pred) != dict:
        pred = {}

    if not hasattr(answer, '__len__') and pd.isna(answer):
        return pd.NA, JACCARD_NONE
    elif answer == ['_None']:
        answer = []
    elif '_None' in answer:
        answer.remove('_None')

    if len(answer) == 0 and len(pred) == 0:
        return pd.NA, JACCARD_EMPTY_BOTH
    elif len(answer) == 0:
        return pd.NA, JACCARD_EMPTY_REF
    elif len(pred) == 0:
        return 0, JACCARD_EMPTY_PRED

    pred_size = sum(1 if v else 0 for v in pred.values())
    ans_size = len(answer)
    intersection = sum(1 if v in pred and pred[v] else 0 for v in answer)

    # if (pred_size + ans_size - intersection) == 0:
    #     assert pred_size == ans_size == 0
    #     return 'Prediction and answer are both empty sets.'

    return intersection / (pred_size + ans_size - intersection), pd.NA


def jaccard_sampled(test_df, get_prediction, columns=('subjectTokens', 'otherTokens', 'implTopicTokens')):
    NR_SAMPLES = 50

    print(test_df['answer_pp_implDetected'].value_counts())
    test_df = test_df[test_df['answer_pp_implDetected']]
    result_records = []
    for lang, lang_df in test_df.groupby('lang'):
        for col in columns:
            for sample_i in tqdm(range(NR_SAMPLES)):
                _df = lang_df[lang_df['label_hasOther'] == 1] if col == 'otherTokens' else lang_df
                sample_df = _df.groupby('comment_id').sample(n=1)
                jaccard_scores = sample_df.apply(
                    lambda row: jaccard_index(
                        get_prediction(row['st_id'], row['st_nr'], row['comment_id'], col),
                        row[f'answer_pp_{col}'], row[f'label_{col}']
                    ), axis=1, result_type='expand'
                )

                result = {'lang': lang, 'annotator': '*', 'column': col, 'sample': sample_i,
                          'jaccard': jaccard_scores[0].mean(), 'support': (~jaccard_scores[0].isna()).sum()}
                result_records.append(result)

    return pd.DataFrame.from_records(result_records)


def f1_by_annotator(test_df, get_prediction, columns=('toxicity', 'counternarrative', 'justInappropriate', 'hasImplication')):
    result_records = []

    for lang, lang_df in test_df.groupby('lang'):
        for annotator, ann_df in lang_df.groupby('workerid'):
            for col in COLUMNS:
                if col not in columns:
                    continue

                predictions = ann_df.apply(
                    lambda row: get_prediction(row['st_id'], row['st_nr'], row['comment_id'], col), axis=1
                )
                result = _cls_report(
                    {'lang': lang, 'annotator': annotator, 'column': col},
                    ann_df, predictions, col
                )
                if result is None:
                    continue

                result_records.extend(result)

    results_df = pd.DataFrame.from_records(result_records)
    return results_df


def f1_majority(test_df, get_prediction,
    columns=('toxicity', 'counternarrative', 'justInappropriate', 'hasImplication', 'hasOther')
):
    """ Only for multi-class """
    COLS = ['st_id', 'st_nr', 'comment_id'] + [f'answer_pp_{c}' for c in columns]

    def majority_vote(series, default_value=None):
        modes = series.mode()
        if len(modes) == 1:
            return modes.iloc[0]
        elif len(modes) == 0:
            return None
        else:
            return default_value

    result_records = []
    for lang, lang_df in test_df.groupby('lang'):
        maj_df = lang_df[COLS].groupby('comment_id').agg({
            'st_id': 'first',
            'st_nr': 'first',
            'answer_pp_toxicity': partial(majority_vote, default_value='Yes/Maybe'),
            'answer_pp_counternarrative': partial(majority_vote, default_value='No'),
            'answer_pp_justInappropriate': partial(majority_vote, default_value='No'),
            'answer_pp_hasImplication': partial(majority_vote, default_value='[]'),
            'answer_pp_hasOther': partial(majority_vote, default_value="['_No other']"),
        }).reset_index()
        for col in columns:
            predictions = maj_df.apply(
                lambda row: get_prediction(row['st_id'], row['st_nr'], row['comment_id'], col), axis=1
            )
            result = _cls_report({'lang': lang, 'annotator': '*', 'column': col}, maj_df, predictions, col)
            if result is None:
                continue

            result_records.extend(result)

    results_df = pd.DataFrame.from_records(result_records)
    return results_df


def f1_conditional_selection(test_df, get_prediction, condition_score, interval=0.1, aggregation='none', nr_samples=50,
                             columns=(
    ('subject', 'subjectTokens'),
    ('subjectGroupType', 'subjectTokens'),
    ('other', 'otherTokens'),
    ('implTopic', 'implTopicTokens'),
    ('implPolarity', 'implTopicTokens'),
    ('implTemporality', 'implTopicTokens'),
)):
    result_records = []
    for eval_col, cond_col in columns:
        print(eval_col, cond_col)
        cond_scores = test_df.apply(
            lambda row: condition_score(
                get_prediction(row['st_id'], row['st_nr'], row['comment_id'], cond_col),
                row[f'answer_pp_{cond_col}'], row[f'label_{cond_col}']
            ), axis=1, result_type='expand'
        )
        # valid_scores = pd.to_numeric(cond_scores, errors='coerce').notna()
        # print('invalid', cond_scores[~valid_scores].value_counts())

        cond_score_masks = [(cond_scores[1] == err, err, 11) for err in cond_scores[1].unique()]

        # cond_scores[~valid_scores] = pd.NA
        print('cond_scores', cond_scores.shape, cond_scores[0].notna().sum())
        test_df['cond_score'] = cond_scores[0]

        # print([i * interval for i in range(int(1 / interval)+1)])
        # boundaries = cond_scores.quantile([i * interval for i in range(int(1 / interval)+1)])
        # print(boundaries)
        # boundaries = [i * interval for i in range(int(1 / interval)+1)]
        cond_score_masks += [
            ((cond_scores[0] >= 0) & (cond_scores[1] != JACCARD_EMPTY_PRED), '>=0.0-non-empty-pred', 0.33)
        ]
        cond_score_masks += [((cond_scores[0] > 0), '> 0.0', 0.66)]
        cond_score_masks += [
            (cond_scores[0] >= (v := i * interval), f'>={v:.1f}', i) for i in range(int(1 / interval)+1)
        ]

        eval_predictions = test_df.apply(
            lambda row: get_prediction(row['st_id'], row['st_nr'], row['comment_id'], eval_col), axis=1
        )
        print('eval_predictions', eval_predictions.shape, eval_predictions.notna().sum())
        for mask, score_condition, condition_order in cond_score_masks:
            rel_preds = eval_predictions[mask]
            rel_rows = test_df[mask]
            # rel_rows['cond_score'] = cond_scores[[cond_scores >= boundary]]
            if len(rel_preds) == 0 or len(rel_rows) == 0:
                continue

            todo = []
            if aggregation == 'none':
                todo = [(rel_rows, {})]
                # print('nr-relevant', mask.sum())
            elif aggregation == 'random':
                grouped = rel_rows.groupby('comment_id')
                for i in range(nr_samples):
                    sample = grouped.sample(n=1)
                    todo.append((sample, {'sample': i}))
            elif aggregation == 'max-score':
                idx = rel_rows.groupby('comment_id')['cond_score'].transform('max') == rel_rows['cond_score']
                selection = rel_rows.loc[idx, :]
                if len(selection) == 0:
                    print(f"Warning: Skipped aggregation=max for '{score_condition}'")
                    continue
                todo = [(selection, {})]

            for (rows, kvs) in todo:
                result = _cls_report(
                    {'lang': '*', 'annotator': '*', 'column': eval_col, 'cond_col': cond_col,
                     'score_condition': score_condition, 'condition_order': condition_order} | kvs,
                    rows, rel_preds[rows.index], eval_col
                )
                result_records.extend(result)

        print('========================================================================')
        print('========================================================================')

    results_df = pd.DataFrame.from_records(result_records)
    return results_df
