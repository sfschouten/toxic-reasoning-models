import warnings

import pandas as pd

from sklearn.metrics import classification_report

from data import COLUMNS


def get_random_baseline(class_counts, baseline_type):
    total = sum(class_counts.values())

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


def _by_annotator(test_df, get_prediction):
    result_records = []

    for lang, lang_df in test_df.groupby('lang'):
        for annotator, ann_df in lang_df.groupby('workerid'):
            for col in COLUMNS:
                if COLUMNS[col].type == 'ml-tokens':
                    continue

                predictions = ann_df.apply(
                    lambda row: get_prediction(row['st_id'], row['st_nr'], row['comment_id'], col), axis=1
                )
                kwargs = {}

                if COLUMNS[col].type == 'mc':
                    references = ann_df[f'answer_pp_{col}']
                    idx = ~references.isna()
                    kwargs['y_true'] = references[idx].astype(str)
                    kwargs['y_pred'] = predictions[idx].astype(str)
                elif COLUMNS[col].type == 'ml':
                    references = ann_df[f'label_{col}']
                    idx = references.apply(lambda x: any(y != -100 for y in x))
                    kwargs['y_true'] = references[idx].tolist()

                    _len = len(references.iat[0])
                    predictions = predictions.apply(lambda x: [-1] * _len if x == 'NA' else x)
                    kwargs['y_pred'] = predictions[idx].apply((lambda x: [float(y) > 0 for y in x])).tolist()
                    kwargs['target_names'] = COLUMNS[col].values

                if len(kwargs['y_pred']) == 0:
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    report = classification_report(output_dict=True, **kwargs)

                result_records.extend([{
                    'lang': lang, 'annotator': annotator, 'column': col, 'label': label
                } | ({'accuracy': value} if label == 'accuracy' else {
                    'precision': value['precision'],
                    'recall': value['recall'],
                    'f1': value['f1-score'],
                    'support': value['support'],
                }) for label, value in report.items()])

                result_records.extend([
                    {'lang': lang, 'annotator': f'{annotator}_baseline', 'column': col} | d
                    for d in get_random_baseline(
                        class_counts={label: value['support'] for label, value in report.items()
                                      if label not in ['accuracy', 'macro avg', 'weighted avg']},
                        baseline_type='match',
                    )
                ])

    results_df = pd.DataFrame.from_records(result_records)
    return results_df
