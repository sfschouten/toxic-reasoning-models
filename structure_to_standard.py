import copy
import re

from rapidfuzz import process, fuzz

from structure import ImplicationCategory, GroupCharacteristic


def with_false(keys):
    return {key: False for key in keys}


def with_underscores(dct):
    return {'_' + key: val for key, val in dct.items()}


def from_enum(enum):
    return with_false([key.value for key in enum])


def impl_cat():
    return with_false([
        opt + ' ' + key.value for opt, key in
        zip(['(a)', '(a.1)', '(b)', '(b.1)', '(c)', '(d)', '(e)'], ImplicationCategory)
    ])


IMPL_CAT_MAP = {enum: json for json, enum in zip(impl_cat(), ImplicationCategory)}


ANSWER_DEFAULT = {
    'trinary':            with_false(['_Yes/Maybe', '_No', '_Counter-speech']),
    'justInappropriate':  {},
    'hasImplication':     with_false(['_Different kind of toxicity']),
    'sameImplication':    with_false(['_Same as a previous comment']),
    'englishImplication': None,
    'subject':            {},
    'subjectGroup':       None,
    'subjectGroupType':   with_underscores(from_enum(GroupCharacteristic)),
    'subjectTokens':      {},
    'hasOther':           {'_No other': False},
    'other':              {},
    'otherGroup':         None,
    'otherTokens':        {},
    'implTopic':          {},
    'implTopicTokens':    {},
    'implPolarity':       {},
    'implTemporality':    with_false(['_Past', '_Present', '_Future']),
    'implStereotype':     {},
    'implSarcasm':        {},
    'authorBelief':       {},
    'authorPrefer':       {},
    'authorAccount':      {},
    'typicalBelief':      {},
    'typicalPrefer':      {},
    'expertBelief':       {},
}


def number_to_ordinal(num_str):
    num = float(num_str)
    if num < 0.15:
        return 'Very low'
    elif num < 0.35:
        return 'Low'
    elif num < 0.65:
        return 'Medium'
    elif num < 0.85:
        return 'High'
    elif num < 1:
        return 'Very high'
    else:
        return 'Not applicable'


def find_substring(full_string, substring):
    """ substring comes from the LLM, so may not be an exact subtring """
    index = full_string.find(substring)

    assert len(substring) <= len(full_string), "substring longer than full string"

    if index == -1:
        candidates = [full_string[i:i+len(substring)] for i in range(0, len(full_string)-len(substring))]
        extracted = process.extractOne(substring, candidates, scorer=fuzz.WRatio)
        if extracted is None:
            print(full_string, substring)
            return -1
        _match, score, index = extracted

    return index


def find_tokens(all_tokens_str, all_whitespace, substring):
    all_tokens = all_tokens_str.split()
    all_tokens_r = [t.split('_', 1)[1] for t in all_tokens]
    if isinstance(all_whitespace, str):  # assume that strings contain list
        all_whitespace = eval(all_whitespace)

    if len(substring) == 0:
        # nothing to find
        return {t: False for t in all_tokens}

    # ... no idea why this was necessary in the notebook where I called ChatGPT but not now... (;_;)
    # assert len(all_tokens) == len(all_whitespace) + 1
    # all_whitespace.append('')

    full_string = ''.join(t+w for t, w in zip(all_tokens_r, all_whitespace))
    idx_string = sum(([i]*(len(toks)+len(ws)) for i,(toks,ws) in enumerate(zip(all_tokens_r, all_whitespace))), [])
    assert len(full_string) == len(idx_string)

    if len(substring) > len(full_string):
        return {t: False for t in all_tokens}

    index = find_substring(full_string, substring)
    start_word_index = idx_string[index]
    end_word_index = idx_string[index + len(substring) - 1]

    result = {t: start_word_index <= i <= end_word_index for i, t in enumerate(all_tokens)}
    # print(full_string)
    # print(substring)
    # print(result)
    # print(start_word_index, end_word_index)
    # print()
    return result


def tokenize(text):
    """ For whatever reason, whitespace was not preserved somewhere down the line; re-tokenizing here. """
    if not isinstance(text, str):
        return None, None
    all_tokens = re.split(r'(\s+)', text)
    tokens = [t for i, t in enumerate(all_tokens) if i % 2 == 0]
    whitespace = [t for i, t in enumerate(all_tokens) if i % 2 == 1]
    tokens = [f"{i}_{token}" for i, token in enumerate(tokens)]
    return ' '.join(tokens), whitespace


def convert(structured_output, st_df):
    result = structured_output
    out_by_message = {annotation['message_nr']: annotation for annotation in result['comment_annotations']}

    answers = {}
    for i, (r_idx, row) in enumerate(st_df.iterrows()):
        answer = copy.deepcopy(ANSWER_DEFAULT)
        if i+1 in out_by_message:
            out = out_by_message[i+1]

            toks, ws = tokenize(row.comment_body)
            # toks, ws = row.comment_body_tokens, row.comment_body_tokens_ws

            answer['trinary']['_'+out['is_toxic']] = True
            answer['justInappropriate'] = 'Yes' if out['is_only_innapropriate'] else 'No'
            if out['is_counter_speech'] == 'true':
                answer['trinary']['_Counter-speech'] = True

            tr = out['toxic_reasoning']
            if tr is None:
                answer['hasImplication']['_Different kind of toxicity'] = True
            else:
                answer['implication'] = tr['implication']

                answer['subject'] = tr['subject_role']
                answer['subjectGroup'] = tr['subject_descr']
                answer['subjectGroupType']['_'+tr['subject_characteristic']] = True
                answer['subjectTokens'] = find_tokens(toks, ws, tr['subject_span'])

                answer['hasOther']['_No other'] = not tr['has_other']
                answer['other'] = tr['other_role']
                answer['otherGroup'] = tr['other_descr']
                answer['otherTokens'] = find_tokens(toks, ws, tr['other_span'])

                answer['implTopic'] = IMPL_CAT_MAP[tr['category']]
                answer['implTopicTokens'] = find_tokens(toks, ws, tr['impl_span'])
                answer['implPolarity'] = tr['polarity']
                answer['implStereotype'] = 'Yes' if tr['stereotype'] else 'No'
                answer['implSarcasm'] = 'Yes' if tr['sarcasm'] else 'No'
                for when in tr['when']:
                    answer['implTemporality']['_'+when] = True

                answer['authorBelief'] = number_to_ordinal(tr['author_belief'])
                answer['authorPrefer'] = number_to_ordinal(tr['author_preference'])
                answer['authorAccount'] = number_to_ordinal(tr['author_responsibility'])
                answer['typicalBelief'] = number_to_ordinal(tr['typical_belief'])
                answer['typicalPrefer'] = number_to_ordinal(tr['typical_preference'])
                answer['expertBelief'] = number_to_ordinal(tr['expert_belief'])
        else:
            print(f'{i+1} not there')

        answers[row.st_nr] = answer

    return answers
