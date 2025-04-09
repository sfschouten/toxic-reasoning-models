from types import MappingProxyType
from collections import defaultdict


NOT_TOXIC = 'not_toxic'
JUST_INAPPROPRIATE = 'just_inappropriate'
DIFFERENT_KIND = 'toxic_without_implication'
OTHER = 'other_implication'


LATENT_HATRED_THRESHOLD_DEFAULTS = MappingProxyType(defaultdict(lambda: 0.5))


def latent_hatred(prediction, attitude_thresholds=LATENT_HATRED_THRESHOLD_DEFAULTS):

    if prediction['toxicity'] == 'No':
        return [NOT_TOXIC]
    if prediction['justInappropriate'] == 'Yes':
        return [JUST_INAPPROPRIATE]
    if prediction['hasImplication'] == "['_Different kind of toxicity']":
        return [DIFFERENT_KIND]

    classes = []
    if prediction['authorBelief'] > attitude_thresholds[('misinformation', 'authorBelief')]     \
    and prediction['expertBelief'] < attitude_thresholds[('misinformation', 'expertBelief')]:
        classes.append('misinformation')

    if prediction['implTopic'] == '(a.1)' \
    and 'future' in prediction['implTemporality'] \
    and prediction['authorPrefer'] > attitude_thresholds[('incitement', 'authorPrefer')]:
        classes.append('incitement')

    if prediction['implTopic'] == '(a.1)' \
    and 'future' in prediction['implTemporality'] \
    and prediction['authorPrefer'] > attitude_thresholds[('threat', 'authorPrefer')] \
    and prediction['authorAccount'] > attitude_thresholds[('threat', 'authorAccount')]:
        classes.append('threat')

    if prediction['implTopic'] == '(a.1)' \
    and prediction['authorBelief'] > attitude_thresholds[('grievance', 'authorBelief')] \
    and prediction['authorPrefer'] < attitude_thresholds[('grievance', 'authorPrefer')] \
    and prediction['authorAccount'] < attitude_thresholds[('grievance', 'authorAccount')] \
    and prediction['typicalPrefer'] < attitude_thresholds[('grievance', 'typicalPrefer')] \
    and prediction['expertBelief'] < attitude_thresholds[('grievance', 'expertBelief')]:
        classes.append('grievance')

    if prediction['implStereotype'] == 'Yes':
        classes.append('stereotype')

    if prediction['hasOther'] == 'Yes' \
    and prediction['implPolarity'] == 'Negative':
        classes.append('inferiority')

    if prediction['implTopic'] == '(b.1)' \
    and prediction['authorBelief'] > attitude_thresholds[('dehumanization', 'authorBelief')]:
        classes.append('dehumanization')

    if len(classes) == 0:
        return [OTHER]

    return classes
