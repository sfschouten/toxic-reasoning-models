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


def edos(prediction, attitude_thresholds=LATENT_HATRED_THRESHOLD_DEFAULTS):

    if prediction['toxicity'] == 'No':
        return [NOT_TOXIC]
    if prediction['justInappropriate'] == 'Yes':
        return [JUST_INAPPROPRIATE]
    if prediction['hasImplication'] == "['_Different kind of toxicity']":
        return [DIFFERENT_KIND]

    classes = []
    # 1.1 threats of harm
    if prediction['implTopic'] == '(a.1)' \
    and 'future' in prediction['implTemporality'] \
    and prediction['authorPrefer'] > attitude_thresholds[('threat', 'authorPrefer')] \
    and prediction['authorAccount'] > attitude_thresholds[('threat', 'authorAccount')]:
        classes.append('1.1 threats of harm')
    # 1.2 incitement and encouragement of harm
    if prediction['implTopic'] == '(a.1)' \
    and 'future' in prediction['implTemporality'] \
    and prediction['authorPrefer'] > attitude_thresholds[('incitement', 'authorPrefer')]:
        classes.append('1.2 incitement and encouragement of harm')

    # 2.1 descriptive attack
    if prediction['implTopic'] == '(b)' \
    and prediction['authorBelief'] > attitude_thresholds[('descriptive', 'authorBelief')] \
    and prediction['expertBelief'] < attitude_thresholds[('descriptive', 'expertBelief')]:
        classes.append('2.1 descriptive attacks')
    # 2.2 aggressive and emotive attacks -  not compatible with Toxic Reasoning, since it is not defined based on
    #                                       what is being communicated
    # 2.3 dehumanisation and overt sexual objectification
    if prediction['implTopic'] == '(b.1)' \
    and prediction['authorBelief'] > attitude_thresholds[('dehumanization', 'authorBelief')]:
        classes.append('2.3 dehumanising attacks & overt sexual objectification')

    # 3.1 casual use of gendered slurs, profanities & insults - not compatible
    # 3.2 immutable gender stereotypes
    if prediction['implStereotype'] == 'Yes' \
    and prediction['authorBelief'] > attitude_thresholds[('stereotypes', 'authorBelief')] \
    and prediction['expertBelief'] < attitude_thresholds[('stereotypes', 'expertBelief')]:
        classes.append('3.2 immutable gender differences and gender stereotypes')

    # 3.3 backhanded gendered compliments  -  not compatible
    # 3.4 condescending explanations or unwelcome advice  -  not compatible

    # TODO use 'subject' to differentiate between these two
    # 4.1 supporting mistreatment of individual women
    if prediction['implTopic'] == '(a)' \
    and prediction['authorPrefer'] > attitude_thresholds[('support_mistreatment', 'authorPrefer')]:
        classes.append('4.1 supporting mistreatment of individual women')
    # 4.2 supporting systemic discrimination against women
    if prediction['implTopic'] == '(a)' \
    and prediction['authorPrefer'] > attitude_thresholds[('support_discrimination', 'authorPrefer')]:
        classes.append('4.2 supporting systemic discrimination against women as a group')

    if len(classes) == 0:
        return [OTHER]

    return classes
