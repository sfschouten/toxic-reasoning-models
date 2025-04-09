from typing import Optional
from enum import Enum

import pandas as pd

from pydantic import BaseModel, Field


class GroupRole(str, Enum):
    author = "the author themselves and/or their ingroup"
    participant = "another participant in the conversation and/or the group they belong to"
    outside = "an individual outside of the conversation"
    another = "another group"
    na = "none of the above"


class GroupCharacteristic(str, Enum):
    sexual = "Sexual orientation"
    gender = "Gender"
    disability = "Disability"
    race = "Race/Ethnicity"
    age = "Age"
    religion = "Religion"
    famous = "Famous individual"
    political = "Political affiliation"
    social = "Social belief"
    body_image = "Body image"
    addiction = "Addiction"
    socioeconomic = "Socioeconomic status"
    profession = "Profession"
    nationality = "Nationality"
    other = "Other"
    na = "Not applicable"


class ImplicationCategory(str, Enum):
    a = ("the subject's circumstances, living conditions, physical condition or health, general wellbeing, access to "
         "resources, etc.")
    a1 = "some kind of harm coming to the subject"
    # b = "the subject's (inherent) qualities, their nature, abilities, etc."
    b = "the subject's nature, inherent qualities or abilities, etc."
    b1 = "dehumanisation of the subject"
    c = "the subject's choices/decisions, lifestyle, beliefs, etc."
    d = "a non-specific comparison (does not fall under other categories) between the subject and the other"
    e = "unclear or none of the above"


class Polarity(str, Enum):
    positive = "Positive"
    neutral = "Neutral"
    negative = "Negative"


class Temporality(str, Enum):
    past = "Past"
    present = "Present"
    future = "Future"


class ToxicReasoning(BaseModel):

    # short description of implication
    implication: str = Field(
        description="If the message might imply something toxic about a person or a group, give the main implication "
                    "as a single sentence. Do not describe the implication, but simply make explicit that which is "
                    "implied; i.e. do not start with 'The comment/author implies ...', instead the sentence should "
                    "generally start with the subject and continue with what is implied about the subject. If there "
                    "is no clear implication do not generate a ToxicReasoning."
    )

    # SUBJECT
    subject_descr: str = Field(description="The subject (person or group) of the implication.")
    subject_role: GroupRole = Field(description="The role of the subject is in the context of this thread.")
    subject_span: str = Field(description="The span in the original text most indicative of the subject.")
    subject_characteristic: GroupCharacteristic = Field(
        description="The characteristic that defines the subject group (if the subject is a group)."
    )

    # OTHER
    has_other: bool = Field(description="If there is an 'other' to which the subject is (explicitly or implicitly) compared.")
    other_descr: str = Field(
        description="A description of the optional 'other' person or group."
    )
    other_role: GroupRole = Field(description="Who the other is in the context of this thread.")
    other_span: str = Field(description="The span in the original text most indicative of the other.")

    # IMPLICATION
    category: ImplicationCategory = Field(
        description="The category (topic) of the implication. Choose the most specific option that clearly applies."
    )
    impl_span: str = Field(
        description="The span in the original text most indicative of the Implication Category."
    )
    polarity: Polarity = Field(
        description="If positive, the main implication being true should be something for the subject to be happy about, or proud of. "
                    "If negative, the main implication being true should be something for the subject to be sad about, or ashamed of."
    )
    stereotype: bool = Field(description="If the implication plays into a widely known stereotype.")
    sarcasm: bool = Field(description="If the implication is conveyed through sarcasm.")
    when: list[Temporality] = Field(description="At what point(s) in time the implication is meant to hold (at least 1).")

    # STAKEHOLDER ATTITUDES
    author_belief: float = Field(
        description="[0-1]: Probability that the author believes what their implication says was/is/will (or should [have] be[en]) the case."
    )
    author_preference: float = Field(
        description="[0-1]: Probability that the author prefers what their implication says was/is/will (or should [have] be[en]) the case."
    )
    author_responsibility: float = Field(
        description="[0-1]: Probability that the author is or feels personally responsible for the (past, present, or "
                    "future) truth of what their implication says was/is/will (or should [have] be[en]) the case."
    )
    typical_belief: float = Field(
        description="[0-1]: Probability that ordinary people would believe what the implication says was/is/will (or should [have] be[en]) the case."
    )
    typical_preference: float = Field(
        description="[0-1]: Probability that ordinary people would prefer what the implication says was/is/will (or should [have] be[en]) the case."
    )
    expert_belief: float = Field(
        description="[0-1]: Probability that experts would believe what the implication says was/is/will (or should [have] be[en]) the case."
    )


class Toxicity(str, Enum):
    yes = "Yes/Maybe"
    no = "No"


class CommentAnnotation(BaseModel):
    message_nr: int = Field(description="The number of the message in the thread to which this annotation applies.")

    is_toxic: Toxicity = Field(description="If this comment might be perceived as toxic.")
    is_only_innapropriate: bool = Field(description="If the comment is only toxic because of an inappropriate or toxic word (rather than what is being said/implied).")
    is_counter_speech: bool = Field(description="If the comment is an argument-based counter of the previous comment, providing an alternative perspective.")
    toxic_reasoning: Optional[ToxicReasoning] = Field(
        description="If toxic and the toxicity stems from what the text is either explicitly or implicitly "
                    "communicating this should contain a toxic reasoning. If toxic for a different reason, it should be"
                    " null."
    )


class ThreadReasonings(BaseModel):
    comment_annotations: list[CommentAnnotation] = Field(
        description="A list of comment annotations specifying message toxicity."
    )


def from_answers_and_labels(example_dict):
    comment_annotations = []
    for i, toxicity in enumerate(example_dict['label_toxicity']):
        reasoning = None
        if example_dict['label_hasImplication'][i] == 1:
            subject_charac = GroupCharacteristic.na
            subject_charac_arr = example_dict['answer_pp_subjectGroupType'][i]
            if len(subject_charac_arr) > 0 and subject_charac_arr != ['_other']:
                subject_charac = GroupCharacteristic(subject_charac_arr[0][1:])  # TODO sample

            def default(v):
                return v is None or pd.isna(v)

            subject_role = GroupRole.na if default(v := example_dict['answer_pp_subject'][i]) else GroupRole(v)
            other_role = GroupRole.na if default(v := example_dict['answer_pp_other'][i]) else GroupRole(v)
            implTopic = ImplicationCategory.e if default(v := example_dict['answer_pp_implTopic2'][i]) else ImplicationCategory(v)
            polarity = Polarity.neutral if default(v := example_dict['answer_pp_implPolarity'][i]) else Polarity(v)

            reasoning = ToxicReasoning(
                implication="" if (v := example_dict['answer_pp_implication'][i]) is None else v,
                subject_descr="" if (v := example_dict['answer_pp_subjectGroup'][i]) is None else v,
                subject_role=subject_role,
                subject_span=" ".join(tok.split('_')[1] for tok in example_dict['answer_pp_subjectTokens'][i]),
                subject_characteristic=subject_charac,
                has_other=example_dict['label_hasOther'][i] == 1,
                other_descr="" if (v := example_dict['answer_pp_otherGroup'][i]) is None else v,
                other_role=other_role,
                other_span=" ".join(tok.split('_')[1] for tok in example_dict['answer_pp_otherTokens'][i]),
                category=implTopic,
                impl_span=" ".join(tok.split('_')[1] for tok in example_dict['answer_pp_implTopicTokens'][i]),
                polarity=polarity,
                stereotype=example_dict['label_implStereotype'][i] == 1,
                sarcasm=example_dict['label_implSarcasm'][i] == 1,
                when=[Temporality(t[1:]) for t in example_dict['answer_pp_implTemporality'][i]],
                author_belief=example_dict['label_authorBelief'][i],
                author_preference=example_dict['label_authorPrefer'][i],
                author_responsibility=example_dict['label_authorAccount'][i],
                typical_belief=example_dict['label_typicalBelief'][i],
                typical_preference=example_dict['label_typicalPrefer'][i],
                expert_belief=example_dict['label_expertBelief'][i],
            )

        comment_annotation = CommentAnnotation(
            message_nr=i+1,
            is_toxic=Toxicity.yes if toxicity == 1 else Toxicity.no,
            is_only_innapropriate=example_dict['label_justInappropriate'][i] == 1,
            is_counter_speech=example_dict['label_justInappropriate'][i] == 1,
            toxic_reasoning=reasoning,
        )

        comment_annotations.append(comment_annotation)

    return ThreadReasonings(comment_annotations=comment_annotations)
