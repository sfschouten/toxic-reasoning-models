from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from transformers import XLMRobertaPreTrainedModel, XLMRobertaModel
from transformers.utils import ModelOutput


class XLMRobertaClassificationHead(nn.Module):

    def __init__(self, config, num_out):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_out)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@dataclass
class ToxicReasoningOutput(ModelOutput):
    loss: torch.FloatTensor = None


class XLMRobertaForToxicReasoning(XLMRobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.comment_token_id = None
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # Classifiers
        self.toxicity_classifier = XLMRobertaClassificationHead(config, 1)
        self.counternarrative_classifier = XLMRobertaClassificationHead(config, 1)
        self.justInappropriate_classifier = XLMRobertaClassificationHead(config, 1)
        self.hasImplication_classifier = XLMRobertaClassificationHead(config, 1)

        self.subject_classifier = XLMRobertaClassificationHead(config, 5)
        self.subjectGroupType_classifier = XLMRobertaClassificationHead(config, 15)
        self.subject_token_classifier = nn.Linear(config.hidden_size, 1)

        self.hasOther_classifier = XLMRobertaClassificationHead(config, 1)
        self.other_classifier = XLMRobertaClassificationHead(config, 5)
        self.other_token_classifier = nn.Linear(config.hidden_size, 1)

        self.implTopic_classifier = XLMRobertaClassificationHead(config, 7)
        self.implTopic_token_classifier = nn.Linear(config.hidden_size, 1)
        self.implPolarity_classifier = XLMRobertaClassificationHead(config, 3)

        self.implTemporality_classifier = XLMRobertaClassificationHead(config, 3)
        self.implStereotype_classifier = XLMRobertaClassificationHead(config, 1)
        self.implSarcasm_classifier = XLMRobertaClassificationHead(config, 1)

        self.authorBelief_classifier = XLMRobertaClassificationHead(config, 1)
        self.authorPrefer_classifier = XLMRobertaClassificationHead(config, 1)
        self.authorAccount_classifier = XLMRobertaClassificationHead(config, 1)
        self.typicalBelief_classifier = XLMRobertaClassificationHead(config, 1)
        self.typicalPrefer_classifier = XLMRobertaClassificationHead(config, 1)
        self.expertBelief_classifier = XLMRobertaClassificationHead(config, 1)

        self.bce_loss_fct = nn.BCEWithLogitsLoss()
        self.ce_loss_fct = nn.CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def further_init(self, comment_token_id):
        self.comment_token_id = comment_token_id
        if comment_token_id >= self.roberta.embeddings.word_embeddings.num_embeddings:
            # comment token has no embedding, so add one
            self.roberta.resize_token_embeddings(self.comment_token_id + 1)
            s_embedding = self.roberta.embeddings.word_embeddings.weight[self.config.bos_token_id].detach().clone()
            with torch.no_grad():
                self.roberta.embeddings.word_embeddings.weight[self.comment_token_id] = s_embedding

    def forward(
            self,
            input_ids: torch.LongTensor,                                 # batch_size x seq_len
            attention_mask: torch.FloatTensor,                           # batch_size x seq_len
            label_toxicity: Optional[torch.LongTensor] = None,           # batch_nr_comments
            label_counternarrative: Optional[torch.LongTensor] = None,   # batch_nr_comments
            label_justInappropriate: Optional[torch.LongTensor] = None,  # batch_nr_comments
            label_hasImplication: Optional[torch.LongTensor] = None,     # batch_nr_comments
            label_subject: Optional[torch.LongTensor] = None,            # batch_nr_comments
            label_subjectGroupType: Optional[torch.LongTensor] = None,   # batch_nr_comments
            label_subjectTokens: Optional[torch.LongTensor] = None,      # batch_size x seq_len
            label_hasOther: Optional[torch.LongTensor] = None,           # batch_nr_comments
            label_other: Optional[torch.LongTensor] = None,              # batch_nr_comments
            label_otherTokens: Optional[torch.LongTensor] = None,        # batch_size x seq_len
            label_implTopic: Optional[torch.LongTensor] = None,          # batch_nr_comments
            label_implTopicTokens: Optional[torch.LongTensor] = None,    # batch_size x seq_len
            label_implPolarity: Optional[torch.LongTensor] = None,       # batch_nr_comments
            label_implTemporality: Optional[torch.LongTensor] = None,    # batch_nr_comments
            label_implStereotype: Optional[torch.LongTensor] = None,     # batch_nr_comments
            label_implSarcasm: Optional[torch.LongTensor] = None,        # batch_nr_comments
            label_authorBelief: Optional[torch.FloatTensor] = None,      # batch_nr_comments
            label_authorPrefer: Optional[torch.FloatTensor] = None,      # batch_nr_comments
            label_authorAccount: Optional[torch.FloatTensor] = None,     # batch_nr_comments
            label_typicalBelief: Optional[torch.FloatTensor] = None,     # batch_nr_comments
            label_typicalPrefer: Optional[torch.FloatTensor] = None,     # batch_nr_comments
            label_expertBelief: Optional[torch.FloatTensor] = None,      # batch_nr_comments
    ):
        if self.comment_token_id is None:
            raise ValueError("Still needs further initialization (call `further_init`).")

        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        comment_mask = input_ids == self.comment_token_id       # batch_nr_comments
        comment_repr = sequence_output[comment_mask]            # batch_nr_comments x nr_dims

        def masked_bce(out, labels):
            if labels is None:
                return None
            mask = labels != -100
            out, lbl = out[mask].view(mask.sum()), labels[mask].float().view(mask.sum())
            return self.bce_loss_fct(out, lbl)

        def apply_ce(out, labels):
            if labels is None:
                return None
            return self.ce_loss_fct(out, labels)

        preds_and_loss = {
            'toxicity': (out := self.toxicity_classifier(comment_repr), masked_bce(out, label_toxicity)),
            'counternarrative': (out := self.counternarrative_classifier(comment_repr), masked_bce(out, label_counternarrative)),
            'justInappropriate': (out := self.justInappropriate_classifier(comment_repr), masked_bce(out, label_justInappropriate)),
            'hasImplication': (out := self.hasImplication_classifier(comment_repr), masked_bce(out, label_hasImplication)),

            'subject': (out := self.subject_classifier(comment_repr).squeeze(), apply_ce(out, label_subject)),
            'subjectGroupType': (out := self.subjectGroupType_classifier(comment_repr), masked_bce(out, label_subjectGroupType)),
            'subject_token': ((out := self.subject_token_classifier(sequence_output)), masked_bce(out, label_subjectTokens)),

            'hasOther': (out := self.hasOther_classifier(comment_repr), masked_bce(out, label_hasOther)),
            'other': (out := self.other_classifier(comment_repr).squeeze(), apply_ce(out, label_other)),
            'other_token': (out := self.other_token_classifier(sequence_output), masked_bce(out, label_otherTokens)),

            'implTopic': (out := self.implTopic_classifier(comment_repr).squeeze(), apply_ce(out, label_implTopic)),
            'implTopic_token': (out := self.implTopic_token_classifier(sequence_output), masked_bce(out, label_implTopicTokens)),
            'implPolarity': (out := self.implPolarity_classifier(comment_repr).squeeze(), apply_ce(out, label_implPolarity)),

            'implTemporality': (out := self.implTemporality_classifier(comment_repr), masked_bce(out, label_implTemporality)),
            'implStereotype': (out := self.implStereotype_classifier(comment_repr), masked_bce(out, label_implStereotype)),
            'implSarcasm': (out := self.implSarcasm_classifier(comment_repr), masked_bce(out, label_implSarcasm)),

            'authorBelief': (out := self.authorBelief_classifier(comment_repr), masked_bce(out, label_authorBelief)),
            'authorPrefer': (out := self.authorPrefer_classifier(comment_repr), masked_bce(out, label_authorPrefer)),
            'authorAccount': (out := self.authorAccount_classifier(comment_repr), masked_bce(out, label_authorAccount)),
            'typicalBelief': (out := self.typicalBelief_classifier(comment_repr), masked_bce(out, label_typicalBelief)),
            'typicalPrefer': (out := self.typicalPrefer_classifier(comment_repr), masked_bce(out, label_typicalPrefer)),
            'expertBelief': (out := self.expertBelief_classifier(comment_repr), masked_bce(out, label_expertBelief)),
        }
        preds = {key: value[0] for key, value in preds_and_loss.items()}
        losses = {key: value[1] for key, value in preds_and_loss.items()}
        losses['total'] = sum(value for value in losses.values() if value is not None)

        return preds, losses
