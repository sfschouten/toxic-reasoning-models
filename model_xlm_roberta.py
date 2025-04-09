from typing import Optional

import torch
import torch.nn as nn

from transformers import XLMRobertaPreTrainedModel, XLMRobertaModel


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


class XLMRobertaForToxicReasoning(XLMRobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.comment_token_id = None
        self.model = XLMRobertaModel(config, add_pooling_layer=False)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # Classifiers
        self.cls = torch.nn.ModuleDict({
            'toxicity':             XLMRobertaClassificationHead(config, 1),
            'counternarrative':     XLMRobertaClassificationHead(config, 1),
            'justInappropriate':    XLMRobertaClassificationHead(config, 1),
            'hasImplication':       XLMRobertaClassificationHead(config, 1),
            'subject':              XLMRobertaClassificationHead(config, 5),
            'subjectGroupType':     XLMRobertaClassificationHead(config, 15),
            'subjectTokens':        nn.Linear(config.hidden_size, 1),
            'hasOther':             XLMRobertaClassificationHead(config, 1),
            'other':                XLMRobertaClassificationHead(config, 5),
            'otherTokens':          nn.Linear(config.hidden_size, 1),
            'implTopic':            XLMRobertaClassificationHead(config, 7),
            'implTopicTokens':      nn.Linear(config.hidden_size, 1),
            'implPolarity':         XLMRobertaClassificationHead(config, 3),
            'implTemporality':      XLMRobertaClassificationHead(config, 3),
            'implStereotype':       XLMRobertaClassificationHead(config, 1),
            'implSarcasm':          XLMRobertaClassificationHead(config, 1),
            'authorBelief':         XLMRobertaClassificationHead(config, 1),
            'authorPrefer':         XLMRobertaClassificationHead(config, 1),
            'authorAccount':        XLMRobertaClassificationHead(config, 1),
            'typicalBelief':        XLMRobertaClassificationHead(config, 1),
            'typicalPrefer':        XLMRobertaClassificationHead(config, 1),
            'expertBelief':         XLMRobertaClassificationHead(config, 1),
        })

        self.bce_loss_fct = nn.BCEWithLogitsLoss()
        self.ce_loss_fct = nn.CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def further_init(self, comment_token_id):
        self.comment_token_id = comment_token_id
        if comment_token_id >= self.model.embeddings.word_embeddings.num_embeddings:
            # comment token has no embedding, so add one
            print(f'WARNING: randomly initializing embedding for {comment_token_id}.')
            self.model.resize_token_embeddings(self.comment_token_id + 1)
            s_embedding = self.model.embeddings.word_embeddings.weight[self.config.bos_token_id].detach().clone()
            with torch.no_grad():
                self.model.embeddings.word_embeddings.weight[self.comment_token_id] = s_embedding

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

        outputs = self.model(input_ids, attention_mask=attention_mask)
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
            (k := 'toxicity'): (out := self.cls[k](comment_repr), masked_bce(out, label_toxicity)),
            (k := 'counternarrative'): (out := self.cls[k](comment_repr), masked_bce(out, label_counternarrative)),
            (k := 'justInappropriate'): (out := self.cls[k](comment_repr), masked_bce(out, label_justInappropriate)),
            (k := 'hasImplication'): (out := self.cls[k](comment_repr), masked_bce(out, label_hasImplication)),

            (k := 'subject'): (out := self.cls[k](comment_repr).squeeze(), apply_ce(out, label_subject)),
            (k := 'subjectGroupType'): (out := self.cls[k](comment_repr), masked_bce(out, label_subjectGroupType)),
            (k := 'subjectTokens'): ((out := self.cls[k](sequence_output)), masked_bce(out, label_subjectTokens)),

            (k := 'hasOther'): (out := self.cls[k](comment_repr), masked_bce(out, label_hasOther)),
            (k := 'other'): (out := self.cls[k](comment_repr).squeeze(), apply_ce(out, label_other)),
            (k := 'otherTokens'): (out := self.cls[k](sequence_output), masked_bce(out, label_otherTokens)),

            (k := 'implTopic'): (out := self.cls[k](comment_repr).squeeze(), apply_ce(out, label_implTopic)),
            (k := 'implTopicTokens'): (out := self.cls[k](sequence_output), masked_bce(out, label_implTopicTokens)),
            (k := 'implPolarity'): (out := self.cls[k](comment_repr).squeeze(), apply_ce(out, label_implPolarity)),

            (k := 'implTemporality'): (out := self.cls[k](comment_repr), masked_bce(out, label_implTemporality)),
            (k := 'implStereotype'): (out := self.cls[k](comment_repr), masked_bce(out, label_implStereotype)),
            (k := 'implSarcasm'): (out := self.cls[k](comment_repr), masked_bce(out, label_implSarcasm)),

            (k := 'authorBelief'): (out := self.cls[k](comment_repr), masked_bce(out, label_authorBelief)),
            (k := 'authorPrefer'): (out := self.cls[k](comment_repr), masked_bce(out, label_authorPrefer)),
            (k := 'authorAccount'): (out := self.cls[k](comment_repr), masked_bce(out, label_authorAccount)),
            (k := 'typicalBelief'): (out := self.cls[k](comment_repr), masked_bce(out, label_typicalBelief)),
            (k := 'typicalPrefer'): (out := self.cls[k](comment_repr), masked_bce(out, label_typicalPrefer)),
            (k := 'expertBelief'): (out := self.cls[k](comment_repr), masked_bce(out, label_expertBelief)),
        }
        preds = {key: value[0] for key, value in preds_and_loss.items()}
        losses = {key: value[1] for key, value in preds_and_loss.items()}
        losses['total'] = sum(value for value in losses.values() if value is not None)

        return preds, losses
