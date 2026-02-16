import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

class MyRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = RobertaClassificationHead(config)
        # self.alpha = torch.as_tensor([0.754631, 0.126094, 0.063217, 0.042233, 0.006326, 0.002803, 0.004242, 0.000303, 0.000152])
        self.alpha = torch.as_tensor([0.739931, 0.137938, 0.064025, 0.044707, 0.005727, 0.003169, 0.004161, 0.000191, 0.000153])
        self.gamma = None
        self.smooth = 1e-4
        self.epsilon = None
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        relation_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        logits = logits + relation_mask

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                if (self.alpha is not None) and (self.gamma is not None):
                    logits = logits.view(-1, self.num_labels)
                    prob = torch.softmax(logits, dim=1)
                    if self.epsilon is not None:
                        prob = (1 - self.epsilon) * prob + (self.epsilon / self.num_labels)
                    alpha = self.alpha.to(logits.device)
                    labels = labels.view(-1, 1)
                    prob = prob.gather(1, labels).view(-1) + self.smooth
                    logpt = torch.log(prob)
                    alpha_class = alpha[labels.squeeze().long()]
                    class_weight = -alpha_class * torch.pow(torch.sub(1.0, prob), self.gamma)
                    loss = class_weight * logpt
                    loss = loss.mean()
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
