import torch
import torch.nn as nn
from base_model import *
from transformers import AutoModel, AutoConfig, AdamW
from collections import defaultdict


class DialoGPT(BaseModule):
    def __init__(
        self,
        transformer_model,
        config=None,
        lr=1e-5,
        dropout=0.2,
        cuda=True,
        warmup_ratio=0.1,
        num_training_steps=1000,
        device_idxs=(),
        #mixed_precision=False,
        # efficientnet_batchsize=32,
        pad_value=0
    ):
        super().__init__(cuda=cuda,
                         warmup_ratio=warmup_ratio,
                         num_training_steps=num_training_steps,
                         device_idxs=device_idxs,)
                         #mixed_precision=mixed_precision)

        self.config = config or AutoConfig.from_pretrained(transformer_model)
        self.transformer = AutoModel.from_pretrained(transformer_model, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.vocab_size)

        self.cross_entropy = nn.CrossEntropyLoss()

        self.reset()

        if self.cuda and len(self.devices) > 1:
            self.transformer = nn.DataParallel(self.transformer,
                                               device_ids=self.devices,
                                               output_device=self.model_device)

        #if self.mixed_precision:
        #    self.transformer.forward = insert_autocast(self.transformer.forward)

        self.optimizer = AdamW(self.parameters(), lr=lr)
        self.scheduler = self.linear_scheduler(self.optimizer)

        self.pad_value = pad_value

        # self.img_buffer = deque(efficientnet_batchsize)

        self.main_losses = {'rg'}

    def forward_single(
        self,
        input_ids,
        attention_mask=None,
        past=None,
        return_past=False
    ):
        position_ids = self.make_position_ids(attention_mask)
        if return_past:
            if past is not None:
                position_ids = position_ids[:, -1]
            outputs = self.transformer(input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past, use_cache=True, return_dict=True)
        else:
            outputs = self.transformer(input_ids, attention_mask=attention_mask, position_ids=position_ids, return_dict=True)

        hidden_state = self.dropout(outputs['last_hidden_state'])

        if return_past:
            responses = self.classifier(hidden_state[:, -1])
        else:
            responses = self.classifier(hidden_state)

        ret = {
            'logits': responses
        }
        if return_past:
            ret['past'] = outputs['past_key_values']

        return ret

    def forward(
        self,
        input_ids,
        attention_mask=None,
        max_len=1,
        past=None,
        return_past=False,
        end_tokens=[1279, 91, 437, 1659, 26209, 91, 29]
    ):
        input_ids = input_ids.to(self.model_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model_device)

        if self.training:
            return self.forward_single(input_ids,
                                       attention_mask=attention_mask,
                                       past=past,
                                       return_past=return_past)
        return self.inference(
            input_ids,
            attention_mask=attention_mask,
            max_len=max_len,
            past=past,
            end_tokens=end_tokens
        )

    @classmethod
    def shift_past(cls, past, shift=1):
        return tuple(p[:, :, :, shift:, :] for p in past)

    def inference(
        self,
        input_ids,
        attention_mask=None,
        max_len=1,
        past=None,
        end_tokens=[1279, 91, 437, 1659, 26209, 91, 29]
    ):
        input_ids = input_ids.to(self.model_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model_device)

        end_tokens = torch.tensor(end_tokens).to(self.model_device).view(-1)
        generation_mask = torch.full((input_ids.shape[0], end_tokens.shape[0]), -1).to(self.model_device)
        next_tokens = input_ids
        all_logits = []
        predictions = []
        past = None
        for i in range(max_len):
            outputs = self.forward_single(next_tokens, attention_mask=attention_mask, past=past, return_past=True)
            logits = outputs['logits'].unsqueeze(1)
            past = outputs['past']
            all_logits.append(logits.detach())
            next_tokens = logits.argmax(dim=-1)
            if i < generation_mask.shape[1]:
                diff_mask = torch.tensor(1).to(self.model_device)
                generation_mask[:, i] = next_tokens.view(-1)
            else:
                diff_mask = (generation_mask != end_tokens).any(dim=1)
                if generation_mask[diff_mask].nelement() == 0:
                    break
                generation_mask[diff_mask] = generation_mask[diff_mask].roll(-1, 1)
                generation_mask[diff_mask, -1] = next_tokens.view(-1)[diff_mask]
            if attention_mask.shape[1] < self.config.n_positions:
                attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1)).to(self.model_device) * diff_mask.view(-1, 1)], dim=1)
            else:
                attention_mask = attention_mask.roll(-1, 1)
                attention_mask[:, -1] = torch.ones(attention_mask.shape[0]).to(self.model_device) * diff_mask.view(-1)
                past = DialoGPT.shift_past(past)
            predicted_tokens = next_tokens.detach()
            predicted_tokens[attention_mask[:, -1] == 0] = self.pad_value
            predictions.append(predicted_tokens)
        predictions = torch.cat(predictions, dim=1)
        all_logits = torch.cat(all_logits, dim=1)
        return {
            'logits': all_logits,
            'rg': predictions
        }

    def reset(self):
        self.loss, self.loss_rg, self.iter = 0, 0, 1

    def predict_image(self, imgs):
        if self.efficientnet is None:
            return None

    def print_loss(self):
        loss_avg = self.loss / self.iter
        loss_rg = self.loss_rg / self.iter
        self.iter += 1
        return 'L:{:.2f} LR:{:.3f}'.format(loss_avg, loss_rg)

    def prepare(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        labels_begin = batch['labels_begin']
        labels_end = batch['labels_end']
        labels_mask = labels != self.pad_value
        end_tokens = batch['end_tokens'][0]
        max_len = batch['max_len'][0]

        # TODO
        # if not self.training:
        #     img_preds = self.predict_image(batch['imgs'])
        #     ...
            
        ids = batch['id']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_len': max_len,
            'end_tokens': end_tokens
        }, {
            'id': ids,
            'labels': labels,
            'labels_begin': labels_begin,
            'labels_end': labels_end,
            'labels_mask': labels_mask
        }

    def accumulate_loss(self, outputs, extras):
        logits = outputs['logits']
        labels = extras['labels'].to(self.model_device)
        labels_begin = extras['labels_begin']
        labels_end = extras['labels_end']
        labels_mask = extras['labels_mask']

        batch_loss_rg = 0
        for i, logit in enumerate(logits):
            label = labels[i][labels_mask[i]]
            if self.training:
                batch_loss_rg += self.cross_entropy(logit[labels_begin[i] - 1: labels_end[i] - 1], label)
            else:
                logit = logit[:label.shape[0]]
                if logit.shape[0] < label.shape[0]:
                    pad_tensor = torch.zeros((label.shape[0], logit.shape[1]), dtype=logit.dtype).to(self.model_device)
                    pad_tensor[:, self.pad_value] = 1
                    pad_tensor[:logit.shape[0]] = logit
                    logit = pad_tensor
                batch_loss_rg += self.cross_entropy(logit, label)
        batch_loss_rg /= logits.shape[0]

        loss = batch_loss_rg
        self.loss_rg += batch_loss_rg.item()

        self.loss_grad = loss
        self.loss += loss.data

    def make_results(self, outputs, extras):
        labels_begin = extras['labels_begin']
        labels_end = extras['labels_end']
        labels_mask = extras['labels_mask']
        if self.training:
            rg = [logit[labels_begin[i] - 1: labels_end[i] - 1].argmax(dim=-1) for i, logit in enumerate(outputs['logits'])]
        else:
            rg = outputs['rg']
        labels = extras['labels']
        ids = extras['id']

        results = defaultdict(list)

        for i, label in enumerate(labels):
            gt = label[labels_mask[i]].tolist()
            if self.training:
                pred = rg[i][:len(gt)].tolist()
            else:
                pred = rg[i][rg[i] != self.pad_value].tolist()

            results[ids[i].item()].append({
                'predictions': {
                    'rg': pred
                },
                'gts': {
                    'rg': gt
                }
            })

        return results
