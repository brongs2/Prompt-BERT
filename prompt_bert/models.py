from re import template
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config, scale=1):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*scale, config.hidden_size*scale)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp



def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    if cls.model_args.mask_embedding_sentence_org_mlp:
        from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
        cls.mlp = BertPredictionHeadTransform(config)
    else:
        cls.mlp = MLPLayer(config, scale=cls.model_args.mask_embedding_sentence_num_masks)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def cl_forward(cls,
               encoder,
               input_ids=None,
               attention_mask=None,
               token_type_ids=None,
               position_ids=None,
               head_mask=None,
               inputs_embeds=None,
               output_attentions=None,
               output_hidden_states=None,
               labels=None,
               return_dict=None,
):
    def get_delta(template_token, length=50):
        with torch.set_grad_enabled(not cls.model_args.mask_embedding_sentence_delta_freeze):
            device = input_ids.device
            d_input_ids = torch.Tensor(template_token).repeat(length, 1).to(device).long()
            if cls.model_args.mask_embedding_sentence_autoprompt:
                d_inputs_embeds = encoder.embeddings.word_embeddings(d_input_ids)
                p = torch.arange(d_input_ids.shape[1]).to(d_input_ids.device).view(1, -1)
                b = torch.arange(d_input_ids.shape[0]).to(d_input_ids.device)
                for i, k in enumerate(cls.dict_mbv):
                    if cls.fl_mbv[i]:
                        index = ((d_input_ids == k) * p).max(-1)[1]
                    else:
                        index = ((d_input_ids == k) * -p).min(-1)[1]
                    #print(d_inputs_embeds[b,index][0].sum().item(), cls.p_mbv[i].sum().item())
                    #print(d_inputs_embeds[b,index][0].mean().item(), cls.p_mbv[i].mean().item())
                    d_inputs_embeds[b, index] = cls.p_mbv[i]
            else:
                d_inputs_embeds = None
            d_position_ids = torch.arange(d_input_ids.shape[1]).to(device).unsqueeze(0).repeat(length, 1).long()
            if not cls.model_args.mask_embedding_sentence_delta_no_position:
                d_position_ids[:, len(cls.bs)+1:] += torch.arange(length).to(device).unsqueeze(-1)
            m_mask = d_input_ids == cls.mask_token_id
            outputs = encoder(input_ids=d_input_ids if d_inputs_embeds is None else None ,
                              inputs_embeds=d_inputs_embeds,
                              position_ids=d_position_ids,  output_hidden_states=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            delta = last_hidden[m_mask]
            template_len = d_input_ids.shape[1]
            if cls.model_args.mask_embedding_sentence_org_mlp:
                delta = cls.mlp(delta)
            return delta, template_len

    if cls.model_args.mask_embedding_sentence_delta:
        delta, template_len = get_delta([cls.mask_embedding_template])
        if len(cls.model_args.mask_embedding_sentence_different_template) > 0:
            delta1, template_len1 = get_delta([cls.mask_embedding_template2])

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    if cls.model_args.mask_embedding_sentence_autoprompt:
        inputs_embeds = encoder.embeddings.word_embeddings(input_ids)
        p = torch.arange(input_ids.shape[1]).to(input_ids.device).view(1, -1)
        b = torch.arange(input_ids.shape[0]).to(input_ids.device)
        for i, k in enumerate(cls.dict_mbv):
            if cls.model_args.mask_embedding_sentence_autoprompt_continue_training_as_positive and i%2 == 0:
                continue
            if cls.fl_mbv[i]:
                index = ((input_ids == k) * p).max(-1)[1]
            else:
                index = ((input_ids == k) * -p).min(-1)[1]
            #print(inputs_embeds[b,index][0].sum().item(), cls.p_mbv[i].sum().item())
            #print(inputs_embeds[b,index][0].mean().item(), cls.p_mbv[i].mean().item())
            inputs_embeds[b, index] = cls.p_mbv[i]

    if inputs_embeds is not None:
        outputs = encoder(
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )
    else:
        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )

    # Pooling (applies to both branches)
    if cls.model_args.mask_embedding_sentence:
        last_hidden = outputs.last_hidden_state
        # Patch: handle CoOp prompt tokens when selecting mask token positions
        if cls.model_args.use_coop and cls.coop_length > 0:
            prompt_len = cls.coop_length
            mask = (input_ids == cls.mask_token_id)
            mask = F.pad(mask, (prompt_len, 0), value=0)
        else:
            mask = (input_ids == cls.mask_token_id)
        pooler_output = last_hidden[mask]

        if cls.model_args.mask_embedding_sentence_delta:
            if cls.model_args.mask_embedding_sentence_org_mlp:
                pooler_output = cls.mlp(pooler_output)

            if len(cls.model_args.mask_embedding_sentence_different_template) > 0:
                pooler_output = pooler_output.view(batch_size, num_sent, -1)
                attention_mask = attention_mask.view(batch_size, num_sent, -1)
                blen = attention_mask.sum(-1) - template_len
                pooler_output[:, 0, :] -= delta[blen[:, 0]]
                blen = attention_mask.sum(-1) - template_len1
                pooler_output[:, 1, :] -= delta1[blen[:, 1]]
                if num_sent == 3:
                    pooler_output[:, 2, :] -= delta1[blen[:, 2]]
            else:
                blen = attention_mask.sum(-1) - template_len
                pooler_output -= delta[blen]

        pooler_output = pooler_output.view(batch_size * num_sent, -1)
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))  # (bs, num_sent, hidden)

        # If using "cls", we add an extra MLP layer
        # (same as BERT's original implementation) over the representation.
        if not (cls.model_args.mask_embedding_sentence_delta and cls.model_args.mask_embedding_sentence_org_mlp):
            pooler_output = cls.mlp(pooler_output)
    else:
        try:
            pooler_output = outputs.pooler_output
        except AttributeError:
            pooler_output = outputs.last_hidden_state[:, 0, :]
    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    if cls.model_args.dot_sim:
        cos_sim = torch.mm(torch.sigmoid(z1), torch.sigmoid(z2.permute(1, 0)))
    else:
        cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative weight normalization
    if cls.model_args.norm_instead_temp:
        cos_sim *= cls.sim.temp
        cmin, cmax = cos_sim.min(), cos_sim.max()
        cos_sim = (cos_sim - cmin)/(cmax - cmin)/cls.sim.temp

    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        # Ensure cos_sim and z1_z3_cos are 2D before concatenation
        if cos_sim.dim() == 2 and z1_z3_cos.dim() == 2:
            cos_sim = torch.cat([cos_sim, z1_z3_cos], dim=1)

    # print(cos_sim[cm].mean()*cls.model_args.temp, cos_sim[~cm].mean()*cls.model_args.temp)
    #import pdb;pdb.set_trace()
    #cos_sim.topk(k=2, dim=-1)

    loss_fct = nn.CrossEntropyLoss()
    labels = torch.arange(cos_sim.size(0)).long().to(input_ids.device)

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(input_ids.device)
        cos_sim = cos_sim + weights

    loss = loss_fct(cos_sim, labels)

    # Calculate loss for MLM
    # if not cls.model_args.add_pseudo_instances and mlm_outputs is not None and mlm_labels is not None:
    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states if not cls.model_args.only_embedding_training else None,
        attentions=outputs.attentions if not cls.model_args.only_embedding_training else None,
    )




def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    # Simplified sent_emb forward: ignore mask_embedding functionality
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    # Flatten input_ids and attention_mask if not using inputs_embeds
    if inputs_embeds is None:
        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)
        seq_len = input_ids.size(2)
        flat_ids = input_ids.view(-1, seq_len)
        flat_attn = attention_mask.view(-1, seq_len)
        inputs_embeds = encoder.embeddings.word_embeddings(flat_ids)
        # If CoOp is enabled, prepend prompt embeddings
        if getattr(cls.model_args, 'use_coop', False) and cls.coop_length > 0:
            coop_len = cls.coop_length
            prompt_embeds = cls.prompt_embeddings.unsqueeze(0).expand(
                batch_size * num_sent, -1, -1
            )
            inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
            coop_mask = torch.ones(
                batch_size * num_sent, coop_len,
                device=flat_attn.device, dtype=flat_attn.dtype
            )
            attention_mask = torch.cat([coop_mask, flat_attn], dim=1)
            token_type_ids = None

    # Pass through encoder using inputs_embeds (or input_ids if provided)
    outputs = encoder(
        input_ids=None if inputs_embeds is not None else input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=False,
        return_dict=True,
    )

    # Extract pooler output (fallback to CLS hidden state if necessary)
    try:
        pooler_output = outputs.pooler_output
    except AttributeError:
        pooler_output = outputs.last_hidden_state[:, 0, :]

    # Apply MLP head unless only_embedding_training
    if not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )

class BertForCLCoOp(BertPreTrainedModel):
    """
    BERT contrastive learning with Contextual Prompting (CoOp).
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config)
        # CoOp: learnable prompt embeddings
        self.coop_length = getattr(self.model_args, 'coop_length', 0)
        if self.model_args.use_coop and self.coop_length > 0:
            # (coop_length, hidden_size)
            self.prompt_embeddings = nn.Parameter(
                torch.randn(self.coop_length, config.hidden_size)
            )
        # initialize contrastive head
        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
    ):
        # Prepare CoOp embeddings if enabled
        if inputs_embeds is None:
            # Flatten input_ids: (batch_size * num_sent, seq_len)
            batch_size, num_sent, seq_len = input_ids.size()
            flat_ids = input_ids.view(-1, seq_len)
            inputs_embeds = self.bert.embeddings.word_embeddings(flat_ids)

            if getattr(self.model_args, 'use_coop', False) and self.coop_length > 0:
                # Expand prompt: (batch_size * num_sent, coop_length, hidden)
                prompt = self.prompt_embeddings.unsqueeze(0).expand(
                    batch_size * num_sent, -1, -1
                )
                # Concatenate prompt + original embeddings
                inputs_embeds = torch.cat([prompt, inputs_embeds], dim=1)
                # Adjust attention mask
                flat_mask = attention_mask.view(-1, seq_len)
                coop_mask = torch.ones(
                    batch_size * num_sent, self.coop_length,
                    device=flat_mask.device, dtype=flat_mask.dtype
                )
                attention_mask = torch.cat([coop_mask, flat_mask], dim=1)
                token_type_ids = None  # or handle similarly if needed

        # Route to contrastive or embedding forward
        if sent_emb:
            return sentemb_forward(self, self.bert,
                                   input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds,
                                   labels=labels,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict)
        else:
            return cl_forward(self, self.bert,
                              input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds,
                              labels=labels,
                              output_attentions=output_attentions,
                              output_hidden_states=output_hidden_states,
                              return_dict=return_dict)

# 1. model.py 내부 - BertForCL 클래스 내부에 추가
class BertForCL(BertPreTrainedModel):
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.coop_length = getattr(self.model_args, 'coop_length', 0)
        self.bert = BertModel(config)

        if self.model_args.use_coop:
            self.prompt_embeddings = nn.Parameter(
                torch.randn(self.coop_length, config.hidden_size)
            )

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
    ):
        if inputs_embeds is None:
            batch_size, num_sent, seq_len = input_ids.size()
            flat_ids = input_ids.view(-1, seq_len)
            inputs_embeds = self.bert.embeddings.word_embeddings(flat_ids)

            if getattr(self.model_args, 'use_coop', False) and hasattr(self, "prompt_embeddings"):
                prompt = self.prompt_embeddings.unsqueeze(0).expand(
                    batch_size * num_sent, -1, -1
                )
                inputs_embeds = torch.cat([prompt, inputs_embeds], dim=1)
                flat_mask = attention_mask.view(-1, seq_len)
                coop_mask = torch.ones(
                    batch_size * num_sent, self.prompt_embeddings.size(0),
                    device=flat_mask.device, dtype=flat_mask.dtype
                )
                attention_mask = torch.cat([coop_mask, flat_mask], dim=1)
                token_type_ids = None

        if sent_emb:
            return sentemb_forward(self, self.bert,
                                   input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds,
                                   labels=labels,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict)
        else:
            return cl_forward(self, self.bert,
                              input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds,
                              labels=labels,
                              output_attentions=output_attentions,
                              output_hidden_states=output_hidden_states,
                              return_dict=return_dict)






class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        #self.roberta = RobertaModel(config)
        self.roberta = RobertaModel(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
