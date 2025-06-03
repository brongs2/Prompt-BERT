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
    
    if not cls.model_args.mask_embedding_sentence:
        if not cls.model_args.use_coop:
            batch_size, num_sent, seq_len = input_ids.size()
            input_ids_flat = input_ids.view(batch_size * num_sent, seq_len)
            attention_mask_flat = attention_mask.view(batch_size * num_sent, -1)
            
            outputs_enc = encoder(
                input_ids=input_ids_flat,
                attention_mask=attention_mask_flat,
                output_hidden_states=False,
                return_dict=True,
            )
            pooler = outputs_enc.pooler_output  # (B*num_sent, H)
            pooler = pooler.view(batch_size, num_sent, -1)  # (B, num_sent, H)
        else:
            # print("use coop")
            # [DEBUG] Print input shapes and expected sequence length for CoOp branch
            batch_size, num_sent, seq_len = input_ids.size()
            expected_len = cls.coop_length + seq_len
            # —————————————— 디버그 시작 ——————————————
            # inputs_embeds와 attention_mask가 NaN을 포함하는지 확인
            # if torch.isnan(inputs_embeds).any():
            #     print("[DEBUG cl_forward] ⚠️ inputs_embeds contains NaN BEFORE encoder")
            # else:
            #     print("[DEBUG cl_forward] inputs_embeds OK (no NaN) BEFORE encoder")

            # if torch.isnan(attention_mask).any():
            #     print("[DEBUG cl_forward] ⚠️ attention_mask contains NaN BEFORE encoder")
            # else:
            #     print("[DEBUG cl_forward] attention_mask OK (no NaN) BEFORE encoder")

            # # inputs_embeds의 일부 값을 샘플링해서 찍어 보기 (예: 첫 문장 첫 토큰)
            # print("[DEBUG cl_forward] inputs_embeds[0, :5, :5] =", inputs_embeds[0, :5, :5])
            # print("[DEBUG cl_forward] attention_mask[0, :10] =", attention_mask.view(-1)[0:10])
           # —————————————— 디버그 끝 ——————————————
            assert inputs_embeds.shape[1] == expected_len, f"CoOp inputs_embeds length {inputs_embeds.shape[1]} != expected {expected_len}"
            assert attention_mask.shape[1] == expected_len, f"CoOp attention_mask length {attention_mask.shape[1]} != expected {expected_len}"
            # print(encoder)
            outputs_enc = encoder(
                attention_mask=attention_mask.view(-1, attention_mask.size(-1)),
                inputs_embeds=inputs_embeds,   # 이미 (B*num_sent, coop_length+seq_len, H) 형태
                output_hidden_states=False,
                return_dict=True,
            )
            pooler = outputs_enc.pooler_output  # (B*num_sent, H)
            # print("pooler= ", pooler)
            batch_size, num_sent, _ = input_ids.size()
            pooler = pooler.view(batch_size, num_sent, -1)
            
        z1, z2 = pooler[:, 0], pooler[:, 1]
        if num_sent == 3:
            z3 = pooler[:, 2]

        # If distributed, gather across GPUs
        if dist.is_initialized() and cls.training:
            # Hard negative gathering (if present)
            if num_sent >= 3:
                z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
                z3_list[dist.get_rank()] = z3
                z3 = torch.cat(z3_list, 0)

            # Gather z1 & z2
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)
        # ---- compact debug (prints once) ---------------------------------
        if not hasattr(cls, "_dbg_printed"):
            print("[DEBUG] NaN?  z1:", torch.isnan(z1).any().item(),
                  " z2:", torch.isnan(z2).any().item(),
                  " cos_sim:", torch.isnan(cos_sim if 'cos_sim' in locals() else z1).any().item())
            cls._dbg_printed = True
        # ------------------------------------------------------------------
        # Compute pairwise similarity
        if cls.model_args.dot_sim:
            cos_sim = torch.mm(torch.sigmoid(z1), torch.sigmoid(z2.permute(1, 0)))
        else:
            cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        # (Optional) renormalize instead of using temperature directly
        if cls.model_args.norm_instead_temp:
            cos_sim *= cls.sim.temp
            cmin, cmax = cos_sim.min(), cos_sim.max()
            cos_sim = (cos_sim - cmin) / (cmax - cmin) / cls.sim.temp

        # If there is a hard negative, append its similarity
        if num_sent >= 3:
            z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        # If using a hard-negative weight, add it to the logits
        if num_sent == 3:
            z3_weight = cls.model_args.hard_negative_weight
            weights = torch.tensor(
                [
                    [0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1))
                    + [0.0] * i
                    + [z3_weight]
                    + [0.0] * (z1_z3_cos.size(-1) - i - 1)
                    for i in range(z1_z3_cos.size(-1))
                ],
                device=input_ids.device,
            )
            cos_sim = cos_sim + weights

        # Contrastive loss
        loss_fct = nn.CrossEntropyLoss()
        labels_contrast = torch.arange(cos_sim.size(0), dtype=torch.long, device=input_ids.device)
        loss = loss_fct(cos_sim, labels_contrast)

        # If caller wants a tuple, return (loss, logits). Otherwise return SequenceClassifierOutput.
        if not return_dict:
            return (loss, cos_sim)
        return SequenceClassifierOutput(
            loss=loss,
            logits=cos_sim,
            hidden_states=outputs_enc.hidden_states if not cls.model_args.only_embedding_training else None,
            attentions=outputs_enc.attentions if not cls.model_args.only_embedding_training else None,
        )
        
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


    # Pooling
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
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

        # If using "cls", we add an extra MLP layer
        # (same as BERT's original implementation) over the representation.
        if not (cls.model_args.mask_embedding_sentence_delta and cls.model_args.mask_embedding_sentence_org_mlp):
            pooler_output = cls.mlp(pooler_output)
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
        #if cls.model_args.mask_embedding_sentence_whole_vocab_cl:
            #z1, z2 = torch.sigmoid(z1), torch.sigmoid(z2)
        cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        #print(cos_sim)
        #import pdb;pdb.set_trace()
    # Hard negative
    if cls.model_args.norm_instead_temp:
        cos_sim *= cls.sim.temp
        cmin, cmax = cos_sim.min(), cos_sim.max()
        cos_sim = (cos_sim - cmin)/(cmax - cmin)/cls.sim.temp

    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

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

    if cls.model_args.mask_embedding_sentence_delta and not cls.model_args.mask_embedding_sentence_delta_no_delta_eval :
        device = input_ids.device
        d_input_ids = torch.Tensor([cls.mask_embedding_template]).repeat(128, 1).to(device).long()
        d_position_ids = torch.arange(d_input_ids.shape[1]).to(device).unsqueeze(0).repeat(128, 1).long()
        if not cls.model_args.mask_embedding_sentence_delta_no_position:
            d_position_ids[:, len(cls.bs)+1:] += torch.arange(128).to(device).unsqueeze(-1)
        m_mask = d_input_ids == cls.mask_token_id

        with torch.no_grad():
            outputs = encoder(input_ids=d_input_ids, position_ids=d_position_ids,  output_hidden_states=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            delta = last_hidden[m_mask]
        delta.requires_grad = False
        template_len = d_input_ids.shape[1]

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    if cls.model_args.mask_embedding_sentence and hasattr(cls, 'bs'):
        new_input_ids = []
        bs = torch.LongTensor(cls.bs).to(input_ids.device)
        es = torch.LongTensor(cls.es).to(input_ids.device)

        for i in input_ids:
            ss = i.shape[0]
            d = i.device
            ii = i[i != cls.pad_token_id]
            ni = [ii[:1], bs]
            if ii.shape[0] > 2:
                ni += [ii[1:-1]]
            ni += [es, ii[-1:]]
            if ii.shape[0] < i.shape[0]:
                ni += [i[i == cls.pad_token_id]]
            ni = torch.cat(ni)
            try:
                assert ss + bs.shape[0] + es.shape[0] == ni.shape[0]
            except:
                print(ss + bs.shape[0] + es.shape[0])
                print(ni.shape[0])
                print(i.tolist())
                print(ni.tolist())
                assert 0

            new_input_ids.append(ni)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = (input_ids != cls.pad_token_id).long()
        token_type_ids = None

    if cls.model_args.mask_embedding_sentence_autoprompt:
        inputs_embeds = encoder.embeddings.word_embeddings(input_ids)
        with torch.no_grad():
            p = torch.arange(input_ids.shape[1]).to(input_ids.device).view(1, -1)
            b = torch.arange(input_ids.shape[0]).to(input_ids.device)
            for i, k in enumerate(cls.dict_mbv):
                if cls.fl_mbv[i]:
                    index = ((input_ids == k) * p).max(-1)[1]
                else:
                    index = ((input_ids == k) * -p).min(-1)[1]
                inputs_embeds[b, index] = cls.p_mbv[i]

    outputs = encoder(
        None if cls.model_args.mask_embedding_sentence_autoprompt else input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=False,
        return_dict=True,
    )

    if cls.model_args.mask_embedding_sentence and hasattr(cls, 'bs'):
        last_hidden = outputs.last_hidden_state
        # Patch: handle CoOp prompt tokens when selecting mask token positions
        if cls.model_args.use_coop and cls.coop_length > 0:
            prompt_len = cls.coop_length
            mask = (input_ids == cls.mask_token_id)
            mask = F.pad(mask, (prompt_len, 0), value=0)
        else:
            mask = (input_ids == cls.mask_token_id)
        pooler_output = last_hidden[mask]
        if cls.model_args.mask_embedding_sentence_delta and not cls.model_args.mask_embedding_sentence_delta_no_delta_eval :
            blen = attention_mask.sum(-1) - template_len
            if cls.model_args.mask_embedding_sentence_org_mlp and not cls.model_args.mlp_only_train:
                pooler_output, delta = cls.mlp(pooler_output), cls.mlp(delta)
            pooler_output -= delta[blen]

        if cls.model_args.mask_embedding_sentence_avg:
            pooler_output = pooler_output.view(input_ids.shape[0], -1)
        else:
            pooler_output = pooler_output.view(input_ids.shape[0], -1, pooler_output.shape[-1]).mean(1)
    if not cls.model_args.mlp_only_train and not cls.model_args.mask_embedding_sentence_org_mlp:
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
        # print("[DEBUG] config.initializer_range =", config.initializer_range)
        # 1) __init__에서 프롬프트 임베딩을 생성하며 곧바로 GPU로 이동
        self.coop_length = getattr(self.model_args, 'coop_length', 0)
        if self.model_args.use_coop and self.coop_length > 0:
            # config.initializer_range를 사용해서 한 번만 초기화
            init_tensor = torch.randn(self.coop_length, config.hidden_size) * 0.02
            self.prompt_embeddings = nn.Parameter(init_tensor)
        else:
            self.prompt_embeddings = None

        # 2) contrastive head 초기화 (masking 등 필요한 분기 포함)
        cl_init(self, config)

        # 3) 전체 모델을 GPU로 옮길 때 prompt_embeddings도 함께 GPU로 가도록 설정
        #    Trainer가 model.to(device)를 호출하면 자동으로 처리됨

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
        # ----------------------------------------------------------------
        # 4) forward 초반에 “prompt_embeddings 디바이스 보장” 코드 추가
        if self.model_args.use_coop and self.prompt_embeddings is not None:
            # inputs_embeds가 None이면 아직 BERT 임베딩을 못 얻은 상태인데, 
            # 그때는 “입력 아이디 기반으로 임베딩을 만든 뒤”에 디바이스를 통일할 것이므로, 여기서는 패스
            pass

        # print(f"[DEBUG] inputs_embeds(초기 인자): {inputs_embeds}")  # inputs_embeds가 None 혹은 외부에서 들어왔는지 확인

        if inputs_embeds is None:
            # Flatten input_ids: (batch_size * num_sent, seq_len)
            batch_size, num_sent, seq_len = input_ids.size()
            flat_ids = input_ids.view(-1, seq_len)

            # 5) BERT token embedding 생성: 이 텐서는 (B*num_sent, seq_len, H) 형태로 GPU 위에 생성됨
            inputs_embeds = self.bert.embeddings.word_embeddings(flat_ids).to(flat_ids.device)
            # print(f"[DEBUG] inputs_embeds(벡터 생성 후) : {inputs_embeds.shape}")

            if self.model_args.use_coop and self.coop_length > 0:
                # 6) 이 시점에서 prompt_embeddings가 GPU로 올라와 있지 않다면 강제 이동
                if self.prompt_embeddings.device != inputs_embeds.device:
                    # 원본 파라미터 자체를 GPU로 옮기고 다시 래핑
                    self.prompt_embeddings = nn.Parameter(self.prompt_embeddings.data.to(inputs_embeds.device))

                # 7) 이제 prompt 임베딩을 (B*num_sent, coop_length, H)로 expand
                prompt = self.prompt_embeddings.unsqueeze(0).expand(
                    batch_size * num_sent, -1, -1
                )
                # 8) prompt + 원래 임베딩을 결합
                inputs_embeds = torch.cat([prompt, inputs_embeds], dim=1)
                # print(f"[DEBUG] inputs_embeds(프롬프트 합친 후) : {inputs_embeds.shape}")

                # 9) attention_mask도 대응해서 확장
                flat_mask = attention_mask.view(-1, seq_len)
                coop_mask = torch.ones(
                    batch_size * num_sent, self.coop_length,
                    device=inputs_embeds.device, dtype=flat_mask.dtype
                )
                # 빈 문장 방지
                flat_mask[flat_mask.sum(-1) == 0, 0] = 1
                attention_mask = torch.cat([coop_mask, flat_mask], dim=1)
                token_type_ids = None

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
