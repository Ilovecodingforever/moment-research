import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


import torch
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2MLP,
    GPT2Block,
    GPT2Model,

)




class GPT2Attention_prompt(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        prompts: Optional[Tuple[torch.FloatTensor]] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)


        if prompts is not None:
            key = torch.cat([prompts[0], key], dim=-2)
            value = torch.cat([prompts[1], value], dim=-2)



        if use_cache is True:
            present = (key, value)
        else:
            present = None


        assert attention_mask is None and head_mask is None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2Block_prompt(GPT2Block):
    def __init__(self, config, c_in, head_dim, num_prefix, layer_idx=None):
        super().__init__(config)
        # hidden_size = config.hidden_size
        # inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        # self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention_prompt(config, layer_idx=layer_idx)
        # self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # if config.add_cross_attention:
        #     self.crossattention = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
        #     self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # self.mlp = GPT2MLP(inner_dim, config)

        self.config = config
        self.c_in = c_in
        self.head_dim = head_dim
        self.num_prefix = num_prefix

        per_layer_dim = config.n_head * self.head_dim
        total_dim = 2 * per_layer_dim
        reparam_dim = 32
        self.prompt_embed = (
            nn.Sequential(
                nn.Embedding(self.num_prefix*c_in, per_layer_dim),
                nn.Linear(per_layer_dim, reparam_dim),
                nn.Tanh(),
                nn.Linear(reparam_dim, total_dim),
            )
        )
        self.input_tokens = torch.arange(self.num_prefix*self.c_in)



    def generate_prefix_item(self, input_ids, embedding):
        bsz = input_ids.size(0)
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(input_ids.device)
        prefix = embedding(input_tokens)  # batch, seq, layer * embed * 2
        prefix = prefix.view(
            bsz,
            self.num_prefix*self.c_in,
            # self.config.num_hidden_layers,
            2,
            self.config.n_head,
            self.head_dim,
        )
        # prefix = prefix.permute([3, 2, 0, 4, 1, 5])  # 2, num_layers, bsz, num_heads, num_prefix, d_kv
        prefix = prefix.permute([2, 0, 3, 1, 4])  # 2, bsz, num_heads, num_prefix, d_kv
        return prefix[0], prefix[1]


    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states


        hidden_states = self.ln_1(hidden_states)
        # TODO: what about doing prompt here?



        n_channels = self.c_in
        batch_size, seq_length, d_model = hidden_states.shape
        batch_size_real = batch_size // n_channels

        hidden_states_ = hidden_states.reshape(-1, n_channels, seq_length, d_model)
        hidden_states_proj = hidden_states_.transpose(1, 2).reshape(-1, n_channels, d_model)
        # hidden_states_proj = self.ln_1(hidden_states_proj)
        attn_output, attn_output_weights = self.shared_prompt_projection['mha'](self.shared_prompt_projection['q'](hidden_states_proj),
                                                                                self.shared_prompt_projection['k'](hidden_states_proj),
                                                                                self.shared_prompt_projection['v'](hidden_states_proj))
        # attn_output = self.shared_prompt_projection['ln2'](self.shared_prompt_projection['act'](attn_output)).reshape(batch_size_real, seq_length, n_channels, -1).permute(0, 2, 3, 1)
        attn_output = (attn_output).reshape(batch_size_real, seq_length, n_channels, -1).permute(0, 2, 3, 1)

        # shared_prompt_projection_k = self.shared_prompt_projection['act'](self.shared_prompt_projection['linear_key'](attn_output)) # bs x channel x d_kv x (num_prefix*n_heads)
        shared_prompt_projection_k = (self.shared_prompt_projection['linear_key'](attn_output)) # bs x channel x d_kv x (num_prefix*n_heads)
        shared_prompt_projection_key = shared_prompt_projection_k.reshape(batch_size_real, n_channels, -1, self.num_prefix, self.config.n_head).permute(0, 4, 1, 3, 2).reshape(batch_size_real, self.config.n_head, n_channels*self.num_prefix, -1).repeat_interleave(n_channels, dim=0)
        # shared_prompt_projection_v = self.shared_prompt_projection['act'](self.shared_prompt_projection['linear_value'](attn_output))
        shared_prompt_projection_v = (self.shared_prompt_projection['linear_value'](attn_output))
        shared_prompt_projection_value = shared_prompt_projection_v.reshape(batch_size_real, n_channels, -1, self.num_prefix, self.config.n_head).permute(0, 4, 1, 3, 2).reshape(batch_size_real, self.config.n_head, n_channels*self.num_prefix, -1).repeat_interleave(n_channels, dim=0)

        # residual connection
        # prefix_key, prefix_value = self.generate_prefix_item(hidden_states, self.prompt_embed)
        shared_prompt_projection_key = self.prefix_key #+ shared_prompt_projection_key
        shared_prompt_projection_value = self.prefix_value #+ shared_prompt_projection_value
        
        # shared_prompt_projection_key = torch.cat([prefix_key, shared_prompt_projection_key], dim=-2)
        # shared_prompt_projection_value = torch.cat([prefix_value, shared_prompt_projection_value], dim=-2)

        # TODO: maybe should ln here?
        # TODO: different ln for key and value?
        # shared_prompt_projection_key = self.shared_prompt_projection['dropout'](shared_prompt_projection_key)
        # shared_prompt_projection_value = self.shared_prompt_projection['dropout'](shared_prompt_projection_value)
        # shared_prompt_projection_key = self.shared_prompt_projection['ln3_k'](shared_prompt_projection_key)
        # shared_prompt_projection_value = self.shared_prompt_projection['ln3_v'](shared_prompt_projection_value)
        

        # assert layer_past is None
        # layer_past = (shared_prompt_projection_key, shared_prompt_projection_value)

        prompts = (shared_prompt_projection_key, shared_prompt_projection_value)
        # prompts = None


        if attention_mask is not None:
            raise NotImplementedError("attention_mask not implemented")
            prefix_mask = torch.zeros(
                batch_size, 1, attention_mask.size(2), self.num_prefix*(n_channels),
                device=hidden_states.device
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)




        attn_outputs = self.attn(
            hidden_states,
            prompts=prompts,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)




class GPT2Model_prompt(GPT2Model):
    def __init__(self, config, patch_num, c_in, num_prefix):
        super().__init__(config)


        head_dim = config.n_embd // config.n_head
        self.head_dim = head_dim

        self.h = nn.ModuleList([GPT2Block_prompt(config, c_in, head_dim, num_prefix,
                                                 layer_idx=i) for i in range(config.num_hidden_layers)])

        self.c_in = c_in
        self.num_prefix = num_prefix


        # time: self.patch_num
        # d_kv = head_dim
        # head: config.n_head


        # config.num_prefix = 1
        # for block in self.h:
        #     block.num_prefix = num_prefix


        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.shared_prompt_projection_k = nn.Linear(config.hidden_size, head_dim, bias=False)
        self.shared_prompt_projection_q = nn.Linear(config.hidden_size, head_dim, bias=False)  # TODO: should dim be smaller than d_kv?
        self.shared_prompt_projection_v = nn.Linear(config.hidden_size, head_dim, bias=False)

        self.shared_prompt_projection_mha = nn.MultiheadAttention(head_dim, num_heads=4, batch_first=True,
                                                                    kdim=head_dim, vdim=head_dim)

        import torch.nn.functional as F
        self.act = F.tanh

        self.ln2 = nn.LayerNorm(head_dim, eps=config.layer_norm_epsilon)

        self.shared_prompt_projection_linear_key = nn.Linear(patch_num, num_prefix*config.n_head)  # TODO: move this to before attention?
        self.shared_prompt_projection_linear_value = nn.Linear(patch_num, num_prefix*config.n_head)


        self.ln3_k = nn.LayerNorm(head_dim, eps=config.layer_norm_epsilon)
        self.ln3_v = nn.LayerNorm(head_dim, eps=config.layer_norm_epsilon)

        self.dropout = nn.Dropout(0.1)

        # vanilla prompt tuning
        per_layer_dim = config.n_head * head_dim
        total_dim = config.num_hidden_layers * 2 * per_layer_dim
        reparam_dim = 32
        self.prompt_embed = (
            nn.Sequential(
                nn.Embedding(num_prefix*c_in, per_layer_dim),
                nn.Linear(per_layer_dim, reparam_dim),
                nn.Tanh(),
                nn.Linear(reparam_dim, total_dim),
            )
        )

        self.shared_prompt_projection = {
            'k': self.shared_prompt_projection_k,
            'q': self.shared_prompt_projection_q,
            'v': self.shared_prompt_projection_v,
            'mha': self.shared_prompt_projection_mha,
            # 'ln1': self.ln1,
            # 'ln2': self.ln2,
            # 'ln3_k': self.ln3_k,
            # 'ln3_v': self.ln3_v,
            # 'act': self.act,
            'linear_key': self.shared_prompt_projection_linear_key,
            'linear_value': self.shared_prompt_projection_linear_value,
            # 'dropout': self.dropout,
        }

        for block in self.h:
            block.shared_prompt_projection = self.shared_prompt_projection


        self.post_init()



    def forward(self, inputs_embeds=None, **kwargs):
        self.input_tokens = torch.arange(self.num_prefix*self.c_in)

        prefix_key, prefix_value = self.generate_prefix_item(inputs_embeds, self.prompt_embed)
        # kwargs['use_cache'] = False

        for block, k, v, in zip(self.h, prefix_key, prefix_value):
            block.prefix_key = k
            block.prefix_value = v

        output = super().forward(inputs_embeds=inputs_embeds, **kwargs)
        # self.clean_up()

        return output


    def clean_up(self):
        # For safety, in case other code uses it
        for block in self.h:
            del block.prefix_key
            del block.prefix_value


    def generate_prefix_item(self, input_ids, embedding):
        bsz = input_ids.size(0)
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(input_ids.device)
        prefix = embedding(input_tokens)  # batch, seq, layer * embed * 2
        prefix = prefix.view(
            bsz,
            self.num_prefix*self.c_in,
            self.config.num_hidden_layers,
            2,
            self.config.n_head,
            self.head_dim,
        )
        prefix = prefix.permute([3, 2, 0, 4, 1, 5])  # 2, num_layers, bsz, num_heads, num_prefix, d_kv
        return prefix[0], prefix[1]


