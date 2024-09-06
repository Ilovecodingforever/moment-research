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



class GPT2Block_prompt(GPT2Block):
    def __init__(self, config, c_in, layer_idx=None):
        super().__init__(config)
        # hidden_size = config.hidden_size
        # inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        # self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # self.attn = GPT2Attention(config, layer_idx=layer_idx)
        # self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # if config.add_cross_attention:
        #     self.crossattention = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
        #     self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # self.mlp = GPT2MLP(inner_dim, config)

        self.config = config
        self.c_in = c_in

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




        n_channels = self.c_in
        batch_size, seq_length, d_model = hidden_states.shape
        batch_size_real = batch_size // n_channels

        hidden_states_ = hidden_states.reshape(-1, n_channels, seq_length, d_model)
        hidden_states_proj = hidden_states_.transpose(1, 2).reshape(-1, n_channels, d_model)
        attn_output, attn_output_weights = self.shared_prompt_projection['mha'](self.shared_prompt_projection['q'](hidden_states_proj),
                                                                                self.shared_prompt_projection['k'](hidden_states_proj),
                                                                                self.shared_prompt_projection['v'](hidden_states_proj))
        attn_output = attn_output.reshape(batch_size_real, seq_length, n_channels, -1).permute(0, 2, 3, 1)

        shared_prompt_projection_k = self.shared_prompt_projection['linear_key'](attn_output) # bs x channel x d_kv x (num_prefix*n_heads)
        shared_prompt_projection_key = shared_prompt_projection_k.reshape(batch_size_real, n_channels, -1, self.num_prefix, self.config.n_head).permute(0, 4, 1, 3, 2).reshape(batch_size_real, self.config.n_head, n_channels*self.num_prefix, -1).repeat_interleave(n_channels, dim=0)
        shared_prompt_projection_v = self.shared_prompt_projection['linear_value'](attn_output)
        shared_prompt_projection_value = shared_prompt_projection_v.reshape(batch_size_real, n_channels, -1, self.num_prefix, self.config.n_head).permute(0, 4, 1, 3, 2).reshape(batch_size_real, self.config.n_head, n_channels*self.num_prefix, -1).repeat_interleave(n_channels, dim=0)

        assert layer_past is None
        layer_past = (shared_prompt_projection_key, shared_prompt_projection_value)

        if attention_mask is not None:
            raise NotImplementedError("attention_mask not implemented")
            prefix_mask = torch.zeros(
                batch_size, 1, attention_mask.size(2), self.num_prefix*(n_channels),
                device=hidden_states.device
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)


        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
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
        self.h = nn.ModuleList([GPT2Block_prompt(config, c_in,
                                                 layer_idx=i) for i in range(config.num_hidden_layers)])
        self.post_init()

        # config.num_prefix = 1
        for block in self.h:
            block.num_prefix = num_prefix

        # time: self.patch_num
        # d_kv = head_dim
        # head: config.n_head

        head_dim = config.n_embd // config.n_head

        self.shared_prompt_projection_k = nn.Linear(config.hidden_size, head_dim, bias=False)
        self.shared_prompt_projection_q = nn.Linear(config.hidden_size, head_dim, bias=False)  # TODO: should dim be smaller than d_kv?
        self.shared_prompt_projection_v = nn.Linear(config.hidden_size, head_dim, bias=False)

        self.shared_prompt_projection_mha = nn.MultiheadAttention(head_dim, num_heads=1, batch_first=True,
                                                                    kdim=head_dim, vdim=head_dim)

        self.shared_prompt_projection_linear_key = nn.Linear(patch_num, num_prefix*config.n_head)  # TODO: move this to before attention?
        self.shared_prompt_projection_linear_value = nn.Linear(patch_num, num_prefix*config.n_head)

        self.shared_prompt_projection = {
            'k': self.shared_prompt_projection_k,
            'q': self.shared_prompt_projection_q,
            'v': self.shared_prompt_projection_v,
            'mha': self.shared_prompt_projection_mha,
            'linear_key': self.shared_prompt_projection_linear_key,
            'linear_value': self.shared_prompt_projection_linear_value,
        }


        for block in self.h:
            block.shared_prompt_projection = self.shared_prompt_projection



    def forward(self, **kwargs):
        return super().forward(**kwargs)



