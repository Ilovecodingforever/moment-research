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
    def __init__(self, config, c_in, head_dim, num_prefix, multivariate_projection, agg, layer_idx=None):
        super().__init__(config)
        self.c_in = c_in
        self.num_prefix = num_prefix
        self.head_dim = head_dim
        self.config = config
        self.multivariate_projection = multivariate_projection
        self.agg = agg


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


        if self.multivariate_projection == 'attention':

            n_channels = self.c_in
            batch_size, seq_length, d_model = hidden_states.shape
            batch_size_real = batch_size // n_channels

            hidden_states_ = hidden_states.reshape(-1, n_channels, seq_length, d_model)
            hidden_states_proj = hidden_states_.transpose(1, 2).reshape(-1, n_channels, d_model)
            # hidden_states_proj = self.ln_1(hidden_states_proj)
            attn_output, attn_output_weights = self.shared_prompt_projection['mha'](self.shared_prompt_projection['q'](hidden_states_proj),
                                                                                    self.shared_prompt_projection['k'](hidden_states_proj),
                                                                                    self.shared_prompt_projection['v'](hidden_states_proj))
            
            
            # attn_output = self.shared_prompt_projection['mha_proj'](attn_output)
            
            # attn_output = self.shared_prompt_projection['ln2'](self.shared_prompt_projection['act'](attn_output)).reshape(batch_size_real, seq_length, n_channels, -1).permute(0, 2, 3, 1)
            attn_output = (attn_output).reshape(batch_size_real, seq_length, n_channels, -1).permute(0, 2, 3, 1)

            # shared_prompt_projection_k = self.shared_prompt_projection['act'](self.shared_prompt_projection['linear_key'](attn_output)) # bs x channel x d_kv x (num_prefix*n_heads)
            # shared_prompt_projection_v = self.shared_prompt_projection['act'](self.shared_prompt_projection['linear_value'](attn_output))
            
            if self.agg == 'mlp':
                shared_prompt_projection_k = self.shared_prompt_projection['agg_key'](attn_output) # bs x channel x d_kv x (num_prefix*n_heads)
                shared_prompt_projection_v = self.shared_prompt_projection['agg_value'](attn_output)

            elif self.agg == 'rnn':
                shared_prompt_projection_k = self.shared_prompt_projection['agg_key'](attn_output.reshape(batch_size, seq_length, -1))[0][:, -1, :].reshape(batch_size_real, n_channels, -1)
                shared_prompt_projection_v = self.shared_prompt_projection['agg_value'](attn_output.reshape(batch_size, seq_length, -1))[0][:, -1, :].reshape(batch_size_real, n_channels, -1)

            shared_prompt_projection_key = shared_prompt_projection_k.reshape(batch_size_real, n_channels, -1, self.num_prefix, self.config.n_head).permute(0, 4, 1, 3, 2).reshape(batch_size_real, self.config.n_head, n_channels*self.num_prefix, -1).repeat_interleave(n_channels, dim=0)
            shared_prompt_projection_value = shared_prompt_projection_v.reshape(batch_size_real, n_channels, -1, self.num_prefix, self.config.n_head).permute(0, 4, 1, 3, 2).reshape(batch_size_real, self.config.n_head, n_channels*self.num_prefix, -1).repeat_interleave(n_channels, dim=0)

            # residual connection
            # shared_prompt_projection_key = self.prefix_key + shared_prompt_projection_key
            # shared_prompt_projection_value = self.prefix_value + shared_prompt_projection_value
            
            # shared_prompt_projection_key = torch.cat([prefix_key, shared_prompt_projection_key], dim=-2)
            # shared_prompt_projection_value = torch.cat([prefix_value, shared_prompt_projection_value], dim=-2)

            # TODO: maybe should ln here?
            # TODO: different ln for key and value?
            # shared_prompt_projection_key = self.shared_prompt_projection['dropout'](shared_prompt_projection_key)
            # shared_prompt_projection_value = self.shared_prompt_projection['dropout'](shared_prompt_projection_value)
            # shared_prompt_projection_key = self.shared_prompt_projection['ln3_k'](shared_prompt_projection_key)
            # shared_prompt_projection_value = self.shared_prompt_projection['ln3_v'](shared_prompt_projection_value)
        
        elif self.multivariate_projection == 'vanilla':
            shared_prompt_projection_key = self.prefix_key
            shared_prompt_projection_value = self.prefix_value
            
        else:
            raise ValueError(f"multivariate_projection should be either 'attention' or 'vanilla'")

        assert layer_past is None
        layer_past = (shared_prompt_projection_key, shared_prompt_projection_value)


        if attention_mask is not None:
            raise NotImplementedError("attention_mask not implemented")
            prefix_mask = torch.zeros(
                batch_size, 1, attention_mask.size(2), self.num_prefix*(n_channels),
                device=hidden_states.device
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)



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
    def __init__(self, config, patch_num, c_in, num_prefix, multivariate_projection, agg):
        super().__init__(config)


        head_dim = config.n_embd // config.n_head
        self.head_dim = head_dim
        self.multivariate_projection = multivariate_projection
        self.agg = agg

        self.h = nn.ModuleList([GPT2Block_prompt(config, c_in, head_dim, num_prefix, multivariate_projection, agg,
                                                 layer_idx=i) for i in range(config.num_hidden_layers)])

        self.c_in = c_in
        self.num_prefix = num_prefix


        # time: self.patch_num
        # d_kv = head_dim
        # head: config.n_head



        if self.multivariate_projection == 'attention':

            # reparam_dim = 1048
            reparam_dim = head_dim

            # shared_prompt_projection_list = []

            # for block in self.h:

            # self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

            self.shared_prompt_projection_k = nn.Linear(config.hidden_size, head_dim, bias=False)
            self.shared_prompt_projection_q = nn.Linear(config.hidden_size, reparam_dim, bias=False)  # TODO: should dim be smaller than d_kv?
            self.shared_prompt_projection_v = nn.Linear(config.hidden_size, head_dim, bias=False)

            self.shared_prompt_projection_mha = nn.MultiheadAttention(reparam_dim, num_heads=4, batch_first=True,
                                                                        kdim=head_dim, vdim=head_dim)


            # self.mha_proj = nn.Linear(reparam_dim, head_dim)


            import torch.nn.functional as F
            # self.act = F.tanh

            # self.ln2 = nn.LayerNorm(head_dim, eps=config.layer_norm_epsilon)


            if self.agg == 'mlp':
                self.shared_prompt_projection_agg_key = nn.Linear(patch_num, num_prefix*config.n_head)  # TODO: move this to before attention?
                self.shared_prompt_projection_agg_value = nn.Linear(patch_num, num_prefix*config.n_head)                
            elif self.agg == 'rnn':
                # TODO: does the dimensions make sense? input is less than output
                self.shared_prompt_projection_agg_key = nn.RNN(reparam_dim, reparam_dim*num_prefix*config.n_head, batch_first=True)
                self.shared_prompt_projection_agg_value = nn.RNN(reparam_dim, reparam_dim*num_prefix*config.n_head, batch_first=True)
            else:
                raise ValueError('Invalid aggregation type')



            # self.ln3_k = nn.LayerNorm(head_dim, eps=config.layer_norm_epsilon)
            # self.ln3_v = nn.LayerNorm(head_dim, eps=config.layer_norm_epsilon)

            # self.dropout = nn.Dropout(0.1)


            self.shared_prompt_projection = torch.nn.ModuleDict({
                'k': self.shared_prompt_projection_k,
                'q': self.shared_prompt_projection_q,
                'v': self.shared_prompt_projection_v,
                'mha': self.shared_prompt_projection_mha,
                # 'ln1': self.ln1,
                # 'ln2': self.ln2,
                # 'ln3_k': self.ln3_k,
                # 'ln3_v': self.ln3_v,
                # 'act': self.act,
                # 'mha_proj': self.mha_proj,
                'agg_key': self.shared_prompt_projection_agg_key,
                'agg_value': self.shared_prompt_projection_agg_value,
                # 'dropout': self.dropout,
            })
            
            
        # shared_prompt_projection_list.append(shared_prompt_projection)
            
        # self.shared_prompt_projection_list = nn.ModuleList(shared_prompt_projection_list)
            

            for i, block in enumerate(self.h):
                block.shared_prompt_projection = self.shared_prompt_projection


        elif self.multivariate_projection == 'vanilla':
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
        else:
            raise ValueError(f"multivariate_projection should be either 'attention' or 'vanilla'")  

        self.post_init()



    def forward(self, inputs_embeds=None, **kwargs):
        if self.multivariate_projection == 'vanilla':
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


