# from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers import CLIPModel


from transformers.activations import ACT2FN


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor




class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim=512):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = 8
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = 0.1

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class CLIPMLP(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = hidden_size * 4
        self.activation_fn = ACT2FN["quick_gelu"]
        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = CLIPAttention(embed_dim=embed_dim)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=1e-05)
        self.mlp = CLIPMLP(hidden_size=embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=1e-05)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs




class CLIPEncoder(nn.Module):

    def __init__(self, number_hidden_layers=3, embed_dim=512):
        super().__init__()
        self.num_hidden_layers = number_hidden_layers
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(embed_dim=embed_dim) 
            for _ in range(self.num_hidden_layers)
        ])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple]:



        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):

            layer_outputs = encoder_layer(
                hidden_states,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]



        return tuple(v for v in [hidden_states] if v is not None)
        # if not return_dict:
        #     return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        # return BaseModelOutput(
        #     last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        # )


class CLIPTextTransformer(nn.Module):
    def __init__(self, number_hidden_layers=3, embed_dim=512):
        super().__init__()
        self.encoder = CLIPEncoder(
            number_hidden_layers=number_hidden_layers,
            embed_dim=embed_dim
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=1e-05)



    def forward(
        self,
        input_embeds,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple]:

        encoder_outputs = self.encoder(
            inputs_embeds=input_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        return (last_hidden_state,)




class FusionModel(nn.Module):
    """Transformer-based fusion model for embedding integration"""

    def __init__(self, number_hidden_layers=12, embed_dim=None, use_projection=False):
        super().__init__()
        
        # Add projection layer
        self.use_projection = use_projection
        if self.use_projection and embed_dim != 512:
            self.input_projection = nn.Linear(embed_dim, 512)
            self.projection_dim = 512
            
            # Use same initialization as other linear layers
            factor = embed_dim ** -0.5  # Use input dimension
            nn.init.normal_(self.input_projection.weight, std=factor)
            if self.input_projection.bias is not None:
                nn.init.zeros_(self.input_projection.bias)
        else:
            self.input_projection = None
            self.projection_dim = embed_dim
            
        # Use projected dimension
        self.text_embed_dim = self.projection_dim
        self.logit_scale_init_value = 2.6592

        text_model = CLIPTextModel(
            number_hidden_layers=number_hidden_layers,
            embed_dim=self.projection_dim
        )
        self.text_model = text_model.text_model

        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.logit_scale_init_value))


    def forward(
        self,
        input_embeds,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> torch.FloatTensor:
        # Apply projection if needed
        if self.input_projection is not None:
            input_embeds = self.input_projection(input_embeds)

        text_outputs = self.text_model(
            input_embeds=input_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = text_outputs[0]
        text_features = self.text_projection(last_hidden_state)

        return text_features




class CLIPTextModel(nn.Module):

    def __init__(self, number_hidden_layers=3, embed_dim=512):
        super().__init__()
        self.text_model = CLIPTextTransformer(
            number_hidden_layers=number_hidden_layers,
            embed_dim=embed_dim
        )


    def forward(
        self,
        input_embeds,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple]:


        return self.text_model(
            input_embeds=input_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )



