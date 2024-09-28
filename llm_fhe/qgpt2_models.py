from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
from concrete.fhe.compilation import Circuit, Configuration
from qgpt2_class import QGPT2
from quant_framework import DualArray
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.pytorch_utils import Conv1D
from utility_functions import slice_ordered_dict


class QGPT2Attention(GPT2Attention):
    """Base class for building a torch module for the quantized attention mechanism."""

    def __init__(self, config: GPT2Config):
        """Initialize the base class.

        Args:
            config (GPT2Config): GPT-2's configuration.
        """
        super().__init__(config)

        self.fhe = "disable"
        self.true_float = False

    def set_fhe_mode(self, fhe: str = "disable", true_float: bool = False):
        """Set the FHE mode for the module's forward pass.

        fhe (str): The FHE mode to consider, either "disable", "simulate" or "execute". Default
            to "disable".
        true_float (bool): If the FHE mode is set to "disable", indicate if the operations
            should be in floating points instead of being quantized. Default to False.
        """
        assert fhe in [
            "disable",
            "simulate",
            "execute",
        ], "Parameter 'fhe' can only be 'disable', 'simulate' or 'execute'."

        self.fhe = fhe
        self.true_float = true_float


class QGPT2LMHeadModel(GPT2LMHeadModel):
    """Base class for integrating quantized operations within GPT2LMHeadModel's forward pass."""

    def __init__(
        self,
        config: GPT2Config,
        n_bits: int,
        attention_module: Union[QGPT2SingleHeadAttention, QGPT2MultiHeadsAttention],
        layer: int = 0,
        num_fhe: int = 1
    ):
        """Initialize the base class.

        This class essentially overwrites GPT-2's attention module found in the layer whose index is
        given with the given quantized module.

        Args:
            config (GPT2Config): GPT-2's configuration.
            n_bits (int): The number of bits to use for quantizing the inputs, weights and
                activations.
            attention (Union[QGPT2SingleHeadAttention, QGPT2MultiHeadsAttention]): The quantized attention module
                to consider.
            layer (int): The index representing the GPT-2 layer to consider. Default to 0.
        """
        assert 0 <= layer <= 11, f"The GPT-2 model only has 12 layers, but got {layer}"

        super().__init__(config)

        self.transformer.h[layer].attn = attention_module(config, n_bits=n_bits, layer=layer, num_fhe=num_fhe)

    @property
    def q_attention(self) -> GPT2Attention:
        """Get GPT-2's attention module found in the first layer.

        Returns:
            GPT2Attention: The attention module.
        """
        return self.transformer.h[0].attn

    def set_fhe_mode(self, fhe: str = "disable", true_float: bool = False):
        """Set the FHE mode for the module's forward pass.

        fhe (str): The FHE mode to consider, either "disable", "simulate" or "execute". Default
            to "disable".
        true_float (bool): If the FHE mode is set to "disable", indicate if the operations
            should be in floating points instead of being quantized. Default to False.
        """
        self.q_attention.set_fhe_mode(fhe=fhe, true_float=true_float)

    def compile(
        self,
        inputset_ids: torch.Tensor,
        configuration: Optional[Configuration] = None,
        msbs_round: Optional[int] = None,
        rounding_kwargs: Optional[Dict] = None,
    ) -> Circuit:
        """Compile the model using the stored calibration data.

        Args:
            inputset_ids (torch.Tensor): The token ids to consider as an inputset.
            configuration (Optional[Configuration]): The configuration to use during compilation.
                Default to None.
            msbs_round (Optional[int]): msbs to keep after rounding
            rounding_kwargs (Optional[Dict]): optional keyword arguments of `InsertRounding`

        Returns:
            Circuit: The underlying FHE circuit.
        """

        # Disable the FHE execution, as the following forward pass should be made in the clear along
        # floating point values. This is done in order to properly calibrate and store the
        # quantization parameters such as the scale and zero points
        self.set_fhe_mode(fhe="disable", true_float=False)

        # Execute a full pass in the clear
        self.forward(inputset_ids, use_cache=False)

        # Compile the attention module using stored calibration data (made of intermediary hidden
        # states)
        return self.q_attention.q_module.compile(
            configuration=configuration, msbs_round=msbs_round, rounding_kwargs=rounding_kwargs
        )


class SingleHeadAttention(QGPT2):
    """Class representing a single attention head implemented with quantization methods.

    The first projections (represented by the "c_attn" weights) is also done with quantized methods.
    In order to properly achieve this, the inputs are expected to have the shape
    (n_batch, 1, n_seq, head_dim) while the weights are extracted with proper shapes at the right
    indices.
    """

    def __init__(self, n_bits: int, layer, n_bits_weights: Optional[int] = None):
        super().__init__(n_bits, layer=layer, n_bits_weights=n_bits_weights)

        # Extract the embedding and the head dimensions, which are respectively 768 and 64 for the
        # GPT-2 model (which uses 12 heads)
        self.n_embd = self.config.n_embd
        self.head_dim = self.config.n_embd // self.config.n_head

    def run_numpy(self, q_inputs: np.ndarray) -> Union[np.ndarray, DualArray]:
        """Run the quantized operators that will be converted to FHE.

        Args:
            q_inputs (np.ndarray): The quantized inputs.

        Returns:
            Union[np.ndarray, DualArray]: The quantized outputs.
        """

        # Convert the input to a DualArray instance using the stored calibration data
        # q_x has shape (n_batch, n_seq, n_embed)
        q_x = DualArray(float_array=self.x_calib, int_array=q_inputs, quantizer=self.quantizer)

        # Extract the attention base module name
        mha_module_name = f"transformer.h.{self.layer}.attn."

        # Extract the query, key and value weight and bias values using the proper indices
        head_0_indices = [
            list(range(i * self.n_embd, i * self.n_embd + self.head_dim)) for i in range(3)
        ]
        q_qkv_weights = self.q_weights[mha_module_name + "c_attn.weight"].slice_array(
            axis=-1, indices=head_0_indices, key=f"slice_qkv_weights_layer_{self.layer}"
        )
        q_qkv_bias = self.q_weights[mha_module_name + "c_attn.bias"].slice_array(
            axis=-1, indices=head_0_indices, key=f"slice_qkv_bias_layer_{self.layer}"
        )

        # Apply the first projection in order to extract Q, K and V as a single array
        # q_qkv has shape (n_batch, n_seq, 3*head_dim)
        q_qkv = q_x.linear(
            weight=q_qkv_weights,
            bias=q_qkv_bias,
            key=f"attention_qkv_proj_layer_{self.layer}",
        )

        # Reshape q_qkv in order to indicate to the attention that we only consider a single head
        # here, meaning the shape is now (n_batch, 1, n_seq, 3*head_dim)
        q_qkv = q_qkv.expand_dims(axis=1, key=f"unsqueeze_{self.layer}")

        # Extract Q, K and V, with shapes (n_batch, 1, n_seq, head_dim)
        q_q, q_k, q_v = q_qkv.enc_split(3, axis=-1, key=f"qkv_split_layer_{self.layer}")

        # Apply the attention mechanism
        q_y = self.attention(q_q, q_k, q_v)

        return self.finalize(q_y)

class QGPT2SingleHeadAttentionSimulationHybrid(QGPT2Attention):
    """Torch module that rewrites GPT-2's attention with quantized operations on a single head."""

    def __init__(self, config: GPT2Config, layer: int, n_bits: int = 16, num_fhe: int=1):
        super().__init__(config)

        # Instantiate the quantized module used for the attention mechanism
        self.q_module = SingleHeadAttention(n_bits=n_bits, layer=layer)

        # Define the number of head to consider with quantized operators
        self.n_qhead = 1

        # Define the new 1D convolution operator with proper shapes, knowing that a single head
        # is done with the quantized module and only 11 heads should be considered in float
        self.float_embed_dim = self.embed_dim - self.n_qhead * self.head_dim
        self.c_attn_1_11 = Conv1D(3 * self.float_embed_dim, self.embed_dim)
        self.split_size = self.float_embed_dim
        self.num_fhe = num_fhe

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        """GPT-2's multi-head attention pass with a single head made for FHE computations.

        The initial implementation can be found in huggingFace's GPT2Attention class.
        """
        if encoder_hidden_states is not None:
            raise ValueError(
                "Class cannot be used as cross attention, please make sure to not instantiate "
                "class with `GPT2Attention(..., is_cross_attention=True)`."
            )

        # Compute Q, K, V in the clear for head 1 up to 11
        query, key, value = self.c_attn_1_11(hidden_states).split(self.split_size, dim=2)

        # Split them into 11 heads
        query = self._split_heads(query, self.num_heads - self.n_qhead, self.head_dim)
        key = self._split_heads(key, self.num_heads - self.n_qhead, self.head_dim)
        value = self._split_heads(value, self.num_heads - self.n_qhead, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            raise ValueError("Method 'reorder_and_upcast_attn' is not implemented")

        query = query[:self.num_fhe, :, :]
        # Apply the multi-head attention mechanism in the clear onto the 11 heads
        attn_output_1_11, _ = self._attn(query, key, value, attention_mask, head_mask)

        # Apply the attention on the first head using FHE-compliant operators
        attn_output_0 = self.q_module.run_torch(
            hidden_states,
            fhe=self.fhe,
            true_float=self.true_float,
        )

        # Concatenate the heads together
        attn_output = torch.cat((attn_output_0, attn_output_1_11), dim=1)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return (attn_output, present)

class QGPT2SingleHeadAttention(QGPT2Attention):
    """Torch module that rewrites GPT-2's attention with quantized operations on a single head."""

    def __init__(self, config: GPT2Config, layer: int, n_bits: int = 16, num_fhe: int=1):
        super().__init__(config)

        # Instantiate the quantized module used for the attention mechanism
        self.q_module = SingleHeadAttention(n_bits=n_bits, layer=layer, num_fhe=num_fhe)

        # Define the number of head to consider with quantized operators
        self.n_qhead = 1

        # Define the new 1D convolution operator with proper shapes, knowing that a single head
        # is done with the quantized module and only 11 heads should be considered in float
        self.float_embed_dim = self.embed_dim - self.n_qhead * self.head_dim
        self.c_attn_1_11 = Conv1D(3 * self.float_embed_dim, self.embed_dim)
        self.split_size = self.float_embed_dim

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
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        """GPT-2's multi-head attention pass with a single head made for FHE computations.

        The initial implementation can be found in huggingFace's GPT2Attention class.
        """
        if encoder_hidden_states is not None:
            raise ValueError(
                "Class cannot be used as cross attention, please make sure to not instantiate "
                "class with `GPT2Attention(..., is_cross_attention=True)`."
            )

        # Compute Q, K, V in the clear for head 1 up to 11
        query, key, value = self.c_attn_1_11(hidden_states).split(self.split_size, dim=2)

        # Split them into 11 heads
        query = self._split_heads(query, self.num_heads - self.n_qhead, self.head_dim)
        key = self._split_heads(key, self.num_heads - self.n_qhead, self.head_dim)
        value = self._split_heads(value, self.num_heads - self.n_qhead, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            raise ValueError("Method 'reorder_and_upcast_attn' is not implemented")

        # Apply the multi-head attention mechanism in the clear onto the 11 heads
        attn_output_1_11, _ = self._attn(query, key, value, attention_mask, head_mask)

        # Apply the attention on the first head using FHE-compliant operators
        attn_output_0 = self.q_module.run_torch(
            hidden_states,
            fhe=self.fhe,
            true_float=self.true_float,
        )

        # Concatenate the heads together
        attn_output = torch.cat((attn_output_0, attn_output_1_11), dim=1)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return (attn_output, present)

class SingleHeadQGPT2Model(QGPT2LMHeadModel):
    """QGPT2LMHeadModel implementation with a single attention head can be executed in FHE."""

    def __init__(self, config: GPT2Config, n_bits: int = 16, layer: int = 0, num_fhe = 1):
        super().__init__(
            config, n_bits=n_bits, num_fhe=num_fhe, attention_module=QGPT2SingleHeadAttention, layer=layer
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs
    ):
        """Load the model from pre-trained files and manually load the new Conv1D module's weights.

        The convolution module must be manually loaded with the proper weights, representing the 11
        heads instead of the usual 12 ones, since it does not exist in Hugging Face's initial
        implementation of GPT-2.
        """

        model = super().from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, *model_args, **kwargs
        )

        # Retrieve the proper attention weights
        c_attn_params = model.q_attention.c_attn.state_dict()

        # Extract the proper indices and shapes
        n_embd = model.config.n_embd
        head_dim = model.config.n_embd // model.config.n_head
        head_1_11_indices = [list(range(i * n_embd + head_dim, (i + 1) * n_embd)) for i in range(3)]

        c_attn_params_1_11 = slice_ordered_dict(c_attn_params, dim=-1, indices=head_1_11_indices)

        # Load the weights into the new convolution module
        c_attn_params = model.q_attention.c_attn_1_11.load_state_dict(c_attn_params_1_11)

        return model


class SingleHeadQGPT2ModelSimulationHybrid(QGPT2LMHeadModel):
    """QGPT2LMHeadModel implementation with a single attention head can be executed in FHE."""

    def __init__(self, config: GPT2Config, n_bits: int = 16, layer: int = 0, num_fhe: int=1):
        super().__init__(
            config, n_bits=n_bits, num_fhe=num_fhe, attention_module=QGPT2SingleHeadAttentionSimulationHybrid, layer=layer
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs
    ):
        """Load the model from pre-trained files and manually load the new Conv1D module's weights.

        The convolution module must be manually loaded with the proper weights, representing the 11
        heads instead of the usual 12 ones, since it does not exist in Hugging Face's initial
        implementation of GPT-2.
        """

        model = super().from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, *model_args, **kwargs
        )

        # Retrieve the proper attention weights
        c_attn_params = model.q_attention.c_attn.state_dict()

        # Extract the proper indices and shapes
        n_embd = model.config.n_embd
        head_dim = model.config.n_embd // model.config.n_head
        head_1_11_indices = [list(range(i * n_embd + head_dim, (i + 1) * n_embd)) for i in range(3)]

        c_attn_params_1_11 = slice_ordered_dict(c_attn_params, dim=-1, indices=head_1_11_indices)

        # Load the weights into the new convolution module
        c_attn_params = model.q_attention.c_attn_1_11.load_state_dict(c_attn_params_1_11)

        return model
