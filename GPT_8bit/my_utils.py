import torch
import time
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn import Linear as _linear
from torch.nn import Embedding as _Embedding
from torch.nn import Module, Parameter
import decimal
from decimal import Decimal
from torch.autograd import Function


import logging

logger = logging.getLogger(__name__)

class QuantLinear(Module):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 weight_bit,
                 bias_bit=None,
                 per_channel=False,
                 quant_mode='none'):
        super(QuantLinear, self).__init__()
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.quant_mode = quant_mode
        self.percentile_mode = False

        self.weight_function = SymmetricQuantFunction.apply

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = Parameter(linear.weight.data.clone())
        self.register_buffer('fc_scaling_factor', torch.zeros(self.out_features))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        try:
            self.bias = Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None
        self.register_buffer('bias_integer', torch.zeros_like(self.bias))


    def forward(self, x, prev_act_scaling_factor=None):
        """
        using quantized weights to forward activation x
        """
        if self.quant_mode == 'none':
            return F.linear(x, weight=self.weight, bias=self.bias), None
      
        # Detaching the weight tensor for transformations
        w_transform = self.weight.data.detach()

        # Calculating min and max for weight scaling
        if self.per_channel:
            w_min, a = torch.min(w_transform, dim=1, out=None)
            w_max, b = torch.max(w_transform, dim=1, out=None)
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)

        # Compute scaling factor for fully connected layer
        self.fc_scaling_factor = symmetric_linear_quantization_params(
            self.weight_bit, w_min, w_max, self.per_channel)

        # Quantize weights
        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.percentile_mode, self.fc_scaling_factor)

        # Compute bias scaling factor
        bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

        # Quantize biases
        self.bias_integer = self.weight_function(
            self.bias, self.bias_bit, False, bias_scaling_factor)

        # Adjusting the scaling factor for input
        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)

        # Quantizing input
        x_int = x / prev_act_scaling_factor

        # Perform linear transformation with quantized weights, biases, and inputs
        output = F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer) * bias_scaling_factor

        # Returning output and the corresponding scaling factor
        return output, bias_scaling_factor


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)



class round_ste(Function):
    """
    Straight-through Estimator(STE) for torch.round()
    """
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
        
class IntLayerNorm(Module):
    """
    Class to quantize given LayerNorm layer
    """
    def __init__(self,
                 output_bit,
                 overflow_handling=True,
                 quant_mode='symmetric',
                 force_dequant='none'):
        super(IntLayerNorm, self).__init__()
        self.quant_mode = quant_mode
        if force_dequant in ['nonlinear', 'layernorm']:
            self.quant_mode = 'none'
        self.register_buffer('shift', torch.zeros(1))
        self.output_bit = output_bit
        self.dim_sqrt = None
        self.activation = QuantAct(output_bit, quant_mode=self.quant_mode)
        self.weight_function = SymmetricQuantFunction.apply

    def set_param(self, ln):
        self.normalized_shape = ln.normalized_shape
        self.eps = ln.eps
        self.weight = Parameter(ln.weight.data.clone())
        self.bias = Parameter(ln.bias.data.clone())

    def forward(self, x, scaling_factor=None,quant_mode ='symmetric', exponents=None):
       
        if self.dim_sqrt is None:
            n = torch.tensor(x.shape[2], dtype=torch.float)  # Assuming 768 as feature dimension
            self.dim_sqrt = torch.sqrt(n).cuda()

        # Quantize input
        x_int = x / scaling_factor

        # Compute the mean and subtract it from the input
        mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
        y_int = x_int - mean_int

        # Shift to avoid overflow
        y_int_shifted = floor_ste.apply(y_int / 2 ** self.shift)

        # Compute the squared values
        y_sq_int = y_int_shifted ** 2

        # Calculate variance
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

        # Compute standard deviation
        std_int = floor_ste.apply(torch.sqrt(var_int)) * 2 ** self.shift

        # Adjust factor for normalization
        factor = floor_ste.apply(2**31 / std_int)
        y_int = floor_ste.apply(y_int * factor / 2)

        # Adjust scaling factor
        scaling_factor = self.dim_sqrt / 2**30

        # Calculate and add the quantized bias
        bias = self.bias.data.detach() / (self.weight.data.detach())
        bias_int = floor_ste.apply(bias / scaling_factor)
        y_int += bias_int

        # Update scaling factor and apply to the adjusted input
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor

        return x, scaling_factor

        

def batch_frexp(inputs, max_bit=31):
    """
    Decompose the scaling factor into mantissa and twos exponent.

    Parameters:
    ----------
    inputs: scaling factor
    return: (mantissa, exponent)
    """

    shape_of_input = inputs.size()

    # trans the input to be a 1-d tensor
    inputs = inputs.view(-1)
    
    output_m, output_e = np.frexp(inputs.cpu().numpy())
    tmp_m = []
    for m in output_m:
        int_m_shifted = int(Decimal(m * (2**max_bit)).quantize(Decimal('1'), 
            rounding=decimal.ROUND_HALF_UP))
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)

    output_e = float(max_bit) - output_e

    return torch.from_numpy( output_m ).cuda().view(shape_of_input), \
           torch.from_numpy( output_e ).cuda().view(shape_of_input)

class fixedpoint_mul(Function):
    """
    Function to perform fixed-point arthmetic that can match integer arthmetic on hardware.
    """
    @ staticmethod
    def forward (ctx, pre_act, pre_act_scaling_factor, 
                 bit_num, quant_mode, z_scaling_factor, 
                 identity=None, identity_scaling_factor=None):

        if len(pre_act_scaling_factor.shape) == 3:
            reshape = lambda x : x
        else:
            reshape = lambda x : x.view(1, 1, -1)
        ctx.identity = identity

        if quant_mode == 'symmetric':
            n = 2 ** (bit_num - 1) - 1
        else:
            n = 2 ** bit_num - 1

        with torch.no_grad():
            pre_act_scaling_factor = reshape(pre_act_scaling_factor)
            if identity is not None:
                identity_scaling_factor = reshape(identity_scaling_factor)

            ctx.z_scaling_factor = z_scaling_factor
            
            z_int = torch.round(pre_act / pre_act_scaling_factor) 
            _A = pre_act_scaling_factor.type(torch.double)
            _B = (z_scaling_factor.type(torch.float)).type(torch.double)
            new_scale = _A / _B
            new_scale = reshape(new_scale)

            m, e = batch_frexp(new_scale)

            output = z_int.type(torch.double) * m.type(torch.double)
            output = torch.round( output / (2.0**e) )

            if identity is not None:
                # needs addition of identity activation
                wx_int = torch.round(identity / identity_scaling_factor)

                _A = identity_scaling_factor.type(torch.double)
                _B = (z_scaling_factor.type(torch.float)).type(torch.double)
                new_scale = _A / _B
                new_scale = reshape(new_scale)

                m1, e1 = batch_frexp(new_scale)
                output1 = wx_int.type(torch.double) * m1.type(torch.double)
                output1 = torch.round(output1 / (2.0**e1))

                output = output1 + output

            if bit_num in [4, 8, 16]:
                if quant_mode == 'symmetric':
                    return torch.clamp( output.type(torch.float), -n - 1, n)
                else:
                    return torch.clamp( output.type(torch.float), 0, n)
            else:
                return output.type(torch.float)

    @ staticmethod
    def backward(ctx, grad_output):
        identity_grad = None
        if ctx.identity is not None:
            identity_grad = grad_output.clone() / ctx.z_scaling_factor
        return grad_output.clone() / ctx.z_scaling_factor, None, None, None, None,\
                identity_grad, None

def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        scale = scale.view(-1)
        zero_point = zero_point.view(-1)
    if inplace:
        input.mul_(1. / scale).add_(zero_point).round_()
        return input
    return torch.round(1. / scale * input + zero_point)

def symmetric_linear_quantization_params(num_bits,
                                        saturation_min,
                                        saturation_max,
                                        per_channel=False):
    """
    Compute the scaling factor with the given quantization range for symmetric quantization.
    """
    # in this part, we do not need any gradient computation,
    # in order to enfore this, we put torch.no_grad()
    with torch.no_grad():
        n = 2 ** (num_bits - 1) - 1

        if per_channel:
            scale, _ = torch.max(torch.stack([saturation_min.abs(), saturation_max.abs()], dim=1), dim=1)
            scale = torch.clamp(scale, min=1e-8) / n 

        else:
            scale = max(saturation_min.abs(), saturation_max.abs())
            scale = torch.clamp(scale, min=1e-8) / n 

    return scale 

class SymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """
    @staticmethod
    def forward(ctx, x, k, percentile_mode=False, specified_scale=None):
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        Note that the current implementation of SymmetricQuantFunction requires pre-calculated scaling factor.
        specified_scale: pre-calculated scaling factor for the tensor x
        """
        
        if specified_scale is not None:
            scale = specified_scale

        zero_point = torch.tensor(0.).cuda()

        n = 2 ** (k - 1) - 1
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        new_quant_x = torch.clamp(new_quant_x, -n, n-1)

        ctx.scale = scale 
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):

        scale = ctx.scale
        if len(grad_output.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
        # reshape scale and zeropoint for linear weights
        elif len(grad_output.shape) == 2:
            scale = scale.view(-1, 1)
        else:
            scale = scale.view(-1)

        return grad_output.clone() / scale, None, None, None, None

def make_positions(tensor, padding_idx = 0, onnx_trace: bool = False):
    """Replace non-padding symbols with their position numbers.
    """

    padding_idx = 0
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

class QuantEmbedding(Module):
    """
    Class to quantize given Embedding layer
    """
    def __init__(self,
                 weight_bit,
                 is_positional=False,
                 momentum=0.95,
                 quant_mode='none'):
        super(QuantEmbedding, self).__init__()

        self.weight_bit = weight_bit
        self.momentum = momentum
        self.quant_mode = quant_mode
        self.per_channel = False
        self.percentile_mode = False
        self.is_positional = is_positional
        self.weight_function = SymmetricQuantFunction.apply
  
    def set_param(self, embedding):
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.padding_idx = embedding.padding_idx
        self.max_norm = embedding.max_norm
        self.norm_type = embedding.norm_type
        self.scale_grad_by_freq = embedding.scale_grad_by_freq
        self.sparse = embedding.sparse
        self.weight = embedding.weight

        if not self.per_channel:
            dim_scaling_factor = 1
        else:
            dim_scaling_factor = self.embedding_dim
        self.register_buffer('weight_scaling_factor', torch.zeros(dim_scaling_factor))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))

        if self.is_positional:
            if self.padding_idx is not None:
                self.max_positions = self.num_embeddings - self.padding_idx - 1
            else:
                self.max_positions = self.num_embeddings


    def forward(self, x, positions=None, incremental_state=None):
        if self.quant_mode == 'none':
            return F.embedding(
                x,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            ), None


        # Detaching the weight tensor for transformations
        w_transform = self.weight.data.detach()

        # Computing the min and max values for weight scaling
        if self.per_channel:
            w_min, _ = torch.min(w_transform, dim=0, keepdim=True)
            w_max, _ = torch.max(w_transform, dim=0, keepdim=True)
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)

        # Calculating the scaling factor for the weights
        self.weight_scaling_factor = symmetric_linear_quantization_params(
            self.weight_bit, w_min, w_max, self.per_channel)

        # Quantizing the weights
        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.percentile_mode, self.weight_scaling_factor)

        # Positional embedding handling
        if self.is_positional:
            # Generating positions if not provided
            if positions is None:
                if incremental_state is not None:
                    # Positions for single step decoding in incremental state
                    positions = torch.zeros((1, 1), device=x.device, dtype=x.dtype).fill_(
                        int(self.padding_idx + x.size(1)))
                else:
                    # Generating positions for input 'x'
                    positions = make_positions(x, self.padding_idx, onnx_trace=False)
                x = positions

        # Applying the embedding operation
        emb_int = F.embedding(
            x,
            self.weight_integer,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        # Returning the quantized embeddings and the corresponding scaling factor
        return emb_int * self.weight_scaling_factor, self.weight_scaling_factor

class QuantAct(Module):
    """
    Class to quantize given activations
    """
    def __init__(self,
                 activation_bit,
                 act_range_momentum=0.95,
                 running_stat=True,
                 per_channel=False,
                 channel_len=None,
                 quant_mode="none"):
        super(QuantAct, self).__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.percentile = False

        if not per_channel:
            self.register_buffer('x_min', torch.zeros(1))
            self.register_buffer('x_max', torch.zeros(1))
            self.register_buffer('act_scaling_factor', torch.zeros(1))
        else:
            self.register_buffer('x_min', torch.zeros(channel_len))
            self.register_buffer('x_max', torch.zeros(channel_len))
            self.register_buffer('act_scaling_factor', torch.zeros(channel_len))

        self.quant_mode = quant_mode
        self.per_channel = per_channel

        if self.quant_mode == "none":
            self.act_function = None
        elif self.quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def forward(self, x, 
                pre_act_scaling_factor=None, 
                identity=None, 
                identity_scaling_factor=None,
                specified_min=None,
                specified_max=None):
        x_act = x if identity is None else identity + x
        if self.running_stat:
            if not self.percentile:
                if not self.per_channel:
                    x_min = x_act.data.min()
                    x_max = x_act.data.max()
                else:
                    x_min = x_act.data.min(axis=0).values.min(axis=0).values
                    x_max = x_act.data.max(axis=0).values.max(axis=0).values

            # Initialization
            if torch.eq(self.x_min, self.x_max).all():
                self.x_min = self.x_min + x_min
                self.x_max = self.x_max + x_max

            elif self.act_range_momentum == -1:
                self.x_min = torch.min(self.x_min, x_min)
                self.x_max = torch.max(self.x_max, x_max)
            else:
                self.x_min = self.x_min * self.act_range_momentum +\
                        x_min * (1 - self.act_range_momentum)
                self.x_max = self.x_max * self.act_range_momentum +\
                        x_max * (1 - self.act_range_momentum)

        if self.quant_mode == 'none':
            return x_act, None

        
        x_min = self.x_min if specified_min is None else specified_min
        x_max = self.x_max if specified_max is None else specified_max

        self.act_scaling_factor = symmetric_linear_quantization_params(
            self.activation_bit, x_min, x_max, 
            per_channel=self.per_channel)

        if pre_act_scaling_factor is None:
            # this is for the input quantization 
            quant_act_int = self.act_function(x, self.activation_bit, \
                    self.percentile, self.act_scaling_factor)
        else:
            quant_act_int = fixedpoint_mul.apply(
                    x, pre_act_scaling_factor, 
                    self.activation_bit, self.quant_mode, 
                    self.act_scaling_factor, 
                    identity, identity_scaling_factor)

        correct_output_scale = self.act_scaling_factor.view(-1)

        return quant_act_int * correct_output_scale, self.act_scaling_factor

class floor_ste(Function):
    """
    Straight-through Estimator(STE) for torch.floor()
    """
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class IntSoftmax(torch.nn.Module):
    """
    A specialized softmax implementation using integer arithmetic for quantized neural networks.
    """
    def __init__(self, output_bit, quant_mode='none', force_dequant='none'):
        super(IntSoftmax, self).__init__()
        self.output_bit = output_bit
        self.quant_mode = quant_mode

        # Setting up quantization for activations
        self.act = QuantAct(16, quant_mode=self.quant_mode)
        self.x0 = -0.6931  # Represents -ln(2)
        self.n = 30  # A large integer value for approximation
        # Polynomial coefficients for approximation
        self.coef = [0.35815147, 0.96963238, 1.]
        # Normalizing coefficients
        self.coef[1] /= self.coef[0]
        self.coef[2] /= self.coef[0]

    def int_polynomial(self, x_int, scaling_factor):
        """
        Performs an integer polynomial approximation.
        """
        with torch.no_grad():
            b_int = torch.floor(self.coef[1] / scaling_factor)
            c_int = torch.floor(self.coef[2] / scaling_factor ** 2)
        z = x_int + b_int
        z = x_int * z
        z += c_int
        scaling_factor = self.coef[0] * scaling_factor ** 2
        return z, scaling_factor

    def int_exp(self, x_int, scaling_factor):
        """
        Calculates the exponential of integer values using the integer polynomial method.
        """
        with torch.no_grad():
            x0_int = torch.floor(self.x0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int, exp_scaling_factor = self.int_polynomial(r, scaling_factor)
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = exp_scaling_factor / 2 ** self.n
        return exp_int, scaling_factor

    def forward(self, x, scaling_factor):
        """
        Forward pass of the IntSoftmax layer.
        """
        x_int = x / scaling_factor

        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int -= x_int_max

        exp_int, exp_scaling_factor = self.int_exp(x_int, scaling_factor)
        exp, exp_scaling_factor = self.act(exp_int, exp_scaling_factor)
        exp_int = exp / exp_scaling_factor
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

        factor = floor_ste.apply(2**32 / exp_int_sum)
        exp_int = floor_ste.apply(exp_int * factor / 2 ** (32 - self.output_bit))
        scaling_factor = 1 / 2 ** self.output_bit
        return exp_int * scaling_factor, scaling_factor

