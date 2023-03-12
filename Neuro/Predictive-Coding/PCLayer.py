import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair
from torch import Tensor
from typing import Optional, List, Tuple, Union

class PCLayer(nn.Module):
    def __init__(self, 
                 e_shape,
                 r_shape,
                 kernel,
                 
                 nu,
                 mu,
                 eta,

                 maxpool=1,
                 bu_actv=nn.Tanh(),
                 td_actv=nn.Tanh(),
                 
                 padding=0,
                 relu_errs=True,
                 **kwargs,
                ):
        super().__init__()
        self.e_shape = e_shape
        self.r_shape = r_shape

        self.nu = nu
        self.mu = mu
        self.eta = eta

        self.relu_errs = relu_errs
        self.device = "cpu"

        self.bottomUp = nn.Sequential(
            nn.Conv2d(e_shape[0], r_shape[0], kernel, padding=padding, **kwargs),
            nn.MaxPool2d(kernel_size=maxpool),
            bu_actv,
        )
        
        self.topDown = nn.Sequential(
            nn.Upsample(scale_factor=maxpool),
            nn.ConvTranspose2d(r_shape[0], e_shape[0], kernel, padding=padding, **kwargs),
            td_actv,
        )
    
    def init_vars(self, batch_size):
        e = torch.zeros((batch_size, self.e_shape[0], self.e_shape[1], self.e_shape[2])).to(self.device)
        r = torch.zeros((batch_size, self.r_shape[0], self.r_shape[1], self.r_shape[2])).to(self.device)
        return e,r
    
    def forward(self, x, e, r, td_err=None):
        e = x - self.topDown(r)
        if self.relu_errs:
            e = F.relu(e)
        r = self.nu*r + self.mu*self.bottomUp(e)
        if td_err is not None:
            r += self.eta*td_err
        return e, r

class PCLayerv2(nn.Modules.Conv._ConvNd):

    def __init__(
            self, 
            e_shape,
            r_shape,
            kernel_size: nn._size_2_t,
            stride: nn._size_2_t = 1,
            padding: Union[str, nn._size_2_t] = 0,
            dilation: nn._size_2_t = 1,
            groups: int = 1,
            r_bias: bool = True,
            e_bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None,
                 
            nu=1.0,
            mu=0.2,
            eta=0.05,

            td_actv=F.tanh(),

            maxpool=None,
            relu_errs=True,
        ) -> None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(PCLayerv2, self).__init__(
            e_shape[0], r_shape[0], kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, False, padding_mode, **self.factory_kwargs
        )
        self.e_shape = e_shape
        self.r_shape = r_shape

        self.nu = nu
        self.mu = mu
        self.eta = eta

        self.td_actv = td_actv
        self.relu_errs = relu_errs
        self.device = "cpu"
    
    def init_vars(self, batch_size):
        e = torch.zeros((batch_size, self.e_shape[0], self.e_shape[1], self.e_shape[2]), **self.factory_kwargs)
        r = torch.zeros((batch_size, self.r_shape[0], self.r_shape[1], self.r_shape[2]), **self.factory_kwargs)
        return e,r

    def _conv_forward(self, input: Tensor):
        if self.padding_mode != 'zeros':
            return F.conv2d(
                F.pad(input, self.reversed_padding_repeated_twice, mode=self.padding_mode),
                self.weight,
                self.r_bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,                            
            )
        else:
            return F.conv2d(
                input,
                self.weight,
                self.r_bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    def _output_padding(self, input: Tensor, output_size: Optional[List[int]],
                            stride: List[int], padding: List[int], kernel_size: List[int],
                            num_spatial_dims: int, dilation: Optional[List[int]] = None) -> List[int]:
            if output_size is None:
                ret = _single(self.output_padding)  # converting to list if was not already
            else:
                has_batch_dim = input.dim() == num_spatial_dims + 2
                num_non_spatial_dims = 2 if has_batch_dim else 1
                if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                    output_size = output_size[num_non_spatial_dims:]
                if len(output_size) != num_spatial_dims:
                    raise ValueError(
                        "ConvTranspose{}D: for {}D input, output_size must have {} or {} elements (got {})"
                        .format(num_spatial_dims, input.dim(), num_spatial_dims,
                                num_non_spatial_dims + num_spatial_dims, len(output_size)))

                min_sizes = torch.jit.annotate(List[int], [])
                max_sizes = torch.jit.annotate(List[int], [])
                for d in range(num_spatial_dims):
                    dim_size = ((input.size(d + num_non_spatial_dims) - 1) * stride[d] -
                                2 * padding[d] +
                                (dilation[d] if dilation is not None else 1) * (kernel_size[d] - 1) + 1)
                    min_sizes.append(dim_size)
                    max_sizes.append(min_sizes[d] + stride[d] - 1)

                for i in range(len(output_size)):
                    size = output_size[i]
                    min_size = min_sizes[i]
                    max_size = max_sizes[i]
                    if size < min_size or size > max_size:
                        raise ValueError((
                            "requested an output size of {}, but valid sizes range "
                            "from {} to {} (for an input of {})").format(
                                output_size, min_sizes, max_sizes, input.size()[2:]))

                res = torch.jit.annotate(List[int], [])
                for d in range(num_spatial_dims):
                    res.append(output_size[d] - min_sizes[d])

                ret = res
            return ret

    def _conv_backward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            raise ValueError('Only "zeros" padding mode is supported for ConvTranspose2d')
        assert isinstance(self.padding, tuple)
        num_spatial_dims = 2
        output_padding = self._output_padding(
            input, self.e_shape, self.stride, self.padding, 
            self.kernel_size, num_spatial_dims, self.dilation
        )  
        return F.conv_transpose2d(
            input, self.weight.T, self.e_bias, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation
        )
    
    def forward(self, x, e, r, td_err=None):
        e = x - self.td_actv(self._conv_backward(r))
        if self.relu_errs:
            e = F.relu(e)
        r = self.nu*r + self.mu*self._conv_forward(e)
        if td_err is not None:
            r += self.eta*td_err
        return e, r
