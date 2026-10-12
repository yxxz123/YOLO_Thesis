'''
本文件由BiliBili：魔傀面具整理  
ultralytics/nn/module_images/CVPR2026-WACGA.png  
ultralytics/nn/module_images/CVPR2026-WACGA.md     
论文链接：https://openaccess.thecvf.com/content/CVPR2026F/papers/Ruan_LWTformer_A_Detail-Aware_Learnable_Wavelet-Transformer_for_Ancient_Chinese_Character_Image_CVPRF_2026_paper.pdf
'''     

import warnings  
warnings.filterwarnings('ignore')
from calflops import calculate_flops
 
## Wavelet-Aware Convolutional Gating Attention (WACGA)
## Extracted from models/LWTformer.py as a self-contained module.   
import torch    
import torch.nn as nn     
import torch.nn.functional as F
import pywt
from typing import Sequence, Tuple, Union, List  
from einops import rearrange     
     

##########################################################################
## Wavelet utilities (inlined from models/wavelet_utils.py)   

def _as_wavelet(wavelet):    
    """Ensure the input argument to be a pywt wavelet compatible object."""
    if isinstance(wavelet, str):     
        return pywt.Wavelet(wavelet)
    else:
        return wavelet 
    
    
def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: 
    """Torch implementation of numpy's outer for 1d vectors."""    
    a_flat = torch.reshape(a, [-1])   
    b_flat = torch.reshape(b, [-1])   
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)   
    return a_mul * b_mul
 
    
def construct_2d_filt(lo, hi) -> torch.Tensor:
    """Construct two dimensional filters using outer products. 

    Returns stacked 2d filters of dimension [filt_no, 1, height, width],
    ordered ll, lh, hl, hh.
    """
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)    
    filt = torch.stack([ll, lh, hl, hh], 0)
    return filt
   

def get_filter_tensors(
        wavelet,
        flip: bool,   
        device: Union[torch.device, str] = 'cpu',
        dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert input wavelet to filter tensors (dec_lo, dec_hi, rec_lo, rec_hi)."""
    wavelet = _as_wavelet(wavelet)

    def _create_tensor(filter: Sequence[float]) -> torch.Tensor:
        if flip:
            if isinstance(filter, torch.Tensor): 
                return filter.flip(-1).unsqueeze(0).to(device)
            else:
                return torch.tensor(filter[::-1], device=device, dtype=dtype).unsqueeze(0)
        else:   
            if isinstance(filter, torch.Tensor):
                return filter.unsqueeze(0).to(device)    
            else:
                return torch.tensor(filter, device=device, dtype=dtype).unsqueeze(0)  

    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank  
    dec_lo_tensor = _create_tensor(dec_lo)
    dec_hi_tensor = _create_tensor(dec_hi) 
    rec_lo_tensor = _create_tensor(rec_lo)
    rec_hi_tensor = _create_tensor(rec_hi)     
    return dec_lo_tensor, dec_hi_tensor, rec_lo_tensor, rec_hi_tensor


def _get_pad(data_len: int, filt_len: int) -> Tuple[int, int]:     
    """Compute the required padding."""
    # pad to ensure we see all filter positions and for pywt compatability.
    padr = (2 * filt_len - 3) // 2  
    padl = (2 * filt_len - 3) // 2   
     
    # pad to even singal length.  
    if data_len % 2 != 0:
        padr += 1    
   
    return padr, padl  

  
def fwt_pad2(   
        data: torch.Tensor, wavelet, mode: str = "replicate"
) -> torch.Tensor:
    """Pad data for the 2d FWT."""     
    wavelet = _as_wavelet(wavelet)   
    padb, padt = _get_pad(data.shape[-2], len(wavelet.dec_lo))
    padr, padl = _get_pad(data.shape[-1], len(wavelet.dec_lo))
 
    data_pad = F.pad(data, [padl, padr, padt, padb], mode=mode)
    return data_pad   
 

class DWT(nn.Module):
    def __init__(self, dec_lo, dec_hi, wavelet='haar', level=1, mode="replicate"): 
        super(DWT, self).__init__()
        self.wavelet = _as_wavelet(wavelet)
        self.dec_lo = dec_lo     
        self.dec_hi = dec_hi
        self.level = level     
        self.mode = mode 
     
    def forward(self, x):
        b, c, h, w = x.shape
        if self.level is None:
            self.level = pywt.dwtn_max_level([h, w], self.wavelet) 
        wavelet_component: List[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = []     

        l_component = x   
        dwt_kernel = construct_2d_filt(lo=self.dec_lo, hi=self.dec_hi)
        dwt_kernel = dwt_kernel.repeat(c, 1, 1)
        dwt_kernel = dwt_kernel.unsqueeze(dim=1)    
        for _ in range(self.level):
            l_component = fwt_pad2(l_component, self.wavelet, mode=self.mode)  
            h_component = F.conv2d(l_component, dwt_kernel, stride=2, groups=c)
            res = rearrange(h_component, 'b (c f) h w -> b c f h w', f=4)
            l_component, lh_component, hl_component, hh_component = res.split(1, 2)   
            wavelet_component.append((lh_component.squeeze(2), hl_component.squeeze(2), hh_component.squeeze(2)))
        wavelet_component.append(l_component.squeeze(2))     
        return wavelet_component[::-1]   
  

##########################################################################
## Wavelet-Aware Convolutional Gating Attention (WACGA)    
class WACGA(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True, wavelet='haar', dec_lo=None, dec_hi=None):  
        super(WACGA, self).__init__()    
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
     
        # QKV projection  
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias) 
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)   
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias) 
     
        # Wavelet filters (learnable or reused) 
        if dec_lo is not None and dec_hi is not None:  
            # Use externally provided learnable filters
            self.dec_lo = dec_lo
            self.dec_hi = dec_hi   
        else:
            dec_lo, dec_hi, _, _ = get_filter_tensors(wavelet, flip=True)   
            self.dec_lo = nn.Parameter(dec_lo, requires_grad=True) 
            self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)
  
        # Define DWT module
        self.dwt = DWT(self.dec_lo, self.dec_hi, wavelet=wavelet, level=1)    

        # Subband-specific processing modules   
        # 1. Low-frequency subband (ya): Regular convolution 
        self.ya_proj = nn.Conv2d(dim, dim // 4, kernel_size=1, bias=bias)
   
        # 2. Horizontal subband (yh): Horizontal convolution kernel (1x3)
        self.yh_conv = nn.Conv2d(dim, dim // 4, kernel_size=(1, 3), padding=(0, 1), groups=dim // 4, bias=bias) 
     
        # 3. Vertical subband (yv): Vertical convolution kernel (3x1)    
        self.yv_conv = nn.Conv2d(dim, dim // 4, kernel_size=(3, 1), padding=(1, 0), groups=dim // 4, bias=bias)
     
        # 4. Diagonal subband (yd): Only activation function 
        self.yd_act = nn.Tanh() 
        self.yd_proj = nn.Conv2d(dim, dim // 4, kernel_size=1, bias=bias)
 
        # Subband fusion and attention generation  
        self.subband_fusion = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias),
            nn.Sigmoid()   
        )
   
        self.ll_conv = nn.Conv2d(dim // 4, dim // 4, kernel_size=3, stride=1, padding=1, groups=dim // 4, bias=bias)     
        # Create directional convolution kernels 
        self.horizontal_conv, self.vertical_conv, self.diagonal_conv = self.create_wave_conv(dim // 4)    
 
    def create_conv_layer(self, kernel, dim):     
        """Create convolution layer with fixed weights"""
        conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False)
        conv.weight.data = kernel.repeat(dim, 1, 1, 1) 
        # conv.weight.requires_grad = False  # Fixed weights    
        return conv  

    def create_wave_conv(self, dim):
        """Create convolution kernels matching subband characteristics"""   
    
        # yh
        horizontal_kernel = torch.tensor([[1, 1, 1],
                                          [0, 0, 0], 
                                          [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
 
        # yv
        vertical_kernel = torch.tensor([[1, 0, -1],  
                                        [1, 0, -1],   
                                        [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
        # yd
        diagonal_kernel = torch.tensor([[0, 1, 0],  
                                        [1, -4, 1], 
                                        [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
 
        horizontal_conv = self.create_conv_layer(horizontal_kernel, dim) 
        vertical_conv = self.create_conv_layer(vertical_kernel, dim)  
        diagonal_conv = self.create_conv_layer(diagonal_kernel, dim)    
  
        return horizontal_conv, vertical_conv, diagonal_conv  

    def forward(self, x):
        b, c, h, w = x.shape

        ya, (yh, yv, yd) = self.dwt(x)  # [B, C, H/2, W/2]     
   
        # ya 
        ya_proc = self.ya_proj(ya)  # [B, C/4, H/2, W/2]    
        ya_proc = self.ll_conv(ya_proc)
     
        # yh 
        yh_proc = self.yh_conv(yh)  # [B, C/4, H/2, W/2]
        yh_proc = self.horizontal_conv(yh_proc)

        # yv  
        yv_proc = self.yv_conv(yv)  # [B, C/4, H/2, W/2]
        yv_proc = self.vertical_conv(yv_proc)    

        # yd   
        yd_proc = self.yd_act(yd)  # [B, C, H/2, W/2]
        yd_proc = self.yd_proj(yd_proc)
        yd_proc = self.diagonal_conv(yd_proc)
     
        subbands_proc = torch.cat([ya_proc, yh_proc, yv_proc, yd_proc], dim=1)  # [B, C, H/2, W/2]
     
        wavelet_attention_map = self.subband_fusion(subbands_proc)     
        wavelet_attention_map = F.interpolate(    
            wavelet_attention_map,     
            size=x.shape[2:],
            mode='bilinear',   
            align_corners=False
        )

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)   
     
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)    

        attn = (q @ k.transpose(-2, -1)) * self.temperature 
        attn = attn.softmax(dim=-1)
        out = attn @ v   

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)  
        out = self.project_out(out) 

        out = out * wavelet_attention_map  
        return out
   

if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, channel, height, width = 1, 16, 20, 20     
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    module = WACGA(channel).to(device)   

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
 
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module, 
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)
     
    half_outputs = module.half()(inputs.half())
    assert half_outputs.shape == outputs.shape and half_outputs.dtype is torch.float16
