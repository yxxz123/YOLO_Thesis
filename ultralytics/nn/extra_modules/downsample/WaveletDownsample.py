'''
本文件由BiliBili：魔傀面具整理     
ultralytics/nn/module_images/CVPR2026-WaveletDownsample.png
ultralytics/nn/module_images/CVPR2026-WaveletDownsample.md  
论文链接：https://openaccess.thecvf.com/content/CVPR2026F/papers/Ruan_LWTformer_A_Detail-Aware_Learnable_Wavelet-Transformer_for_Ancient_Chinese_Character_Image_CVPRF_2026_paper.pdf
'''
   
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops     
     
## Self-contained WaveletDownsample module extracted from models/LWTformer.py    
## Wavelet filters (dec_lo, dec_hi) are initialized inside the class.  
from typing import List, Sequence, Tuple, Union
import pywt 
import torch
import torch.nn as nn     
import torch.nn.functional as F     
from einops import rearrange  
   
##########################################################################
## Wavelet helpers (inlined from models/wavelet_utils.py)

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

    Returns:   
        torch.Tensor: Stacked 2d filters of dimension
            [filt_no, 1, height, width]. Ordered ll, lh, hl, hh.
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
    """Convert input wavelet to filter tensors dec_lo, dec_hi, rec_lo, rec_hi."""
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
    padr = (2 * filt_len - 3) // 2  
    padl = (2 * filt_len - 3) // 2   

    # pad to even signal length.
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
## WaveletDownsample   
class WaveletDownsample(nn.Module):
    def __init__(self, dim, out_dim=None, wavelet='haar', initialize_wavelet=True):     
        super().__init__()

        # Output channels; defaults to 2*dim (original behavior) 
        self.out_dim = out_dim if out_dim is not None else dim * 2

        self.pixel_down = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=False),    
            nn.PixelUnshuffle(2)  # (dim//2)*4 = 2*dim
        ) 
   
        # 2D-DWT filters, initialized inside the class 
        dec_lo, dec_hi, _, _ = get_filter_tensors(wavelet, flip=True)
        if initialize_wavelet:  
            self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)  
            self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)
        else:     
            self.dec_lo = nn.Parameter(torch.rand_like(dec_lo) * 2 - 1, requires_grad=True)  
            self.dec_hi = nn.Parameter(torch.rand_like(dec_hi) * 2 - 1, requires_grad=True)
        self.dwt = DWT(self.dec_lo, self.dec_hi, wavelet=wavelet, level=1)     

        # Subband Attention Branch (SAB)
        self.subband_attn = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=1, groups=dim * 4, bias=True),
            nn.GroupNorm(dim * 4, dim * 4),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim * 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()  # to [0,1]
        )
   
        # fusion: PWConv projects dim*2 -> out_dim   
        self.DWConv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=True)
        self.PWConv = nn.Conv2d(dim * 2, self.out_dim, kernel_size=1, stride=1, padding=0, bias=True)     
  
    def forward(self, x):     
        x_down = self.pixel_down(x)
        # 2D-DWT
        ya, (yh, yv, yd) = self.dwt(x)
    
        # Subband Attention Branch (SAB)  
        concat_subbands = torch.cat([ya, yh, yv, yd], dim=1)  # [B, 4C, H/2, W/2]
  
        attn_weights = self.subband_attn(concat_subbands)  # (B, 2*dim, H/2, W/2)   

        x_enhanced = x_down * attn_weights + x_down

        x_enhanced = self.PWConv(self.DWConv(x_enhanced))

        return x_enhanced

   
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)   

    module = WaveletDownsample(in_channel, out_channel).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
 
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module, 
                                     input_shape=(batch_size, in_channel, height, width),  
                                     output_as_string=True, 
                                     output_precision=4,     
                                     print_detailed=True)
    print(RESET)
   
    half_outputs = module.half()(inputs.half())  
    assert half_outputs.shape == outputs.shape and half_outputs.dtype is torch.float16
