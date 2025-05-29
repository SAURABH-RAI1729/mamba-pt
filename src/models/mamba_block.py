"""
MAMBA block implementation for particle tracking
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

try:
    from mamba_ssm import selective_scan_fn, selective_scan_ref
except ImportError:
    print("Warning: mamba_ssm not available, using reference implementation")
    selective_scan_fn = None

class MambaBlock(nn.Module):
    """
    MAMBA block with state space model for sequence modeling
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path and selective_scan_fn is not None
        
        # Linear projections
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=conv_bias,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize dt projection
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # SSM matrices
        self.A = nn.Parameter(torch.randn(self.d_state, self.d_inner))
        self.D = nn.Parameter(torch.randn(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, inference_params=None):
        """
        Args:
            x: (batch, length, d_model)
        Returns:
            output: (batch, length, d_model)
        """
        batch, length, _ = x.shape
        
        # Dual path: x and z
        xz = self.in_proj(x)  # (batch, length, 2 * d_inner)
        x, z = xz.split(self.d_inner, dim=-1)
        
        # Convolution with causal padding
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :length]
        x = rearrange(x, 'b d l -> b l d')
        
        # Apply activation
        x = F.silu(x)
        
        # SSM
        y = self.ssm(x, inference_params)
        
        # Gating
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        return self.dropout(output)
    
    def ssm(self, x, inference_params=None):
        """Selective scan (state space model)"""
        batch, length, _ = x.shape
        
        # Compute ∆, B, C
        deltaBC = self.x_proj(x)  # (batch, length, dt_rank + 2*d_state)
        delta, B, C = torch.split(
            deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        # Compute dt
        delta = self.dt_proj(delta)  # (batch, length, d_inner)
        delta = F.softplus(delta)
        
        if self.use_fast_path:
            # Use optimized CUDA kernel if available
            y = selective_scan_fn(
                x, delta, self.A, B, C, self.D, z=None,
                delta_softplus=False, return_last_state=False
            )
        else:
            # Reference implementation
            y = self.selective_scan_ref(x, delta, B, C)
        
        return y
    
    def selective_scan_ref(self, x, delta, B, C):
        """Reference implementation of selective scan"""
        batch, length, d_inner = x.shape
        
        A = -torch.exp(self.A.float())  # (d_state, d_inner)
        A = A.to(x.dtype)
        
        # State space model loop
        y = torch.zeros_like(x)  # (batch, length, d_inner)
        h = torch.zeros(batch, self.d_state, d_inner, device=x.device, dtype=x.dtype)
        
        for t in range(length):
            # Get current timestep values
            delta_t = delta[:, t]  # (batch, d_inner)
            B_t = B[:, t]          # (batch, d_state)  
            C_t = C[:, t]          # (batch, d_state)
            x_t = x[:, t]          # (batch, d_inner)
            
            # Discretize A: exp(delta * A)
            # A: (d_state, d_inner), delta_t: (batch, d_inner)
            # Result: (batch, d_state, d_inner)
            deltaA_t = torch.exp(delta_t.unsqueeze(1) * A.unsqueeze(0))
            
            # Discretize B: delta * B  
            # delta_t: (batch, d_inner), B_t: (batch, d_state)
            # Result: (batch, d_state, d_inner)
            deltaB_t = delta_t.unsqueeze(1) * B_t.unsqueeze(2)
            
            # Update state: h = deltaA * h + deltaB * x
            # h: (batch, d_state, d_inner)
            # deltaA_t: (batch, d_state, d_inner) 
            # deltaB_t: (batch, d_state, d_inner)
            # x_t: (batch, d_inner)
            h = deltaA_t * h + deltaB_t * x_t.unsqueeze(1)
            
            # Output: y = C * h + D * x
            # C_t: (batch, d_state), h: (batch, d_state, d_inner)
            # Result: (batch, d_inner)
            y[:, t] = torch.sum(C_t.unsqueeze(2) * h, dim=1) + self.D * x_t
        
        return y


class ResidualMambaBlock(nn.Module):
    """MAMBA block with residual connection and layer normalization"""
    
    def __init__(self, d_model, **mamba_kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(d_model, **mamba_kwargs)
        
    def forward(self, x, inference_params=None):
        """
        Args:
            x: (batch, length, d_model)
        Returns:
            output: (batch, length, d_model)
        """
        residual = x
        x = self.norm(x)
        x = self.mamba(x, inference_params)
        return residual + x
