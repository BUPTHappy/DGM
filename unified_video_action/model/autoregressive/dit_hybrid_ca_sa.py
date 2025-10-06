import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Your timestep embedder (unchanged API) ----------
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element (may be fractional).
        :return: (N, D)
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # t can be [B], [B,1], or [B,1,1]; squeeze to [B]
        if t.dim() == 3: t = t.squeeze(-1).squeeze(-1)
        elif t.dim() == 2: t = t.squeeze(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)  # [B, hidden_size]
        return t_emb


# ---------- The model you asked for (renamed to DiT) ----------
class DiT(nn.Module):
    """
    Policy-only DiT for action diffusion/consistency.
      - Input noisy actions: [B, L, in_channels]
      - Condition tokens:    [B, L_cond, z_channels]
      - One global t per sample: t ~ [B] / [B,1] / [B,1,1]

    Args (mapped from your signature):
      in_channels:  action_dim
      hidden_size:  transformer width D
      depth:        number of DiTPolicyBlock layers
      num_heads:    attention heads
      mlp_ratio:    MLP expansion in blocks
      z_channels:   condition token dim
      learn_sigma:  predict [eps, sigma] if True
    """
    def __init__(
        self,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        z_channels=1000,
        learn_sigma=False,
        max_len=32,             # max action steps per chunk (e.g., 16 or 32)
        out_parameterization="eps",  # "eps" / "v" / "x0" (handled in your loss)
        t_freq_dim=256,         # frequency dim for TimestepEmbedder
    ):
        super().__init__()
        self.action_dim   = in_channels
        self.d_model      = hidden_size
        self.n_layers     = depth
        self.nhead        = num_heads
        self.mlp_ratio    = mlp_ratio
        self.cond_dim     = z_channels
        self.learn_sigma  = learn_sigma
        self.max_len      = max_len
        self.out_parameterization = out_parameterization

        # Projections
        self.in_proj   = nn.Linear(self.action_dim, self.d_model)
        self.cond_proj = nn.Linear(self.cond_dim,   self.d_model)

        # Learned 1D temporal pos-emb inside the model
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, self.d_model))

        # Timestep embedder (your class)
        self.time_embed = TimestepEmbedder(self.d_model, frequency_embedding_size=t_freq_dim)

        # Stack of blocks
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=int(self.d_model * self.mlp_ratio), batch_first=True) #直接使用 PyTorch 内置的 TransformerDecoderLayer
            for _ in range(self.n_layers)
        ])

        # Output head
        out_dim = self.action_dim * (2 if self.learn_sigma else 1)
        self.out_norm = nn.LayerNorm(self.d_model)
        self.out_proj = nn.Linear(self.d_model, out_dim)

        # Init
        self.initialize_weights()

    # ----- Initialization -----
    def initialize_weights(self):
        # Generic linear/ln init
        def init_linear(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(init_linear)

        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Positional embedding starts at zero => identity at init
        with torch.no_grad():
            self.pos_embed.zero_()

        # Output head
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    # ----- Forward -----
    def forward(
        self,
        x_noisy: torch.Tensor,          # [B, L, action_dim]
        t: torch.Tensor,                # [B], [B,1], or [B,1,1]
        c: torch.Tensor,      # [B, L_cond, z_channels]
        attn_mask: torch.Tensor = None, # optional cross-attn mask (usually None)
        key_pad_mask: torch.Tensor = None, # optional [B, L_cond] bool, True=pad
    ):
        B, L, _ = x_noisy.shape
        assert L <= self.max_len, f"Increase max_len ({self.max_len}) or reduce L ({L})."

        # Project inputs
        x = self.in_proj(x_noisy)                   # [B, L, D]
        x = x + self.pos_embed[:, :L, :]            # learned temporal pos-emb

        cond = self.cond_proj(c)          # [B, L_cond, D]

        # Global timestep embedding per sample -> [B, D]
        t_emb = self.time_embed(t)
        cond = torch.cat([cond, t_emb[:, None, :]], dim=1)

        # Blocks
        for blk in self.blocks:
            x = blk(x, cond)

        # Output
        x = self.out_norm(x)
        out = self.out_proj(x)                      # [B, L, action_dim] or [B, L, 2*action_dim]
        return out