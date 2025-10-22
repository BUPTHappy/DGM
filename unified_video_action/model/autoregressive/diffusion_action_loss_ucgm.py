import torch
import torch.nn as nn
from einops import rearrange

from unified_video_action.model.ucgm.ucgm import UCGMTS
from unified_video_action.model.autoregressive.diffusion_loss import SimpleMLPAdaLN


class DiffActLossUCGM(nn.Module):
    """Diffusion Loss with UCGM"""

    def __init__(
        self,
        target_channels,
        z_channels,
        depth,
        width,
        num_sampling_steps,
        grad_checkpointing=False,
        n_frames=4,
        act_diff_training_steps=1000,
        act_diff_testing_steps="100",
        act_model_type="conv_fc",
        ucgmts_config={},
        **kwargs
    ):
        super(DiffActLossUCGM, self).__init__()
        self.in_channels = target_channels
        self.n_frames = n_frames

        self.language_emb_model = kwargs["language_emb_model"]
        self.language_emb_model_type = kwargs["language_emb_model_type"]

        self.act_model_type = act_model_type

        if self.act_model_type == "conv_fc":
            self.w = 16
            self.h = 16
            self.num_frames = 4
            self.num_actions = 16

            # Single convolutional layer for spatial processing
            self.conv = nn.Sequential(
                nn.Conv2d(z_channels, z_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),  # Reduce to a fixed spatial size of 4x4
            )

            # Fully connected layer for action latent prediction
            self.fc = nn.Sequential(
                nn.Linear(z_channels * 4 * 4, z_channels),
                nn.ReLU(),
                nn.Linear(z_channels, z_channels),  # Predict latents for all actions
            )

            self.interpolate = nn.Linear(self.num_frames, self.num_actions)

            self.refine = nn.Sequential(
                nn.Linear(z_channels, z_channels),
                nn.ReLU(),
                nn.Linear(z_channels, z_channels),
            )

        elif self.act_model_type == "conv_ori":
            self.w = 16
            self.h = 16
            self.conv_transpose3d = nn.ConvTranspose3d(
                in_channels=z_channels,
                out_channels=z_channels,
                kernel_size=(4, 1, 1),
                stride=(4, 1, 1),
            )
            self.avg_pool = nn.AvgPool3d(kernel_size=(1, self.w, self.h))
            
        elif self.act_model_type == 'conv2':
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(in_channels=256, out_channels=16, kernel_size=7, padding=3)
            )
        
        elif self.act_model_type == 'fc2':
            self.fc = nn.Sequential(
                nn.Linear(1024, 256),
                nn.ReLU(),  # Add an activation function (optional, but common practice)
                nn.Linear(256, 16)
            )
            
        else:
            raise NotImplementedError

        # Use only MLP architecture
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2,  # 4D输出，与原始MLP版本一致
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
        )

        self.num_sampling_steps = num_sampling_steps

        print(f"DiffActLossUCGM: num_sampling_steps: {num_sampling_steps}")
        print("UCGMTS config values:")
        print("  transport_type:", ucgmts_config.get("transport_type", "Linear"))
        print("  scaled_cbl_eps:", ucgmts_config.get("scaled_cbl_eps", 0.0))
        print("  ema_decay_rate:", ucgmts_config.get("ema_decay_rate", 0.0))
        print("  consistc_ratio:", ucgmts_config.get("consistc_ratio", 1.0))
        print("  rfba_gap_steps:", ucgmts_config.get("rfba_gap_steps", [0.001, 0.5]))
        print("  lab_drop_ratio:", ucgmts_config.get("lab_drop_ratio", 0.1))
        print("  enhanced_ratio:", ucgmts_config.get("enhanced_ratio", 0.0))
        print("  wt_cosine_loss:", ucgmts_config.get("wt_cosine_loss", False))
        print("  weight_function:", ucgmts_config.get("weight_function", None))
        print("  time_dist_ctrl:", ucgmts_config.get("time_dist_ctrl", [1.0, 1.0, 1.0]))

        self.ucgmts = UCGMTS(
            transport_type=ucgmts_config.get("transport_type", "Linear"),
            scaled_cbl_eps=ucgmts_config.get("scaled_cbl_eps", 0.0),
            ema_decay_rate=ucgmts_config.get("ema_decay_rate", 0.0),
            consistc_ratio=ucgmts_config.get("consistc_ratio", 1.0),
            lab_drop_ratio=ucgmts_config.get("lab_drop_ratio", 0.1),
            enhanced_ratio=ucgmts_config.get("enhanced_ratio", 0.0),
            wt_cosine_loss=ucgmts_config.get("wt_cosine_loss", False),
            weight_funcion=ucgmts_config.get("weight_function", None),
            time_dist_ctrl=ucgmts_config.get("time_dist_ctrl", [1.0, 1.0, 1.0])
        )
        self.stochasticity_ratio = ucgmts_config.get("consistc_ratio", 1.0)
        self.rfba_gap_steps = ucgmts_config.get("rfba_gap_steps", [0.001, 0.5])

        
    def forward(self, target, z, task_mode=None, text_latents=None):
        bsz, seq_len, _ = target.shape

        if self.act_model_type == "conv_fc":
            z = rearrange(z, "b (t s) c -> (b t) s c", t=self.n_frames)
            z = rearrange(z, "b (w h) c -> b w h c", w=self.w)
            z = rearrange(z, "b w h c -> b c w h")
            z = self.conv(z)
            z = rearrange(z, "b c w h -> b (c w h)")
            z = self.fc(z) 

            z = rearrange(z, "(b t) c -> b t c", t=self.n_frames)
            z = z.permute(0, 2, 1)
            z = self.interpolate(z)
            z = z.permute(0, 2, 1)
            z = self.refine(z)
            
        elif self.act_model_type == "conv_ori":
            z = rearrange(
                z, "b (t s) c -> b t s c", t=self.n_frames
            )
            z = rearrange(
                z, "b t (w h) c -> b c t w h", w=self.w
            )
            z = self.conv_transpose3d(z)
            z = self.avg_pool(z)
            z = rearrange(z, "b c t w h -> b (t w h) c")
            
        elif self.act_model_type == 'conv2':
            z = self.conv(z)
        
        elif self.act_model_type == 'fc2':
            z = self.fc(z.transpose(1, 2))
            z = z.transpose(1, 2)
        else:
            raise NotImplementedError

        # Reshape for MLP processing
        z = z.reshape(bsz * seq_len, -1)
        target = target.reshape(bsz * seq_len, -1)

        loss = self.ucgmts.training_step(model=self.net, x=target, c=z)
        loss = torch.mean(loss)
        return loss
    
    def sample(self, z, temperature=1.0, cfg=1.0, text_latents=None):
        if self.act_model_type == "conv_fc":
            z = rearrange(z, "b (t s) c -> (b t) s c", t=self.n_frames)
            z = rearrange(z, "b (w h) c -> b w h c", w=self.w)
            z = rearrange(z, "b w h c -> b c w h")
            z = self.conv(z)
            z = rearrange(z, "b c w h -> b (c w h)")
            z = self.fc(z)

            z = rearrange(z, "(b t) c -> b t c", t=self.n_frames)
            z = z.permute(0, 2, 1)
            z = self.interpolate(z)
            z = z.permute(0, 2, 1)
            z = self.refine(z)
            
        elif self.act_model_type == "conv_ori":
            z = rearrange(
                z, "b (t s) c -> b t s c", t=self.n_frames
            )
            z = rearrange(
                z, "b t (w h) c -> b c t w h", w=self.w
            )
            z = self.conv_transpose3d(z)
            z = self.avg_pool(z)
            z = rearrange(z, "b c t w h -> b (t w h) c")
        
        elif self.act_model_type == 'conv2':
            z = self.conv(z)
            
        elif self.act_model_type == 'fc2':
            z = self.fc(z.transpose(1, 2))
            z = z.transpose(1, 2)
        else:
            raise NotImplementedError
        
        bsz, seq_len, _ = z.shape

        # Reshape for MLP processing
        z = rearrange(z, "b t c -> (b t) c")
        noise = torch.randn(z.shape[0], self.in_channels, device=z.device)

        model_kwargs = dict(c=z)

        sampled_token = self.ucgmts.uni_sample(
                inital_noise_z=noise,
                sampling_model=self.net,
                sampling_steps=self.num_sampling_steps,
                stochast_ratio=self.stochasticity_ratio,
                extrapol_ratio=0,
                sampling_order=1,
                time_dist_ctrl=[1.17, 0.8, 1.1],
                rfba_gap_steps=self.rfba_gap_steps,
                **model_kwargs,)[-1]

        sampled_token = rearrange(
            sampled_token, "(b t) c -> b t c", b=bsz
        )
        # 取前2维作为最终动作输出，与原始MLP版本一致
        return sampled_token[:, :, :2]

