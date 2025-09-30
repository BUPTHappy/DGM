import torch
import math
import numpy as np

import torch
import math

def interpolate_fn(x, xp, yp):
    """
    Linear interpolation.
    x: points to evaluate at.
    xp: data points (x-coordinates).
    yp: data points (y-coordinates).
    Assumes xp is sorted.
    """
    # Ensure tensors are on the same device and dtype
    xp = xp.to(x.device, dtype=x.dtype)
    yp = yp.to(x.device, dtype=x.dtype)

    # Find the indices of the interval that each x belongs to
    # torch.searchsorted returns the index i such that xp[i-1] < x <= xp[i]
    # We need to handle cases where x is outside the range of xp
    i = torch.searchsorted(xp.squeeze(), x.squeeze(), right=True)

    # Clamp indices to be within the valid range
    i = torch.clamp(i, 1, len(xp.squeeze()) - 1)

    # Perform linear interpolation
    # y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    x0 = xp.squeeze()[i - 1]
    x1 = xp.squeeze()[i]
    y0 = yp.squeeze()[i - 1]
    y1 = yp.squeeze()[i]

    # Avoid division by zero if x0 == x1
    # If x0 == x1, it means we're at a data point, so the interpolated value is y0 (or y1)
    denom = x1 - x0
    # Add a small epsilon to prevent division by zero in cases where x1 and x0 are extremely close
    # but not exactly equal, which might happen due to floating point inaccuracies.
    # Or, more robustly, handle the case where denom is zero.
    output = y0 + (x.squeeze() - x0) * (y1 - y0) / torch.where(denom == 0, torch.ones_like(denom), denom)
    # If denom is 0, return y0. This handles the case where x falls exactly on xp[i] and xp[i-1] == xp[i].
    output = torch.where(denom == 0, y0, output)

    return output.reshape(x.shape)


class NoiseScheduleVP:
    def __init__(
            self,
            schedule='discrete',
            betas=None, # torch.Tensor
            alphas_cumprod=None, # torch.Tensor
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            device='cpu', # Added device parameter
    ):
        """Create a wrapper class for the forward SDE (VP type).

        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        ***

        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:

                log_alpha_t = self.marginal_log_mean_coeff(t)
                sigma_t = self.marginal_std(t)
                lambda_t = self.marginal_lambda(t)

        Moreover, as lambda(t) is an invertible function, we also support its inverse function:

                t = self.inverse_lambda(lambda_t)

        ===============================================================

        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).

        1. For discrete-time DPMs:

                For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                        t_i = (i + 1) / N
                e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
                We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.

                Args:
                        betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                        alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)

                Note that we always have alphas_cumprod = cumprod(1 - betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.

                **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                        The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                                q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
                        Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                                alpha_{t_n} = \sqrt{\hat{alpha_n}},
                        and
                                log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).


        2. For continuous-time DPMs:

                We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
                schedule are the default settings in DDPM and improved-DDPM:

                Args:
                        beta_min: A `float` number. The smallest beta for the linear schedule. (Handled by continuous_beta_0)
                        beta_max: A `float` number. The largest beta for the linear schedule. (Handled by continuous_beta_1)
                        cosine_s: A `float` number. The hyperparameter in the cosine schedule.
                        cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.
                        T: A `float` number. The ending time of the forward process.

        ===============================================================

        Args:
                schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                        'linear' or 'cosine' for continuous-time DPMs.
                device: A `str` or `torch.device`. The device to store tensors on.
        Returns:
                A wrapper object of the forward SDE (VP type).

        ===============================================================

        Example:

        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        >>> betas_tensor = torch.linspace(0.0001, 0.02, 1000)
        >>> ns = NoiseScheduleVP('discrete', betas=betas_tensor)

        # For discrete-time DPMs, given alphas_cumprod (the \hat{alpha_n} array for n = 0, 1, ..., N - 1):
        >>> alphas_c_tensor = torch.cumprod(1. - torch.linspace(0.0001, 0.02, 1000), dim=0)
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_c_tensor)

        # For continuous-time DPMs (VPSDE), linear schedule:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)
        """

        if schedule not in ['discrete', 'linear', 'cosine']:
            raise ValueError(f"Unsupported noise schedule {schedule}. The schedule needs to be 'discrete', 'linear', or 'cosine'")

        self.schedule = schedule
        self.device = device # Store the device

        if schedule == 'discrete':
            if betas is not None:
                # Ensure betas is a torch tensor and on the correct device
                betas = torch.as_tensor(betas, dtype=torch.float32, device=self.device)
                # Original JAX: log_alphas = 0.5 * jnp.log(1 - betas).cumsum(axis=0)
                # PyTorch:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None, "alphas_cumprod must be provided if betas is None for discrete schedule"
                # Ensure alphas_cumprod is a torch tensor and on the correct device
                alphas_cumprod = torch.as_tensor(alphas_cumprod, dtype=torch.float32, device=self.device)
                # Original JAX: log_alphas = 0.5 * jnp.log(alphas_cumprod)
                # PyTorch:
                log_alphas = 0.5 * torch.log(alphas_cumprod)

            self.total_N = len(log_alphas)
            self.T = 1.
            # Original JAX: self.t_array = jnp.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1))
            # PyTorch:
            self.t_array = torch.linspace(0., 1., self.total_N + 1, device=self.device)[1:].reshape((1, -1))
            # Original JAX: self.log_alpha_array = log_alphas.reshape((1, -1,))
            # PyTorch:
            self.log_alpha_array = log_alphas.reshape((1, -1,))
        else:
            self.total_N = 1000 # Default N for continuous schedules
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.
            # Original JAX: self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            # PyTorch: (math functions work with floats, no change needed for these scalar initializations)
            self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            # Original JAX: self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
            # PyTorch:
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))

            if schedule == 'cosine':
                # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
                # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
                self.T = 0.9946
            else: # linear
                self.T = 1.

    def _t_to_tensor(self, t):
        """Helper to convert t to a tensor and move to the correct device."""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device, dtype=torch.float32)
        else:
            t = t.to(self.device, dtype=torch.float32)
        return t

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        t can be a float or a torch.Tensor.
        """
        t = self._t_to_tensor(t) # Ensure t is a tensor on the correct device

        if self.schedule == 'discrete':
            # Original JAX: return interpolate_fn(t.reshape((-1, 1)), self.t_array, self.log_alpha_array).reshape((-1))
            # PyTorch:
            # Ensure t has the right shape for interpolate_fn if it expects (N, 1)
            # and self.t_array, self.log_alpha_array are (1, M)
            t_reshaped = t.reshape((-1, 1))
            interpolated_values = interpolate_fn(t_reshaped, self.t_array.T, self.log_alpha_array.T) # Transpose for 1D interpolation
            return interpolated_values.reshape(-1)
        elif self.schedule == 'linear':
            # Original JAX: return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
            # PyTorch: (direct translation)
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            # Original JAX:
            # log_alpha_fn = lambda s: jnp.log(jnp.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            # log_alpha_t =  log_alpha_fn(t) - self.cosine_log_alpha_0
            # PyTorch:
            log_alpha_t = torch.log(torch.cos((t + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.)) - self.cosine_log_alpha_0
            return log_alpha_t
        else: # Should not happen due to init check
             raise ValueError(f"Unknown schedule: {self.schedule}")


    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        # Original JAX: return jnp.exp(self.marginal_log_mean_coeff(t))
        # PyTorch:
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        # Original JAX: return jnp.sqrt(1. - jnp.exp(2. * self.marginal_log_mean_coeff(t)))
        # PyTorch:
        log_mean_coeff_t = self.marginal_log_mean_coeff(t)
        # Ensure the argument to sqrt is non-negative
        # Clamp to avoid NaN from sqrt of small negative numbers due to precision issues
        value_inside_sqrt = 1. - torch.exp(2. * log_mean_coeff_t)
        return torch.sqrt(torch.clamp(value_inside_sqrt, min=1e-8)) # Added clamp for numerical stability

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        # Original JAX:
        # log_mean_coeff = self.marginal_log_mean_coeff(t)
        # log_std = 0.5 * jnp.log(1. - jnp.exp(2. * log_mean_coeff))
        # return log_mean_coeff - log_std
        # PyTorch:
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        # Use torch.log1p for better precision with log(1-x) when x is small
        # exp(2*log_mean_coeff) = alpha_t^2
        # log(1 - alpha_t^2) = log(sigma_t^2) = 2 * log(sigma_t)
        # So, 0.5 * log(1 - exp(2*log_mean_coeff)) is log(sigma_t)
        # Clamp to avoid log of zero or negative
        term_inside_log = 1. - torch.exp(2. * log_mean_coeff)
        log_std = 0.5 * torch.log(torch.clamp(term_inside_log, min=1e-8)) # Added clamp
        return log_mean_coeff - log_std

    def _lamb_to_tensor(self, lamb):
        """Helper to convert lambda to a tensor and move to the correct device."""
        if not isinstance(lamb, torch.Tensor):
            lamb = torch.tensor(lamb, device=self.device, dtype=torch.float32)
        else:
            lamb = lamb.to(self.device, dtype=torch.float32)
        return lamb

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        lamb can be a float or a torch.Tensor.
        """
        lamb = self._lamb_to_tensor(lamb) # Ensure lamb is a tensor on the correct device

        if self.schedule == 'linear':
            # Original JAX:
            # tmp = 2. * (self.beta_1 - self.beta_0) * jnp.logaddexp(-2. * lamb, jnp.zeros((1,)))
            # Delta = self.beta_0**2 + tmp
            # return tmp / (jnp.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
            # PyTorch:
            # jnp.logaddexp(x, y) = log(exp(x) + exp(y))
            # jnp.logaddexp(-2. * lamb, jnp.zeros((1,))) -> log(exp(-2*lamb) + 1)
            log_add_exp_term = torch.log(torch.exp(-2. * lamb) + 1.)
            # Ensure it handles scalar lamb correctly by matching dimensions if needed for broadcasting
            # if lamb.ndim == 0: log_add_exp_term = torch.log(torch.exp(-2. * lamb) + 1.)
            # else: log_add_exp_term = torch.logaddexp(-2. * lamb, torch.zeros_like(lamb)) # More robust way

            tmp = 2. * (self.beta_1 - self.beta_0) * log_add_exp_term
            delta_sqrt_arg = self.beta_0**2 + tmp
            # Clamp to avoid sqrt of negative numbers
            delta_sqrt_arg = torch.clamp(delta_sqrt_arg, min=1e-8)
            sqrt_delta = torch.sqrt(delta_sqrt_arg)

            # Avoid division by zero if beta_1 == beta_0
            denominator_factor = self.beta_1 - self.beta_0
            if abs(denominator_factor) < 1e-8: # Practically zero
                # This case implies beta_0 = beta_1, which means beta is constant.
                # The formula for t simplifies or might need special handling.
                # For now, returning NaN or raising error as it's an edge case.
                # If beta_0 = beta_1 != 0, then log_alpha_t = -0.5 * t * beta_0
                # lamb = log_alpha_t - 0.5 * log(1 - exp(2*log_alpha_t))
                # This inversion is more complex. The original code likely assumes beta_1 != beta_0
                # For simplicity, if beta_1 == beta_0, we might treat it as no change or return a specific value.
                # Let's assume beta_1 != beta_0 as per original code structure.
                if abs(self.beta_1 - self.beta_0) < 1e-8:
                    # Fallback for beta_1 == beta_0, e.g., t = -2 * log_alpha / self.beta_0.
                    # This simplified version assumes beta_0 !=0.
                    # log_alpha = -0.5 * torch.logaddexp(torch.zeros_like(lamb), -2. * lamb)
                    # return -2. * log_alpha / self.beta_0 # This is an approximation, true form is complex
                    # For now, stick to the original structure, assuming beta_1 != beta_0 implies non-zero denominator_factor
                    # This case might indicate an issue with parameters if beta_1 is very close to beta_0.
                     return torch.full_like(lamb, float('nan')) # Or handle as per specific model needs

            return tmp / (sqrt_delta + self.beta_0) / denominator_factor

        elif self.schedule == 'discrete':
            # Original JAX:
            # log_alpha = -0.5 * jnp.logaddexp(jnp.zeros((1,)), -2. * lamb)
            # t = interpolate_fn(log_alpha.reshape((-1, 1)), jnp.flip(self.log_alpha_array, [1]), jnp.flip(self.t_array, [1]))
            # return t.reshape((-1,))
            # PyTorch:
            # log_alpha = -0.5 * torch.logaddexp(torch.zeros_like(lamb), -2. * lamb)
            log_alpha = -0.5 * (torch.log(torch.exp(torch.zeros_like(lamb)) + torch.exp(-2. * lamb)))

            # Ensure arrays for interpolation are 1D for this interpolate_fn
            # Original flips entire array content, then reshapes.
            # torch.flip flips along specified dimensions.
            # self.log_alpha_array and self.t_array are (1, M)
            flipped_log_alpha_array = torch.flip(self.log_alpha_array.squeeze(), dims=[0])
            flipped_t_array = torch.flip(self.t_array.squeeze(), dims=[0])

            # interpolate_fn expects xp to be sorted.
            # If self.log_alpha_array was originally increasing, flipping makes it decreasing.
            # The `interpolate_fn` needs to be robust to this or data needs to be sorted.
            # Assuming interpolate_fn can handle it or log_alpha_array (after flip) is sorted as expected for xp.
            # Check if flipped_log_alpha_array is sorted (it should be if original log_alpha_array was monotonic)
            # For safety, let's sort if necessary, though flip should maintain order if original was sorted.
            # The DPM paper implies log_alpha_t is decreasing, so log_alpha_array is decreasing.
            # Flipping it makes it increasing, which is good for `searchsorted`.

            t_interpolated = interpolate_fn(log_alpha.reshape((-1,1)),
                                            flipped_log_alpha_array.unsqueeze(-1), # Make it (M,1)
                                            flipped_t_array.unsqueeze(-1))       # Make it (M,1)
            return t_interpolated.reshape(-1)
        else: # cosine
            # Original JAX:
            # log_alpha = -0.5 * jnp.logaddexp(-2. * lamb, jnp.zeros((1,)))
            # t_fn = lambda log_alpha_t: jnp.arccos(jnp.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            # t = t_fn(log_alpha)
            # return t
            # PyTorch:
            # log_alpha = -0.5 * torch.logaddexp(-2. * lamb, torch.zeros_like(lamb))
            log_alpha = -0.5 * (torch.log(torch.exp(-2. * lamb) + torch.exp(torch.zeros_like(lamb))))

            # Argument for arccos needs to be in [-1, 1]
            arg_arccos = torch.exp(log_alpha + self.cosine_log_alpha_0)
            # Clamp the argument to avoid NaNs from acos due to precision errors
            arg_arccos_clamped = torch.clamp(arg_arccos, -1.0 + 1e-7, 1.0 - 1e-7)

            t_val = torch.acos(arg_arccos_clamped) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            return t_val

# class Cosine:
#     def __init__(self):
#         self.schedule = NoiseScheduleVP(schedule='cosine')

#     def alpha_in(self, t):
#         # noise rate
#         #return torch.sqrt(1 - self.schedule.marginal_alpha(t)**2)
#         return 1 - self.schedule.marginal_alpha(t) ** 2

#     def gamma_in(self, t):
#         # return self.schedule.marginal_alpha(t)
#         return self.schedule.marginal_alpha(t) ** 2

#     # These are the "hat" versions, defining the target for the model F_theta.
#     # If F_theta predicts noise (epsilon), then z_target = 1*epsilon + 0*x_0.
#     def alpha_to(self, t_continuous):
#         """
#         UCGM's target coefficient \hat{\alpha}(t).
#         For a DDPM predicting noise (epsilon, which is UCGM's 'z'), this is 1.
#         t_continuous: Tensor, time scaled to [0, 1]. (Argument kept for consistency)
#         """
#         return 1

#     def gamma_to(self, t_continuous):
#         """
#         UCGM's target coefficient \hat{\gamma}(t).
#         For a DDPM predicting noise, this is 0.
#         t_continuous: Tensor, time scaled to [0, 1]. (Argument kept for consistency)
#         """
#         return -1


class Cosine:
    def __init__(self, s: float = 0.008):
        self.s = s
        # w = π/2 / (1+s)
        self.w = math.pi / (2 * (1 + self.s))

    def alpha_in(self, t: torch.Tensor) -> torch.Tensor:
        """
        α(t) = sqrt(bar_alpha(t)), 
        bar_alpha(t) = cos(w*(t + s))
        """
        bar = torch.cos(self.w * (t + self.s))
        return torch.sqrt(bar)

    def gamma_in(self, t: torch.Tensor) -> torch.Tensor:
        """
        γ(t) = sqrt(1 - bar_alpha(t))
        """
        bar = torch.cos(self.w * (t + self.s))
        return torch.sqrt(1 - bar)

    def alpha_to(self, t: torch.Tensor) -> torch.Tensor:
        """
        α'(t) = d/dt sqrt(bar_alpha(t))
              = bar_alpha'(t) / (2 * sqrt(bar_alpha(t)))
        where bar_alpha'(t) = -w * sin(w*(t+s))
        """
        bar = torch.cos(self.w * (t + self.s))
        bar_grad = -self.w * torch.sin(self.w * (t + self.s))
        return bar_grad / (2 * torch.sqrt(bar))

    def gamma_to(self, t: torch.Tensor) -> torch.Tensor:
        """
        γ'(t) = d/dt sqrt(1 - bar_alpha(t))
              = -bar_alpha'(t) / (2 * sqrt(1 - bar_alpha(t)))
        """
        bar = torch.cos(self.w * (t + self.s))
        bar_grad = -self.w * torch.sin(self.w * (t + self.s))
        return -bar_grad / (2 * torch.sqrt(1 - bar))