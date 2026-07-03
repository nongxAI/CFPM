import random
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from piq import LPIPS
from torchvision.transforms import RandomCrop
from . import dist_util
from .nn import mean_flat, append_dims, append_zero
from .random_util import get_generator
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def _psnr_ssim_field(pred, gt, data_range=None):
    pred = pred.detach().cpu().float().numpy()
    gt = gt.detach().cpu().float().numpy()
    if pred.ndim == 3:
        pred = pred[:, None, ...]
    if gt.ndim == 3:
        gt = gt[:, None, ...]
    N, C, H, W = pred.shape
    psnr_sum, ssim_sum = (0.0, 0.0)
    for n in range(N):
        _range = data_range
        if _range is None:
            _range = float(gt[n].max() - gt[n].min() + 1e-08)
        for c in range(C):
            x = pred[n, c]
            y = gt[n, c]
            psnr_sum += peak_signal_noise_ratio(y, x, data_range=_range)
            ssim_sum += structural_similarity(y, x, data_range=_range, gaussian_weights=True)
    total = N * C
    return (psnr_sum / total, ssim_sum / total)

def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == 'snr':
        weightings = snrs
    elif weight_schedule == 'snr+1':
        weightings = snrs + 1
    elif weight_schedule == 'karras':
        weightings = snrs + 1.0 / sigma_data ** 2
    elif weight_schedule == 'truncated-snr':
        weightings = th.clamp(snrs, min=1.0)
    elif weight_schedule == 'uniform':
        weightings = th.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings

class KarrasDenoiser:

    def __init__(self, sigma_data: float=6.13, sigma_max=80.0, sigma_min=0.002, rho=7.0, weight_schedule='karras', distillation=False, loss_norm='lpips', ensemble_size=2):
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.distillation = distillation
        self.loss_norm = loss_norm
        self.ensemble_size = max(int(ensemble_size), 1)
        if loss_norm == 'lpips':
            self.lpips_loss = LPIPS(replace_pooling=True, reduction='none')
        self.rho = rho
        self.num_timesteps = 40

    def _maybe_resize_for_lpips(self, x):
        if x.shape[-1] < 256:
            return F.interpolate(x, size=224, mode='bilinear')
        return x

    def _per_sample_distance(self, pred, target, loss_norm):
        if loss_norm == 'l1':
            diffs = th.abs(pred - target)
            return mean_flat(diffs)
        if loss_norm == 'l2':
            diffs = (pred - target) ** 2
            return mean_flat(diffs)
        if loss_norm == 'l2-32':
            pred_32 = F.interpolate(pred, size=32, mode='bilinear')
            target_32 = F.interpolate(target, size=32, mode='bilinear')
            diffs = (pred_32 - target_32) ** 2
            return mean_flat(diffs)
        if loss_norm == 'lpips':
            pred = self._maybe_resize_for_lpips(pred)
            target = self._maybe_resize_for_lpips(target)
            lpips_out = self.lpips_loss((pred + 1) / 2.0, (target + 1) / 2.0)
            return mean_flat(lpips_out)
        raise ValueError(f'Unknown loss norm {loss_norm}')

    def get_snr(self, sigmas):
        sigmas = sigmas.float()
        return sigmas ** (-2)

    def get_sigmas(self, sigmas):
        return sigmas

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return (c_skip, c_out, c_in)

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data ** 2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data ** 2)
        c_out = (sigma - self.sigma_min) * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return (c_skip, c_out, c_in)

    def training_losses(self, model, x_start, sigmas, cond=None, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        terms = {}
        dims = x_start.ndim
        x_t = x_start + noise * append_dims(sigmas, dims)
        model_output, denoised = self.denoise(model, x_t, sigmas, cond=cond, **model_kwargs)
        snrs = self.get_snr(sigmas)
        weights = append_dims(get_weightings(self.weight_schedule, snrs, self.sigma_data), dims)
        terms['xs_mse'] = mean_flat((denoised - x_start) ** 2)
        terms['mse'] = mean_flat(weights * (denoised - x_start) ** 2)
        if 'vb' in terms:
            terms['loss'] = terms['mse'] + terms['vb']
        else:
            terms['loss'] = terms['mse']
        terms['pred_xstart'] = denoised.detach()
        return terms

    def consistency_losses(self, model, x_start, num_scales, target_model=None, teacher_model=None, teacher_diffusion=None, cond=None, noise=None, model_kwargs=None, global_step=0, total_training_steps=0):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        dims = x_start.ndim
        global_step = global_step

        def denoise_fn(x, t):
            return self.denoise(model, x, t, cond=cond, **model_kwargs)[1]
        if target_model:

            @th.no_grad()
            def target_denoise_fn(x, t):
                return self.denoise(target_model, x, t, cond=cond, **model_kwargs)[1]
        else:
            raise NotImplementedError('Must have a target model')
        if teacher_model:

            @th.no_grad()
            def teacher_denoise_fn(x, t):
                return teacher_diffusion.denoise(teacher_model, x, t, cond=cond, **model_kwargs)[1]

        @th.no_grad()
        def heun_solver(samples, t, next_t, x0):
            x = samples
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(x, t)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(samples, next_t)
            next_d = (samples - denoiser) / append_dims(next_t, dims)
            samples = x + (d + next_d) * append_dims((next_t - t) / 2, dims)
            return samples

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(x, t)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            return samples
        indices = th.randint(0, num_scales - 1, (x_start.shape[0],), device=x_start.device)
        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        t = t ** self.rho
        progress = min(global_step / total_training_steps, 1.0)
        max_step = 40
        min_step = 10
        step_size = int(max_step - progress * (max_step - min_step))
        target_indices = indices + step_size
        target_indices = target_indices.clamp(max=num_scales - 1)
        t2 = self.sigma_max ** (1 / self.rho) + (target_indices + 1) / (num_scales - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        t2 = t2 ** self.rho
        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
        ensemble_preds = []
        ensemble_targets = []
        for i in range(self.ensemble_size):
            noise_i = noise if noise is not None and i == 0 else th.randn_like(x_start)
            x_t = x_start + noise_i * append_dims(t, dims)
            dropout_state = th.get_rng_state()
            distiller = denoise_fn(x_t, t)
            if teacher_model is None:
                x_t2 = euler_solver(x_t, t, t2, x_start).detach()
            else:
                x_t2 = heun_solver(x_t, t, t2, x_start).detach()
            th.set_rng_state(dropout_state)
            distiller_target = target_denoise_fn(x_t2, t2).detach()
            ensemble_preds.append(distiller)
            ensemble_targets.append(distiller_target)
        preds = th.stack(ensemble_preds, dim=0)
        targets = th.stack(ensemble_targets, dim=0)
        base_losses = []
        for i in range(self.ensemble_size):
            base_losses.append(self._per_sample_distance(preds[i], targets[i], self.loss_norm))
        base_loss = th.stack(base_losses, dim=0).mean(dim=0)
        loss = base_loss * weights
        terms = {}
        terms['loss'] = loss
        distiller_mean = preds.mean(dim=0)
        terms['xs_mse'] = mean_flat((distiller_mean - x_start) ** 2)
        terms['xs_R2'] = 1 - (distiller_mean - x_start).pow(2).mean() / ((x_start - x_start.mean()).pow(2).mean() + 1e-08)
        terms['pred_xstart'] = distiller_mean.detach()
        terms['base_loss'] = base_loss
        terms['sigma_t'] = t
        return terms

    def progdist_losses(self, model, x_start, num_scales, model_kwargs=None, teacher_model=None, teacher_diffusion=None, cond=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        dims = x_start.ndim

        def denoise_fn(x, t):
            return self.denoise(model, x, t, cond=cond, **model_kwargs)[1]

        @th.no_grad()
        def teacher_denoise_fn(x, t):
            return teacher_diffusion.denoise(teacher_model, x, t, cond=cond, **model_kwargs)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t):
            x = samples
            denoiser = teacher_denoise_fn(x, t)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            return samples

        @th.no_grad()
        def euler_to_denoiser(x_t, t, x_next_t, next_t):
            denoiser = x_t - append_dims(t, dims) * (x_next_t - x_t) / append_dims(next_t - t, dims)
            return denoiser
        indices = th.randint(0, num_scales, (x_start.shape[0],), device=x_start.device)
        t = self.sigma_max ** (1 / self.rho) + indices / num_scales * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        t = t ** self.rho
        t2 = self.sigma_max ** (1 / self.rho) + (indices + 0.5) / num_scales * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        t2 = t2 ** self.rho
        t3 = self.sigma_max ** (1 / self.rho) + (indices + 1) / num_scales * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        t3 = t3 ** self.rho
        x_t = x_start + noise * append_dims(t, dims)
        denoised_x = denoise_fn(x_t, t)
        x_t2 = euler_solver(x_t, t, t2).detach()
        x_t3 = euler_solver(x_t2, t2, t3).detach()
        target_x = euler_to_denoiser(x_t, t, x_t3, t3).detach()
        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
        ensemble_preds = []
        ensemble_targets = []
        for i in range(self.ensemble_size):
            noise_i = noise if noise is not None and i == 0 else th.randn_like(x_start)
            x_t = x_start + noise_i * append_dims(t, dims)
            denoised_x = denoise_fn(x_t, t)
            x_t2 = euler_solver(x_t, t, t2).detach()
            x_t3 = euler_solver(x_t2, t2, t3).detach()
            target_x = euler_to_denoiser(x_t, t, x_t3, t3).detach()
            ensemble_preds.append(denoised_x)
            ensemble_targets.append(target_x)
        preds = th.stack(ensemble_preds, dim=0)
        targets = th.stack(ensemble_targets, dim=0)
        base_losses = []
        for i in range(self.ensemble_size):
            base_losses.append(self._per_sample_distance(preds[i], targets[i], self.loss_norm))
        base_loss = th.stack(base_losses, dim=0).mean(dim=0)
        loss = base_loss * weights
        terms = {}
        terms['loss'] = loss
        terms['base_loss'] = base_loss
        return terms

    def denoise(self, model, x_t, sigmas, cond=None, **model_kwargs):
        import torch.distributed as dist
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        c_skip, c_out, c_in = [append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)]
        model_output = model(c_in * x_t, rescaled_t, cond=cond, **model_kwargs)
        denoised = c_out * model_output + c_skip * x_t
        return (model_output, denoised)

def karras_sample(diffusion, model, shape, steps, clip_denoised=False, progress=False, callback=None, model_kwargs=None, device=None, sigma_min=0.002, sigma_max=80, rho=7.0, sampler='heun', s_churn=0.0, s_tmin=0.0, s_tmax=float('inf'), s_noise=1.0, generator=None, ts=None, cond=None, extra_sigma=None):
    if generator is None:
        generator = get_generator('dummy')
    if sampler == 'progdist':
        sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, rho, device=device)
    else:
        sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)
    x_T = generator.randn(*shape, device=device) * sigma_max
    sample_fn = {'heun': sample_heun, 'dpm': sample_dpm, 'ancestral': sample_euler_ancestral, 'onestep': sample_onestep, 'progdist': sample_progdist, 'euler': sample_euler, 'multistep': stochastic_iterative_sampler, 'twostep': sample_twostep}[sampler]
    if sampler in ['heun', 'dpm']:
        sampler_args = dict(s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise)
    elif sampler == 'multistep':
        sampler_args = dict(ts=ts, t_min=sigma_min, t_max=sigma_max, rho=diffusion.rho, steps=steps)
    elif sampler == 'twostep':
        extra_sigma = 4.0
        sampler_args = dict(extra_sigma=extra_sigma)
    else:
        sampler_args = {}

    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model, x_t, sigma, cond=cond, **model_kwargs)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised
    x_0 = sample_fn(denoiser, x_T, sigmas, generator, progress=progress, callback=callback, **sampler_args)
    return x_0

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device='cpu'):
    ramp = th.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)

def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)

def get_ancestral_step(sigma_from, sigma_to):
    sigma_up = (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return (sigma_down, sigma_up)

@th.no_grad()
def sample_euler_ancestral(model, x, sigmas, generator, progress=False, callback=None):
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)
    for i in indices:
        denoised = model(x, sigmas[i] * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + generator.randn_like(x) * sigma_up
    return x

@th.no_grad()
def sample_midpoint_ancestral(model, x, ts, generator, progress=False, callback=None):
    s_in = x.new_ones([x.shape[0]])
    step_size = 1 / len(ts)
    if progress:
        from tqdm.auto import tqdm
        ts = tqdm(ts)
    for tn in ts:
        dn = model(x, tn * s_in)
        dn_2 = model(x + step_size / 2 * dn, (tn + step_size / 2) * s_in)
        x = x + step_size * dn_2
        if callback is not None:
            callback({'x': x, 'tn': tn, 'dn': dn, 'dn_2': dn_2})
    return x

@th.no_grad()
def sample_heun(denoiser, x, sigmas, generator, progress=False, callback=None, s_churn=0.0, s_tmin=0.0, s_tmax=float('inf'), s_noise=1.0):
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)
    for i in indices:
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            x = x + d * dt
        else:
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x

@th.no_grad()
def sample_euler(denoiser, x, sigmas, generator, progress=False, callback=None):
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)
    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'denoised': denoised})
        dt = sigmas[i + 1] - sigma
        x = x + d * dt
    return x

@th.no_grad()
def sample_dpm(denoiser, x, sigmas, generator, progress=False, callback=None, s_churn=0.0, s_tmin=0.0, s_tmax=float('inf'), s_noise=1.0):
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)
    for i in indices:
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[i + 1] - sigma_hat
        x_2 = x + d * dt_1
        denoised_2 = denoiser(x_2, sigma_mid * s_in)
        d_2 = to_d(x_2, sigma_mid, denoised_2)
        x = x + d_2 * dt_2
    return x

@th.no_grad()
def sample_onestep(distiller, x, sigmas, generator=None, progress=False, callback=None):
    s_in = x.new_ones([x.shape[0]])
    return distiller(x, sigmas[0] * s_in)

@th.no_grad()
def sample_twostep(distiller, x, sigmas, generator=None, progress=False, callback=None, extra_sigma=None):
    s_in = x.new_ones([x.shape[0]])
    x0 = distiller(x, sigmas[0] * s_in)
    return distiller(x0, extra_sigma * s_in)

@th.no_grad()
def stochastic_iterative_sampler(distiller, x, sigmas, generator, ts, progress=False, callback=None, t_min=0.002, t_max=80.0, rho=7.0, steps=40):
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t ** 2 - t_min ** 2)
    return x

@th.no_grad()
def sample_progdist(denoiser, x, sigmas, generator=None, progress=False, callback=None):
    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas[:-1]
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)
    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma
        x = x + d * dt
    return x
