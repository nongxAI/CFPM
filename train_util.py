import copy
import functools
import os
import blobfile as bf
import torch as th
import torch.distributed as dist
from .karras_diffusion import karras_sample
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .random_util import get_generator
from .resample import LossAwareSampler, UniformSampler
from .fp16_util import get_param_groups_and_shapes, make_master_params, master_params_to_model_params
import numpy as np
INITIAL_LOG_LOSS_SCALE = 20.0

def monitor_gradients(model, step):
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_stats[name] = {'grad_norm': grad_norm, 'grad_max': param.grad.data.abs().max().item(), 'grad_mean': param.grad.data.abs().mean().item(), 'has_nan': th.isnan(param.grad.data).any().item(), 'has_inf': th.isinf(param.grad.data).any().item()}
    sorted_layers = sorted(grad_stats.items(), key=lambda x: x[1]['grad_norm'], reverse=True)[:10]
    for name, stats in sorted_layers:
        pass
    problematic_layers = [(name, stats) for name, stats in grad_stats.items() if stats['has_nan'] or stats['has_inf']]
    if problematic_layers:
        for name, stats in problematic_layers[:5]:
            pass
    return grad_stats
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

class TrainLoop:

    def __init__(self, *, model, diffusion, data, val_data=None, val_interval=1000, batch_size, microbatch, lr, ema_rate, log_interval, save_interval, resume_checkpoint, use_fp16=False, fp16_scale_growth=0.001, schedule_sampler=None, weight_decay=0.0, lr_anneal_steps=0):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.val_data = val_data
        self.val_interval = val_interval
        self.best_mse = float('inf')
        self.dataset_length = None
        if hasattr(self.data, 'dataset'):
            self.dataset_length = len(data.dataset)
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.best_ssim = -1
        self.patience = 10
        self.processed_samples = 0
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else [float(x) for x in ema_rate.split(',')]
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.sync_cuda = th.cuda.is_available()
        self.latest_metrics = {}
        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(model=self.model, use_fp16=self.use_fp16, fp16_scale_growth=fp16_scale_growth)
        self.opt = RAdam(self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [self._load_ema_parameters(rate) for rate in self.ema_rate]
        else:
            self.ema_params = [copy.deepcopy(self.mp_trainer.master_params) for _ in range(len(self.ema_rate))]
        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(self.model, device_ids=[dist_util.dev()], output_device=dist_util.dev(), broadcast_buffers=False, bucket_cap_mb=128, find_unused_parameters=True)
        else:
            if dist.get_world_size() > 1:
                logger.warn('Distributed training requires CUDA. Gradients will not be synchronized properly!')
            self.use_ddp = False
            self.ddp_model = self.model
        self.step = self.resume_step
        self.global_step = self.step

    def save_best_model(self, metric, value):
        state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
        if dist.get_rank() == 0:
            filename = f'best_{metric}_{value:.6f}_model{self.global_step:06d}.pt'
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), 'wb') as f:
                th.save(state_dict, f)
            logger.log(f'Saved best main model: {filename}')

    @th.no_grad()
    def run_validation(self):
        self.model.eval()
        generator = get_generator('determ', 10, 42)
        psnr_sum, ssim_sum, mse_sum, n = (0.0, 0.0, 0.0, 0)
        val_steps = min(1, len(self.val_data))
        for _ in range(val_steps):
            hr, lr = next(self.val_data)
            hr, lr = (hr.to(dist_util.dev()), lr.to(dist_util.dev()))
            pred = karras_sample(self.diffusion, self.model, shape=hr.shape, steps=40, cond=lr, clip_denoised=False, sampler='heun', generator=generator, sigma_min=0.002, sigma_max=10, device=dist_util.dev(), model_kwargs={})
            psnr, ssim = _psnr_ssim_field(pred, hr)
            mse = th.mean((pred - hr) ** 2).item()
            bs = hr.size(0)
            psnr_sum += psnr * bs
            ssim_sum += ssim * bs
            mse_sum += mse * bs
            n += bs
        self.model.train()
        if dist.get_world_size() > 1:
            psnr_sum = th.tensor(psnr_sum).to(dist_util.dev())
            ssim_sum = th.tensor(ssim_sum).to(dist_util.dev())
            mse_sum = th.tensor(mse_sum).to(dist_util.dev())
            n = th.tensor(n).to(dist_util.dev())
            dist.all_reduce(psnr_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(ssim_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(mse_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(n, op=dist.ReduceOp.SUM)
            psnr = (psnr_sum / n).item()
            ssim = (ssim_sum / n).item()
            val_mse = (mse_sum / n).item()
        else:
            psnr = psnr_sum / n
            ssim = ssim_sum / n
            val_mse = mse_sum / n
        logger.logkv('val_psnr', psnr)
        logger.logkv('val_ssim', ssim)
        logger.logkv('val_mse', val_mse)
        logger.dumpkvs()
        if val_mse < self.best_mse:
            self.best_mse = val_mse
            if dist.get_rank() == 0:
                logger.log(f'Found best MSE model (current: {val_mse:.6f}, best: {self.best_mse:.6f}), saving...')
                self.save_best_model(metric='mse', value=val_mse)

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f'loading model from checkpoint: {resume_checkpoint}...')
                self.model.load_state_dict(dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev()))
        dist_util.sync_params(self.model.parameters())
        dist_util.sync_params(self.model.buffers())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f'loading EMA from checkpoint: {ema_checkpoint}...')
                state_dict = dist_util.load_state_dict(ema_checkpoint, map_location=dist_util.dev())
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint), f'opt{self.resume_step:06}.pt')
        if bf.exists(opt_checkpoint):
            logger.log(f'loading optimizer state from checkpoint: {opt_checkpoint}')
            state_dict = dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while not self.lr_anneal_steps or self.step < self.lr_anneal_steps:
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                if os.environ.get('DIFFUSION_TRAINING_TEST', '') and self.step > 0:
                    return
            if self.val_data and self.global_step % self.val_interval == 0 and (self.global_step > 0):
                if dist.get_rank() == 0:
                    self.run_validation()
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self.step += 1
            self.global_step += 1
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i:i + self.microbatch].to(dist_util.dev())
            micro_cond = cond[i:i + self.microbatch].to(dist_util.dev())
            last_batch = i + self.microbatch >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            compute_losses = functools.partial(self.diffusion.training_losses, self.ddp_model, micro, t, cond=micro_cond)
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses['loss'].detach())
            loss = (losses['loss'] * weights).mean()
            loss_dict_for_log = {}
            for k, v in losses.items():
                if k == 'pred_xstart':
                    continue
                else:
                    loss_dict_for_log[k] = v * weights
            log_loss_dict(self.diffusion, t, loss_dict_for_log)
            if 'pred_xstart' in losses:
                with th.no_grad():
                    pred_field = losses['pred_xstart']
                    gt_field = micro
                    psnr, ssim = _psnr_ssim_field(pred_field, gt_field, data_range=None)
                logger.logkv_mean('psnr', psnr)
                logger.logkv_mean('ssim', ssim)
                self.latest_metrics['psnr'] = psnr
                self.latest_metrics['ssim'] = ssim
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def log_step(self):
        logger.logkv('step', self.step + self.resume_step)
        logger.logkv('samples', (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):

        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f'saving model {rate}...')
                if not rate:
                    filename = f'model{self.step + self.resume_step:06d}.pt'
                else:
                    filename = f'ema_{rate}_{self.step + self.resume_step:06d}.pt'
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), 'wb') as f:
                    th.save(state_dict, f)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)
        if dist.get_rank() == 0:
            with bf.BlobFile(bf.join(get_blob_logdir(), f'opt{self.step + self.resume_step:06d}.pt'), 'wb') as f:
                th.save(self.opt.state_dict(), f)
        save_checkpoint(0, self.mp_trainer.master_params)
        dist.barrier()

class CMTrainLoop(TrainLoop):

    def __init__(self, *, target_model, teacher_model, teacher_diffusion, training_mode, ema_scale_fn, total_training_steps, val_data=None, val_interval=1000, **kwargs):
        super().__init__(**kwargs)
        self.training_mode = training_mode
        self.ema_scale_fn = ema_scale_fn
        self.target_model = target_model
        self.teacher_model = teacher_model
        self.teacher_diffusion = teacher_diffusion
        self.total_training_steps = total_training_steps
        self.val_data = val_data
        self.val_interval = val_interval
        self.latest_metrics = {}
        self.best_mse = float('inf')
        if target_model:
            self._load_and_sync_target_parameters()
            self.target_model.requires_grad_(False)
            self.target_model.train()
            self.target_model_param_groups_and_shapes = get_param_groups_and_shapes(self.target_model.named_parameters())
            self.target_model_master_params = make_master_params(self.target_model_param_groups_and_shapes)
        if teacher_model:
            self._load_and_sync_teacher_parameters()
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()
        self.global_step = self.step
        if training_mode == 'progdist':
            self.target_model.eval()
            _, scale = ema_scale_fn(self.global_step)
            if scale == 1 or scale == 2:
                _, start_scale = ema_scale_fn(0)
                n_normal_steps = int(np.log2(start_scale // 2)) * self.lr_anneal_steps
                step = self.global_step - n_normal_steps
                if step != 0:
                    self.lr_anneal_steps *= 2
                    self.step = step % self.lr_anneal_steps
                else:
                    self.step = 0
            else:
                self.step = self.global_step % self.lr_anneal_steps

    @th.no_grad()
    def run_validation(self):
        self.model.eval()
        generator = get_generator('dummy')
        psnr_sum, ssim_sum, R2_sum, mse_sum = (0.0, 0.0, 0.0, 0.0)
        ensemble_std_sum = 0.0
        n = 0
        val_steps = min(4, len(self.val_data))
        val_ensemble_size = max(1, self.diffusion.ensemble_size // 2) if hasattr(self.diffusion, 'ensemble_size') else 2
        for _ in range(val_steps):
            hr, lr = next(self.val_data)
            hr, lr = (hr.to(dist_util.dev()), lr.to(dist_util.dev()))
            ensemble_preds = []
            for i in range(val_ensemble_size):
                pred_i = karras_sample(self.diffusion, self.model, shape=hr.shape, steps=40, cond=lr, clip_denoised=False, sampler='onestep', generator=generator, sigma_min=0.002, sigma_max=self.diffusion.sigma_max, device=dist_util.dev(), model_kwargs={})
                ensemble_preds.append(pred_i)
            ensemble_preds = th.stack(ensemble_preds, dim=0)
            pred = ensemble_preds.mean(dim=0)
            pred_std = ensemble_preds.std(dim=0).mean().item()
            psnr, ssim = _psnr_ssim_field(pred, hr)
            bs = hr.size(0)
            R2 = 1 - (pred - hr).pow(2).mean() / ((hr - hr.mean()).pow(2).mean() + 1e-08)
            mse = th.mean((pred - hr) ** 2).item()
            psnr_sum += psnr * bs
            ssim_sum += ssim * bs
            R2_sum += R2 * bs
            mse_sum += mse * bs
            ensemble_std_sum += pred_std * bs
            n += bs
        self.model.train()
        if dist.get_world_size() > 1:
            psnr_sum = th.tensor(psnr_sum).to(dist_util.dev())
            ssim_sum = th.tensor(ssim_sum).to(dist_util.dev())
            R2_sum = th.tensor(R2_sum).to(dist_util.dev())
            mse_sum = th.tensor(mse_sum).to(dist_util.dev())
            ensemble_std_sum = th.tensor(ensemble_std_sum).to(dist_util.dev())
            n = th.tensor(n).to(dist_util.dev())
            dist.all_reduce(psnr_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(ssim_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(R2_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(mse_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(ensemble_std_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(n, op=dist.ReduceOp.SUM)
            psnr = (psnr_sum / n).item()
            ssim = (ssim_sum / n).item()
            R2 = (R2_sum / n).item()
            val_mse = (mse_sum / n).item()
            ensemble_std = (ensemble_std_sum / n).item()
        else:
            psnr = psnr_sum / n
            ssim = ssim_sum / n
            R2 = R2_sum / n
            val_mse = mse_sum / n
            ensemble_std = ensemble_std_sum / n
        logger.logkv('val_psnr', psnr)
        logger.logkv('val_ssim', ssim)
        logger.logkv('val_R2', R2)
        logger.logkv('val_mse', val_mse)
        logger.logkv('val_ensemble_std', ensemble_std)
        logger.dumpkvs()
        if val_mse < self.best_mse:
            self.best_mse = val_mse
            if dist.get_rank() == 0:
                logger.log(f'Found best MSE model (current: {val_mse:.6f}, best: {self.best_mse:.6f}), saving...')
                self.save_best_model(metric='mse', value=val_mse)

    def save_best_model(self, metric, value):
        state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
        if dist.get_rank() == 0:
            filename = f'best_{metric}_model.pt'
            logger.log(f'best{metric} model: value={value:.6f}(step={self.global_step}), overwriting previous file...')
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), 'wb') as f:
                th.save(state_dict, f)
            logger.log(f'best :{filename}')

    def _load_and_sync_target_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            path, name = os.path.split(resume_checkpoint)
            target_name = name.replace('model', 'target_model')
            resume_target_checkpoint = os.path.join(path, target_name)
            if bf.exists(resume_target_checkpoint) and dist.get_rank() == 0:
                logger.log('loading model from checkpoint: {resume_target_checkpoint}...')
                self.target_model.load_state_dict(dist_util.load_state_dict(resume_target_checkpoint, map_location=dist_util.dev()))
        dist_util.sync_params(self.target_model.parameters())
        dist_util.sync_params(self.target_model.buffers())

    def _load_and_sync_teacher_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            path, name = os.path.split(resume_checkpoint)
            teacher_name = name.replace('model', 'teacher_model')
            resume_teacher_checkpoint = os.path.join(path, teacher_name)
            if bf.exists(resume_teacher_checkpoint) and dist.get_rank() == 0:
                logger.log('loading model from checkpoint: {resume_teacher_checkpoint}...')
                self.teacher_model.load_state_dict(dist_util.load_state_dict(resume_teacher_checkpoint, map_location=dist_util.dev()))
        dist_util.sync_params(self.teacher_model.parameters())
        dist_util.sync_params(self.teacher_model.buffers())

    def run_loop(self):
        saved = False
        while not self.lr_anneal_steps or self.step < self.lr_anneal_steps or self.global_step < self.total_training_steps:
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.val_data and self.global_step % self.val_interval == 0 and (self.global_step > 0):
                self.run_validation()
            saved = False
            if self.global_step and self.save_interval != -1 and (self.global_step % self.save_interval == 0):
                self.save()
                saved = True
                th.cuda.empty_cache()
                if os.environ.get('DIFFUSION_TRAINING_TEST', '') and self.step > 0:
                    return
            if self.global_step % self.log_interval == 0:
                logger.dumpkvs()
        if not saved:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
            if self.target_model:
                self._update_target_ema()
            if self.training_mode == 'progdist':
                self.reset_training_for_progdist()
            self.step += 1
            self.global_step += 1
        self._anneal_lr()
        self.log_step()

    def _update_target_ema(self):
        target_ema, scales = self.ema_scale_fn(self.global_step)
        with th.no_grad():
            update_ema(self.target_model_master_params, self.mp_trainer.master_params, rate=target_ema)
            master_params_to_model_params(self.target_model_param_groups_and_shapes, self.target_model_master_params)

    def reset_training_for_progdist(self):
        assert self.training_mode == 'progdist', 'Training mode must be progdist'
        if self.global_step > 0:
            scales = self.ema_scale_fn(self.global_step)[1]
            scales2 = self.ema_scale_fn(self.global_step - 1)[1]
            if scales != scales2:
                with th.no_grad():
                    update_ema(self.teacher_model.parameters(), self.model.parameters(), 0.0)
                self.opt = RAdam(self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay)
                self.ema_params = [copy.deepcopy(self.mp_trainer.master_params) for _ in range(len(self.ema_rate))]
                if scales == 2:
                    self.lr_anneal_steps *= 2
                self.teacher_model.eval()
                self.step = 0

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i:i + self.microbatch].to(dist_util.dev())
            micro_cond = cond[i:i + self.microbatch].to(dist_util.dev())
            last_batch = i + self.microbatch >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            ema, num_scales = self.ema_scale_fn(self.global_step)
            if self.training_mode == 'progdist':
                if num_scales == self.ema_scale_fn(0)[1]:
                    compute_losses = functools.partial(self.diffusion.progdist_losses, self.ddp_model, micro, num_scales, target_model=self.teacher_model, target_diffusion=self.teacher_diffusion, cond=micro_cond)
                else:
                    compute_losses = functools.partial(self.diffusion.progdist_losses, self.ddp_model, micro, num_scales, target_model=self.target_model, target_diffusion=self.diffusion, cond=micro_cond)
            elif self.training_mode == 'consistency_distillation':
                compute_losses = functools.partial(self.diffusion.consistency_losses, self.ddp_model, micro, num_scales, target_model=self.target_model, teacher_model=self.teacher_model, teacher_diffusion=self.teacher_diffusion, cond=micro_cond)
            elif self.training_mode == 'consistency_training':
                compute_losses = functools.partial(self.diffusion.consistency_losses, self.ddp_model, micro, num_scales, target_model=self.target_model, cond=micro_cond, global_step=self.global_step, total_training_steps=max(self.total_training_steps, 1))
            else:
                raise ValueError(f'Unknown training mode {self.training_mode}')
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses['loss'].detach())
            loss = (losses['loss'] * weights).mean()
            loss_dict_for_log = {}
            for k, v in losses.items():
                if k == 'pred_xstart':
                    continue
                elif k == 'sigma_t':
                    continue
                else:
                    loss_dict_for_log[k] = v * weights
            if self.training_mode in ['consistency_training', 'consistency_distillation']:
                if 'sigma_t' in losses:
                    sigma_t = losses['sigma_t']
                    log_loss_dict(self.diffusion, sigma_t, loss_dict_for_log, use_sigma=True)
                else:
                    log_loss_dict(self.diffusion, t, loss_dict_for_log, use_sigma=False)
            else:
                log_loss_dict(self.diffusion, t, loss_dict_for_log, use_sigma=False)
            if 'pred_xstart' in losses:
                with th.no_grad():
                    pred_field = losses['pred_xstart']
                    gt_field = micro
                    psnr, ssim = _psnr_ssim_field(pred_field, gt_field, data_range=None)
                logger.logkv_mean('psnr', psnr)
                logger.logkv_mean('ssim', ssim)
                self.latest_metrics['psnr'] = psnr
                self.latest_metrics['ssim'] = ssim
            self.mp_trainer.backward(loss)

    def save(self):
        import blobfile as bf
        step = self.global_step

        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f'saving model {rate}...')
                if not rate:
                    filename = f'model{step:06d}.pt'
                else:
                    filename = f'ema_{rate}_{step:06d}.pt'
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), 'wb') as f:
                    th.save(state_dict, f)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)
        logger.log('saving optimizer state...')
        if dist.get_rank() == 0:
            with bf.BlobFile(bf.join(get_blob_logdir(), f'opt{step:06d}.pt'), 'wb') as f:
                th.save(self.opt.state_dict(), f)
        if dist.get_rank() == 0:
            if self.target_model:
                logger.log('saving target model state')
                filename = f'target_model{step:06d}.pt'
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), 'wb') as f:
                    th.save(self.target_model.state_dict(), f)
            if self.teacher_model and self.training_mode == 'progdist':
                logger.log('saving teacher model state')
                filename = f'teacher_model{step:06d}.pt'
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), 'wb') as f:
                    th.save(self.teacher_model.state_dict(), f)
        save_checkpoint(0, self.mp_trainer.master_params)
        if 'psnr' in getattr(self, 'latest_metrics', {}):
            logger.log(f"[save step={self.global_step}]  PSNR={self.latest_metrics['psnr']:.3f}  SSIM={self.latest_metrics['ssim']:.4f}")
        dist.barrier()

    def log_step(self):
        step = self.global_step
        logger.logkv('step', step)
        logger.logkv('samples', (step + 1) * self.global_batch)

def parse_resume_step_from_filename(filename):
    split = filename.split('model')
    if len(split) < 2:
        return 0
    split1 = split[-1].split('.')[0]
    try:
        return int(split1)
    except ValueError:
        return 0
    finally:
        pass

def get_blob_logdir():
    return logger.get_dir()

def find_resume_checkpoint():
    return None

def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f'ema_{rate}_{step:06d}.pt'
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

def log_loss_dict(diffusion, ts, losses, use_sigma=False):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            if use_sigma:
                sigma_range = diffusion.sigma_max - diffusion.sigma_min
                if sigma_range > 0:
                    normalized = 1.0 - (sub_t - diffusion.sigma_min) / sigma_range
                    normalized = max(0.0, min(1.0, normalized))
                    quartile = int(4 * normalized)
                    quartile = min(quartile, 3)
                else:
                    quartile = 0
            else:
                quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f'{key}_q{quartile}', sub_loss)
