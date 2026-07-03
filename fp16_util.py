import numpy as np
import torch as th
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from . import logger
INITIAL_LOG_LOSS_SCALE = 20.0

def convert_module_to_f16(l):
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()

def convert_module_to_f32(l):
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()

def make_master_params(param_groups_and_shapes):
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        master_param = nn.Parameter(_flatten_dense_tensors([param.detach().float() for _, param in param_group]).view(shape))
        master_param.requires_grad = True
        master_params.append(master_param)
    return master_params

def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    for master_param, (param_group, shape) in zip(master_params, param_groups_and_shapes):
        master_param.grad = _flatten_dense_tensors([param_grad_or_zeros(param) for _, param in param_group]).view(shape)

def master_params_to_model_params(param_groups_and_shapes, master_params):
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(param_group, unflatten_master_params(param_group, master_param.view(-1))):
            param.detach().copy_(unflat_master_param)

def unflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param for _, param in param_group])

def get_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params)
    scalar_vector_named_params = ([(n, p) for n, p in named_model_params if p.ndim <= 1], -1)
    matrix_named_params = ([(n, p) for n, p in named_model_params if p.ndim > 1], (1, -1))
    return [scalar_vector_named_params, matrix_named_params]

def master_params_to_state_dict(model, param_groups_and_shapes, master_params, use_fp16):
    if use_fp16:
        state_dict = model.state_dict()
        for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
            for (name, _), unflat_master_param in zip(param_group, unflatten_master_params(param_group, master_param.view(-1))):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    else:
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
    return state_dict

def state_dict_to_master_params(model, state_dict, use_fp16):
    if use_fp16:
        named_model_params = [(name, state_dict[name]) for name, _ in model.named_parameters()]
        param_groups_and_shapes = get_param_groups_and_shapes(named_model_params)
        master_params = make_master_params(param_groups_and_shapes)
    else:
        master_params = [state_dict[name] for name, _ in model.named_parameters()]
    return master_params

def zero_master_grads(master_params):
    for param in master_params:
        param.grad = None

def zero_grad(model_params):
    for param in model_params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

def param_grad_or_zeros(param):
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return th.zeros_like(param)

class MixedPrecisionTrainer:

    def __init__(self, *, model, use_fp16=False, fp16_scale_growth=1, growth_interval=200, initial_lg_loss_scale=20.0, min_lg_loss_scale=-5.0, max_lg_loss_scale=24.0):
        self.model = model
        self.use_fp16 = use_fp16
        self.growth_interval = growth_interval
        self.fp16_scale_growth = fp16_scale_growth
        self.min_lg_loss_scale = min_lg_loss_scale
        self.max_lg_loss_scale = max_lg_loss_scale
        self.lg_loss_scale = initial_lg_loss_scale
        self.success_steps = 0
        self.step = 0
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.param_groups_and_shapes = None
        if self.use_fp16:
            self.param_groups_and_shapes = get_param_groups_and_shapes(self.model.named_parameters())
            self.master_params = make_master_params(self.param_groups_and_shapes)
            self.model.convert_to_fp16()

    def zero_grad(self):
        zero_grad(self.model_params)

    def backward(self, loss: th.Tensor):
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()
        self._cached_true_loss = loss.detach()

    def optimize(self, opt: th.optim.Optimizer):
        if self.use_fp16:
            ok = self._optimize_fp16(opt)
        else:
            ok = self._optimize_normal(opt)
        self.step += 1
        return ok

    def _optimize_fp16(self, opt: th.optim.Optimizer):
        logger.logkv_mean('lg_loss_scale', self.lg_loss_scale)
        model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        found_inf = any([check_overflow(p.grad) for p in self.master_params if p.grad is not None])
        if found_inf:
            self._handle_overflow()
            return False
        grad_norm, param_norm = self._compute_norms(grad_scale=2 ** self.lg_loss_scale)
        logger.logkv_mean('grad_norm', grad_norm)
        logger.logkv_mean('param_norm', param_norm)
        scale_recip = 1.0 / 2 ** self.lg_loss_scale
        for p in self.master_params:
            if p.grad is not None:
                p.grad.mul_(scale_recip)
        opt.step()
        zero_master_grads(self.master_params)
        master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        self.success_steps += 1
        if self.success_steps >= self.growth_interval:
            self.lg_loss_scale = min(self.lg_loss_scale + self.fp16_scale_growth, self.max_lg_loss_scale)
            self.success_steps = 0
        return True

    def _handle_overflow(self):
        self.lg_loss_scale = max(self.lg_loss_scale - 1, self.min_lg_loss_scale)
        logger.log(f'Found NaN/Inf, decreased lg_loss_scale to {self.lg_loss_scale}')
        zero_master_grads(self.master_params)
        self.success_steps = 0

    def _optimize_normal(self, opt: th.optim.Optimizer):
        grad_norm, param_norm = self._compute_norms()
        logger.logkv_mean('grad_norm', grad_norm)
        logger.logkv_mean('param_norm', param_norm)
        opt.step()
        return True

    def _compute_norms(self, grad_scale=1.0):
        grad_norm_sq, param_norm_sq = (0.0, 0.0)
        for p in self.master_params:
            with th.no_grad():
                param_norm_sq += th.norm(p, 2, dtype=th.float32).item() ** 2
                if p.grad is not None:
                    grad_norm_sq += th.norm(p.grad, 2, dtype=th.float32).item() ** 2
        return (np.sqrt(grad_norm_sq) / grad_scale, np.sqrt(param_norm_sq))

    def master_params_to_state_dict(self, master_params):
        return master_params_to_state_dict(self.model, self.param_groups_and_shapes, master_params, self.use_fp16)

    def state_dict_to_master_params(self, state_dict):
        return state_dict_to_master_params(self.model, state_dict, self.use_fp16)

def check_overflow(value):
    import torch as th
    if isinstance(value, th.Tensor):
        return (th.isinf(value).any() | th.isnan(value).any()).item()
    else:
        return value == float('inf') or value == -float('inf') or value != value
