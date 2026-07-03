import argparse
from .karras_diffusion import KarrasDenoiser
from .unet import UNetModel
import numpy as np
NUM_CLASSES = 1000

def cm_train_defaults():
    return dict(teacher_model_path='', teacher_dropout=0.1, training_mode='consistency_distillation', target_ema_mode='fixed', scale_mode='fixed', total_training_steps=200000, start_ema=0.0, start_scales=40, end_scales=40, distill_steps_per_iter=50000)

def model_and_diffusion_defaults():
    res = dict(patch_size='4,8,16', sigma_min=0.002, sigma_max=80.0, num_channels=128, num_res_blocks=2, num_heads=4, num_heads_upsample=-1, num_head_channels=-1, attention_resolutions='32,16,8', channel_mult='', dropout=0.3, class_cond=False, use_checkpoint=False, use_scale_shift_norm=True, resblock_updown=False, use_fp16=False, use_new_attention_order=False, learn_sigma=False, weight_schedule='karras', loss_norm='lpips', ensemble_size=2)
    return res

def create_model_and_diffusion(patch_size, class_cond, learn_sigma, num_channels, num_res_blocks, channel_mult, num_heads, num_head_channels, num_heads_upsample, attention_resolutions, dropout, use_checkpoint, use_scale_shift_norm, resblock_updown, use_fp16, use_new_attention_order, weight_schedule, loss_norm='lpips', ensemble_size=2, sigma_min=0.002, sigma_max=80.0, distillation=False):
    model = create_model(patch_size, num_channels, num_res_blocks, channel_mult=channel_mult, learn_sigma=learn_sigma, class_cond=class_cond, use_checkpoint=use_checkpoint, attention_resolutions=attention_resolutions, num_heads=num_heads, num_head_channels=num_head_channels, num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm, dropout=dropout, resblock_updown=resblock_updown, use_fp16=use_fp16, use_new_attention_order=use_new_attention_order)
    diffusion = KarrasDenoiser(sigma_data=0.5, sigma_max=sigma_max, sigma_min=sigma_min, distillation=distillation, weight_schedule=weight_schedule, loss_norm=loss_norm, ensemble_size=ensemble_size)
    return (model, diffusion)

def create_model(patch_size, num_channels, num_res_blocks, channel_mult='1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6', learn_sigma=False, class_cond=False, use_checkpoint=False, attention_resolutions='16', num_heads=1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, dropout=0, resblock_updown=False, use_fp16=False, use_new_attention_order=False):
    channel_mult = tuple((float(ch_mult) for ch_mult in channel_mult.split(',')))
    attention_ds = []
    for res in attention_resolutions.split(','):
        attention_ds.append(768 // int(res))
    return UNetModel(patch_size=patch_size, in_channels=1, model_channels=num_channels, out_channels=1, num_res_blocks=num_res_blocks, attention_resolutions=tuple(attention_ds), dropout=dropout, channel_mult=channel_mult, num_classes=NUM_CLASSES if class_cond else None, use_checkpoint=use_checkpoint, use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels, num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm, resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order)

def create_ema_and_scales_fn(target_ema_mode, start_ema, scale_mode, start_scales, end_scales, total_steps, distill_steps_per_iter):

    def ema_and_scales_fn(step):
        if target_ema_mode == 'fixed' and scale_mode == 'fixed':
            target_ema = start_ema
            scales = start_scales
        elif target_ema_mode == 'fixed' and scale_mode == 'progressive':
            target_ema = start_ema
            scales = np.ceil(np.sqrt(step / total_steps * ((end_scales + 1) ** 2 - start_scales ** 2) + start_scales ** 2) - 1).astype(np.int32)
            scales = np.maximum(scales, 1)
            scales = scales + 1
        elif target_ema_mode == 'adaptive' and scale_mode == 'progressive':
            scales = np.ceil(np.sqrt(step / total_steps * ((end_scales + 1) ** 2 - start_scales ** 2) + start_scales ** 2) - 1).astype(np.int32)
            scales = np.maximum(scales, 1)
            c = -np.log(start_ema) * start_scales
            target_ema = np.exp(-c / scales)
            scales = scales + 1
        elif target_ema_mode == 'fixed' and scale_mode == 'progdist':
            distill_stage = step // distill_steps_per_iter
            scales = start_scales // 2 ** distill_stage
            scales = np.maximum(scales, 2)
            sub_stage = np.maximum(step - distill_steps_per_iter * (np.log2(start_scales) - 1), 0)
            sub_stage = sub_stage // (distill_steps_per_iter * 2)
            sub_scales = 2 // 2 ** sub_stage
            sub_scales = np.maximum(sub_scales, 1)
            scales = np.where(scales == 2, sub_scales, scales)
            target_ema = 1.0
        else:
            raise NotImplementedError
        return (float(target_ema), int(scales))
    return ema_and_scales_fn

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f'--{k}', default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected')
