import bisect
import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

def _dist_info():
    if dist.is_available() and dist.is_initialized():
        return (dist.get_rank(), dist.get_world_size())
    return (0, 1)

def _list_nc_files(data_dir):
    if not os.path.exists(data_dir):
        raise ValueError(f'Directory does not exist: {data_dir}')
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nc')]
    files = sorted(files)
    if len(files) == 0:
        raise ValueError(f'No .nc files found in directory: {data_dir}')
    return files

def _list_npy_files(data_dir):
    if not os.path.exists(data_dir):
        raise ValueError(f'Directory does not exist: {data_dir}')
    files = sorted((os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')))
    if len(files) == 0:
        raise ValueError(f'No .npy files found in directory: {data_dir}')
    return files

class UniversalDataset(Dataset):

    def __init__(self, data_paths, cond_paths=None, data_type='npy', cond_type='npy', var_name='sfcWind'):
        super().__init__()
        assert data_type in ('nc', 'npy')
        if cond_paths is not None:
            assert cond_type in ('nc', 'npy')
        self.data_type = data_type
        self.cond_type = cond_type
        self.var_name = var_name
        self.has_lr = cond_paths is not None
        self.hr_files = data_paths
        self.hr_idx_map = None
        self.hr_lens = None
        self.lr_files = cond_paths
        self.lr_idx_map = None
        self.lr_lens = None
        if self.data_type == 'npy':
            self.hr_idx_map, self.hr_total = self._build_npy_index(self.hr_files)
        else:
            self.hr_idx_map, self.hr_total = self._build_nc_index(self.hr_files, self.var_name)
        if self.has_lr:
            if self.cond_type == 'npy':
                self.lr_idx_map, self.lr_total = self._build_npy_index(self.lr_files)
            else:
                self.lr_idx_map, self.lr_total = self._build_nc_index(self.lr_files, self.var_name)
        else:
            self.lr_total = 0

    def __len__(self):
        if self.has_lr:
            return min(self.hr_total, self.lr_total)
        return self.hr_total

    @staticmethod
    def _build_npy_index(files):
        idx_map = []
        total = 0
        for f in files:
            mm = np.load(f, mmap_mode='r')
            total += mm.shape[0]
            idx_map.append(total)
        if total == 0:
            raise ValueError('The npy dataset is empty.')
        return (idx_map, total)

    @staticmethod
    def _build_nc_index(files, var_name):
        idx_map = []
        total = 0
        for f in files:
            with xr.open_dataset(f, engine='netcdf4', decode_cf=False, mask_and_scale=False, cache=False) as ds:
                if var_name not in ds:
                    raise ValueError(f'Variable {var_name}: {f}')
                tlen = int(ds[var_name].sizes.get('time', ds[var_name].shape[0]))
            total += tlen
            idx_map.append(total)
        if total == 0:
            raise ValueError('The nc dataset is empty or has an invalid variable dimension.')
        return (idx_map, total)

    @staticmethod
    def _locate(idx_map, gidx):
        pos = bisect.bisect_right(idx_map, gidx)
        prev = 0 if pos == 0 else idx_map[pos - 1]
        row = gidx - prev
        return (pos, row)

    def _fetch_npy(self, files, idx_map, idx):
        fpos, row = self._locate(idx_map, idx)
        mm = np.load(files[fpos], mmap_mode='r')
        arr = mm[row]
        return torch.tensor(arr[None], dtype=torch.float32)

    def _fetch_nc(self, files, idx_map, idx):
        fpos, row = self._locate(idx_map, idx)
        path = files[fpos]
        with xr.open_dataset(path, engine='netcdf4') as ds:
            frame = ds[self.var_name].isel(time=row).astype('float32').load().values
        return torch.from_numpy(frame[None]).float()

    def _fetch(self, side, idx):
        if side == 'hr':
            if self.data_type == 'npy':
                return self._fetch_npy(self.hr_files, self.hr_idx_map, idx)
            else:
                return self._fetch_nc(self.hr_files, self.hr_idx_map, idx)
        else:
            if not self.has_lr:
                return torch.tensor([])
            if self.cond_type == 'npy':
                return self._fetch_npy(self.lr_files, self.lr_idx_map, idx)
            else:
                return self._fetch_nc(self.lr_files, self.lr_idx_map, idx)

    def __getitem__(self, idx):
        hr = self._fetch('hr', idx)
        lr = self._fetch('lr', idx) if self.has_lr else torch.tensor([])
        return (hr, lr)

def load_data(*, data_dir, cond_dir=None, batch_size, deterministic=False, data_type='npy', cond_type='npy', var_name='sfcWind', num_workers=2, drop_last=True, pin_memory=True):
    assert data_type in ('nc', 'npy')
    if cond_dir is not None:
        assert cond_type in ('nc', 'npy')
    if data_type == 'nc':
        hr_files = _list_nc_files(data_dir)
    else:
        hr_files = _list_npy_files(data_dir)
    lr_files = None
    if cond_dir is not None:
        if cond_type == 'nc':
            lr_files = _list_nc_files(cond_dir)
        else:
            lr_files = _list_npy_files(cond_dir)
    dataset = UniversalDataset(data_paths=hr_files, cond_paths=lr_files, data_type=data_type, cond_type=cond_type, var_name=var_name)
    rank, world = _dist_info()
    sampler = None
    if world > 1:
        num_workers = 0
        pin_memory = False
    if world > 1:
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=not deterministic, drop_last=drop_last)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=sampler is None and (not deterministic), sampler=sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, persistent_workers=num_workers > 0)
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1
