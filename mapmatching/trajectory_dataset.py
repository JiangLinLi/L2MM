import torch
from torch.utils.data import Dataset
import numpy as np
import sys
from torch._six import string_classes
IS_PYTHON2 = sys.version_info[0] < 3

if IS_PYTHON2:
    import collections
else:
    import collections.abc as collections


class Trajectory(Dataset):
    def __init__(self, all_data):
        super(Trajectory, self).__init__()
        self.BOS = 1    # change with dataset, Porto:1 CQ:input_cell_size-2
        self.EOS = 2
        self.interval = 10
        self.src = []
        self.trg = []
        for xy in all_data:
            x, y = xy[0], xy[1]
            down_data = self.down_data(x, self.interval)
            self.src.append(down_data)
            y = [self.BOS] + y + [self.EOS]
            self.trg.append(y)

    def down_data(self, data, interval):
        if len(data) < interval:
            return data
        elif (len(data) - 1) % interval == 0:
            index = np.arange(0, len(data), interval)
            line = np.arange(data)
            return line[index].tolist()
        elif (len(data) - 1) % interval != 0:
            index = np.arange(0, (int((len(data) - 1) / interval) + 1) * interval, interval)
            line = np.arange(data)
            tmp_res = np.append(line[index], line[-1])
            return tmp_res.tolist()


    def __getitem__(self, item):
        return self.src[item], self.trg[item]

    def __len__(self):
        return len(self.src)


def collate_fn(batch):
    batch.sort(key=lambda x:len(x[0]), reverse=True)
    src_len = [len(x) for x, _ in batch]
    trg_len = [len(y) for _, y in batch]
    max_src_len = max(src_len)
    max_trg_len = max(trg_len)

    batch_src = []
    batch_trg = []
    batch_mask = []
    for i, xy in enumerate(batch):
        x, y = xy
        x = x+[0]*(max_src_len-src_len[i])
        y = y+[0]*(max_trg_len-trg_len[i])
        mask = [1]*trg_len[i]+[0]*(max_trg_len-trg_len[i])
        batch_src.append(x)
        batch_trg.append(y)
        batch_mask.append(mask)

    batch_src = torch.LongTensor(batch_src)
    batch_trg = torch.LongTensor(batch_trg)
    batch_length = torch.LongTensor(src_len)
    batch_mask = torch.ByteTensor(batch_mask)
    return (batch_src, batch_length), (batch_trg, batch_mask)


def convert_tensor(input_, device=None, non_blocking=False):
    def _func(tensor):
        return tensor.to(device=device, non_blocking=non_blocking) if device else tensor

    return apply_to_tensor(input_, _func)


def apply_to_tensor(input_, func):
    return apply_to_type(input_, torch.Tensor, func)


def apply_to_type(input_, input_type, func):
    if isinstance(input_, input_type):
        return func(input_)
    elif isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: apply_to_type(sample, input_type, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [apply_to_tensor(sample, input_type, func) for sample in input_]
    else:
        raise TypeError(("input must contain {}, dicts or lists; found {}").format(input_type, type(input_)))