from __future__ import annotations
from einops.einops import rearrange
import torch
from typing import Callable, List, Union, Any, Tuple, Dict
from torch.utils import data
from torch.utils.data import Dataset
from torch import Tensor
from pathlib import Path
import pandas as pd
from pypeln import process
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm


class ScientistKeypointsDataset(Dataset):
    labels: Dict[str, int] = {'pick_up': 0, 'walking': 1, 'put_back': 2, 'raise_hand': 3, 'standing': 4}

    def __init__(self, df: pd.DataFrame, label: str, seq_len: int = 9, transform: Callable[[Tensor], Tensor] = None):
        self.df = df
        self.transform = transform
        self.target = torch.Tensor([-1]).long() if label == '' else torch.Tensor([self.labels[label]]).long()
        self.frames = list(df.index.unique())
        self.seq_len = seq_len

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        start = self.frames[idx]
        seq_len = min(idx + self.seq_len, len(self.frames) - 1)
        end = self.frames[seq_len - 1]
        df_slice = self.df.loc[start:end]
        
        keypoints = torch.zeros(self.seq_len, 2, 18)
        # showing how much I am a noob in pandas
        for i, slice_idx in enumerate(df_slice.index.unique()):
            rows_slice = self.df.loc[slice_idx]

            xs = torch.Tensor(rows_slice.x.values)
            ys = torch.Tensor(rows_slice.y.values)
            #   center them on left top corner
            xs -= xs.min()
            ys -= ys.min()

            keypoints[i, 0, rows_slice.p.values - 1] = xs
            keypoints[i, 1, rows_slice.p.values - 1] = ys
            
        if self.transform:  
            # we want to normalize the xs and the ys
            keypoints = self.transform(keypoints.permute(1,2,0)).permute(2,0,1)
        # seq len to the end
        keypoints = rearrange(keypoints, 'seq dims features -> (dims features) seq')
        return keypoints, self.target

    def __len__(self) -> int:
        return len(self.frames)

    @classmethod
    def from_path(cls, path: Path,  *args, **kwrags) -> ScientistKeypointsDataset:
        df = pd.read_csv(path, index_col=0,
                         header=0,
                         names=['p', 'x', 'y', 'score'])

        return cls(df, *args, **kwrags)

    @classmethod
    def from_root(cls, root: Path, *args, **kwargs) -> Tuple[ConcatDataset, Dict[str, int]]:
        labels = {}
        label_idx = 0
        paths = root.glob('*.csv')
        datasets = []
        bar = tqdm(paths)

        for path in bar:
            bar.set_description(f'{path.stem}')
            label = '_'.join(str(path.stem).split('_')[1:])
            if label not in labels:
                labels[label] = label_idx
                label_idx += 1
            ds = cls.from_path(path, label=label, *args, **kwargs)
            datasets.append(ds)
        ds = ConcatDataset(datasets)
        return ds, labels

# if __name__ == '__main__':
#     # not elegenta but code to print mean and stds
#     # tensor([173.4721, 247.3274]) tensor([143.8747, 229.3648])

#     ds, labels = ScientistKeypointsDataset.from_root(Path(
#         '/home/zuppif/Documents/scientists-keypoints-classification/dataset/train/'))
    
#     print(labels)
#     print(ds[1])

#     means, stds = torch.zeros(2), torch.zeros(2)


#     for (k, l) in tqdm(ds):
#         means += k.mean(dim=1)
#         stds += k.std(dim=1)

#     print(means, stds, means/len(ds), stds/len(ds))
