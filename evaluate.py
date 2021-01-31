import torch
import pandas as pd
from tqdm import tqdm
from Project import Project
from data.ScientistKeypointsDataset import ScientistKeypointsDataset
from data.transformation import val_transform
from models import MyCNN

from system import MySystem
from torch.nn.functional import softmax
from pprint import pprint

if __name__ == '__main__':
    project = Project()
    checkpoint = './checkpoint/1612021398.2369566/best.ckpt'

    model = MyCNN()
    system = MySystem.load_from_checkpoint(checkpoint, model=model)
    system.freeze()
    system.eval()

    datasets, _ = ScientistKeypointsDataset.from_root(
        project.data_dir / 'test', transform=val_transform)


    idx2label = { v: k for k, v in datasets.datasets[0].labels.items()}

    pprint(idx2label)
    
    records = []
    with torch.no_grad():
        for ds in tqdm(datasets.datasets):
            ds_records = []
            for x, y in ds:
                # add batch dim
                x = x.unsqueeze(0)
                out = system.model(x)
                probs = softmax(out, dim=1)
                label = torch.argmax(probs, dim=1)
                # compute also the score, why not
                score = probs.squeeze()[label]
                ds_records.append({'label': label, 'score': score})

            df = pd.DataFrame.from_records(ds_records)
            # vote the most frequent
            counts = df.label.value_counts()
            final_pred = counts.index.tolist()[0].item()
            records.append({'name': ds.name + '.csv', 'label': idx2label[final_pred]})

        df = pd.DataFrame.from_records(records)
        df = df.sort_values(by='name').set_index('name', drop=True)
        df.to_csv('./predictions.csv')
