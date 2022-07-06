import shutil
from pathlib import Path

import torch
from tqdm.auto import tqdm

ixi_dir = Path('/tmp/ixi_tiny')
image_dir = ixi_dir / 'image'
label_dir = ixi_dir / 'label'
image_paths = sorted(image_dir.glob('*gz'))
label_paths = sorted(label_dir.glob('*gz'))
pairs = list(zip(image_paths, label_paths))
lengths = 190, 190, 186
splits = torch.utils.data.random_split(pairs, lengths)

for i, split in enumerate(splits, start=1):
    out_dir = Path(f'ixi_tiny_{i}')
    image_dir = out_dir / 'image'
    label_dir = out_dir / 'label'
    shutil.rmtree(image_dir)
    shutil.rmtree(label_dir)
    image_dir.mkdir(exist_ok=True, parents=True)
    label_dir.mkdir(exist_ok=True, parents=True)
    for image_path, label_path in tqdm(split):
        dst = image_dir / image_path.name
        if dst.is_file(): continue
        shutil.copy(image_path, dst)
        dst = label_dir / label_path.name
        shutil.copy(label_path, dst)
