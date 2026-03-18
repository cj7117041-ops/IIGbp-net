# IGBP-Net

IGBP-Net is a PyTorch-based deep learning project for image processing tasks (modify according to your project functionality), providing training and testing pipelines.

## Features

- Model training
- Model testing
- Custom dataset support
- Pretrained weights support

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.0 (optional, for GPU support)


## Dataset Preparation

Please use your own dataset with the following structure:

```
dataset/
│
├─ train/
│   ├─ images/
│   └─ masks/
│   └─ boundary/
│
├─ val/
│   ├─ images/
│   └─ masks/
│   └─ boundary/
│
└─ test/
    ├─ images/
    └─ masks/
    └─ boundary/
```

- `images/` contains the input images
- `masks/` contains the corresponding masks
- `boundary/` contains the corresponding boundary


> Make sure to update the `--data_path` argument to point to your dataset.

## Pretrained Weights

This project uses `pvt_v2_b3.pth` as pretrained weights.Provide pre-training for this model. You can download it from the official repository:

[PVT v2 pretrained weights](https://github.com/whai362/PVT)

Place the weights in the `pretrained/` folder or update the `--pretrained_path` argument accordingly.This weight is used in the code of this model structure.

## Training

```bash
python train.py --data_path /path/to/your/dataset 
```

- `--data_path`: path to your dataset

Use `--help` to see additional training options:

```bash
python train.py --help
```

## Testing

```bash
python test.py --data_path /path/to/your/dataset --model_path /path/to/saved_model.pth
```

- `--model_path`: path to the trained model weights


## Notes

- Remove hard-coded local paths in the code; use command-line arguments instead.
- Ensure dataset structure matches the required format.
- Adjust batch size if GPU memory is limited.

