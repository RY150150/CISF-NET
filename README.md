# CISF-NET

Source code for training and testing CISF-NET. Datasets and model weights are
not included.

## Files

- `train.py`: training script.
- `test.py`: testing script.
- `model/`: CISF-NET implementation, including PMiT, MSAM, and SIEM.
- `utils/`: data loading, loss functions, and utilities.
- `requirements.txt`: required Python packages.

## Data layout

```text
data/training/
  images/
  masks/

data/test/
  DatasetName/
    images/
    masks/
```

## Install

```bash
pip install -r requirements.txt
```

## Train

Place the backbone initialization file at `pretrained_pth/mit_b3.pth`, then
run:

```bash
python train.py \
  --train_path data/training/ \
  --pretrained_path pretrained_pth/mit_b3.pth
```

Training runs for 60 epochs by default without a validation set. The final
model is saved as `model_pth/CISF-NET/final.pth`.

## Test

```bash
python test.py \
  --pth_path model_pth/CISF-NET/final.pth \
  --test_path data/test/ \
  --datasets DatasetName
```

Prediction maps are saved in `result_map/CISF-NET/`.
