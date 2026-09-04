# Code and Data

Code and data for reproducing the H₂S gas-sensor gas-discrimination results.
A 1D-CNN classifies four gases (H₂S, CH₄, CO₂, air) from 500-point resistance
waveforms, with RF / SVM / XGBoost / KNN baselines, cross-concentration and
unseen-concentration tests, and mixed-gas inference.

## Structure

```
code/              # preprocessing, training, inference, plotting scripts
data/raw_data/     # raw CSV recordings (H2S, CH4, CO2, AIR, unseen_H2S)
data/preprocessed/ # X.npy (320×500), y.npy (320) — ready for training
results/           # trained models + result CSVs for all tables
```

Trained models and result CSVs for each table (S3–S7, Fig 3, Fig S27) are
already under `results/`. The commands below either reproduce them from
`data/preprocessed/` or run inference with the shipped models.

## Reproduce results

```
pip install -r code/requirements.txt
export PYTHONPATH=code

python code/train_cnn_cv.py --seed 42      # CNN 5-fold (Table S3)
python code/train_knn_cv.py --seed 1        # KNN 5-fold
python code/train_ml_cv.py --seed 123        # RF/SVM/XGB 5-fold
python code/eval_unseen_h2s.py             # unseen H₂S (Table S5)
python code/visualize_tsne.py                # t-SNE (Fig S27)
```

Each script writes its models and CSVs back to `results/`.

## Data processing

`data/preprocessed/` is ready to use. To rebuild it from raw CSVs instead:

```
python code/data_process.py --gas H2S   # repeat for CH4, CO2, AIR
python code/data_convert.py             # → data/converted/X.npy, y.npy
```

`data_process.py` normalizes each gas's raw recordings into 500-point
waveforms; `data_convert.py` stacks them into the `X.npy` / `y.npy` arrays.
Point training scripts at `--data_dir data/converted` to use the rebuilt data.

See `code/requirements.txt` for dependencies (Python 3.9+, PyTorch 1.13).
