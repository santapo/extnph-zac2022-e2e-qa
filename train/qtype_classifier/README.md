# Description
Create classfication model with 3 labels: `wiki`, `ngày tháng`, `số liệu` base on given dataset zac2022 of this challenge.

# Training pipeline
## Preprocessing

Data default was stored at `./data/e2eqa-train+public_test-v1/zac2022_train_merged_final.json`

```bash
    python3 train/qtype_classifier/pre_processing.py
```

After pre-processing, we stored training data at `./data/e2eqa-train+public_test-v1/classify_data.csv`

## Training

All arguments are located at [`train/qtype_classifier/arguments.py`](https://github.com/santapo/extnph-zac2022-e2e-qa/blob/main/train/qtype_classifier/arguments.py)

Run command to train the model:
```bash
    python3 train/qtype_classifier/run.py
```