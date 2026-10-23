# Dataset placement

The original image dataset is omitted because its archive is too large for this repository. To rerun the training notebooks, place the original archive where their dataset setup cell expects it. The notebooks select the dataset-provided `8020` split and require this structure after extraction:

```text
8020/
  train/fake/  train/real/
  valid/fake/  valid/real/
  test/fake/   test/real/
```

The recorded split has 4,000 images per class for training and 500 per class in each of validation and test. Do not create a new random split if reproducing the published results. The notebooks run integrity checks before training. The saved aggregate integrity report is in `results/dataset_integrity_report.json`.
