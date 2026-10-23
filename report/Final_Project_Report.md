# Deepfake Image Detection with a Custom CNN and EfficientNet-B0

## Abstract

This project classifies face images as FAKE or REAL and compares a custom convolutional neural network (CNN) trained from scratch with an ImageNet-pretrained EfficientNet-B0. Both models used the dataset-provided 8020 split and the same held-out test set of 1,000 images. EfficientNet-B0 achieved 96.9% test accuracy, 0.969 FAKE-class F1, and 0.996 ROC-AUC; the CNN achieved 72.3%, 0.739, and 0.802. A local image-upload demo uses the saved EfficientNet checkpoint to return a label and model probability. These results describe performance on this dataset, not general reliability across other image sources.

## 1. Problem and objective

Manipulated face images can be difficult to distinguish visually. The objective was to build a binary image classifier, establish a CNN baseline, evaluate a transfer-learning alternative, and provide a working inference demonstration. The positive evaluation class is FAKE. The model emits one logit; its sigmoid is interpreted as P(FAKE), and 0.50 is the decision threshold.

## 2. Dataset and integrity

The notebooks use the dataset-provided `8020` directory. Its `train`, `valid`, and `test` folders contain `fake` and `real` subfolders. The recorded counts are 4,000 FAKE and 4,000 REAL for training, 500 of each for validation, and 500 of each for testing. The saved integrity report lists no corrupt images, exact duplicate groups within a split, cross-class duplicate groups, or exact duplicate groups crossing splits. The source image archive is not included in this package because of its size; the original split must be supplied to rerun training.

## 3. Preprocessing and augmentation

Images are converted to RGB. The evaluation transform resizes to 256 × 256 pixels, center crops to 224 × 224, converts to a tensor, and normalizes using ImageNet channel means and standard deviations. Training uses random resized cropping, horizontal flipping, and small rotation, as recorded in the notebooks. The demo repeats the evaluation transform; it does not apply random augmentation.

## 4. Custom CNN baseline

The baseline notebook defines and trains a custom CNN from scratch. It uses a 224 × 224 input, batch size 32, AdamW optimization, validation monitoring, and early stopping. The saved history contains 11 epochs. Its best checkpoint is stored in `checkpoints_cnn_baseline/cnn_baseline_best.pth`. Training from scratch supplies a direct comparison for the pretrained model.

## 5. EfficientNet-B0 method

EfficientNet-B0 starts from ImageNet pretrained weights and replaces its classifier with a single-logit output. Stage 1 trains the classifier head for 10 epochs. Stage 2 unfreezes the last three feature stages for fine-tuning at a lower learning rate, with validation-loss monitoring and early stopping. The saved history contains 25 total epochs. The checkpoint with the lowest validation loss is `checkpoints/efficientnet_b0_best.pth` and powers the demo.

## 6. Training and evaluation procedure

The same training, validation, and test partitions support both models. Validation results guide checkpoint selection; the held-out test set supplies the reported final metrics. Precision, recall, and F1 treat FAKE as positive. ROC-AUC measures ranking across thresholds. The class folder ordering (`fake=0`, `real=1`) is distinct from the binary target convention (`FAKE=1`, `REAL=0`); the notebooks convert labels before loss and metric calculation. The inference code checks this convention in the checkpoint.

## 7. Held-out test results

| Metric | Custom CNN | EfficientNet-B0 |
| --- | ---: | ---: |
| Accuracy | 72.3% | 96.9% |
| Precision (FAKE) | 69.9% | 96.6% |
| Recall (FAKE) | 78.4% | 97.2% |
| F1 (FAKE) | 73.9% | 96.9% |
| ROC-AUC | 0.8018 | 0.9963 |

EfficientNet improves test accuracy by 24.6 percentage points and FAKE-class F1 by 23.0 percentage points. Both models exceed the proposed 70% accuracy and 0.70 F1 targets, while EfficientNet performs better on every listed metric. The plotted training curves, confusion matrices, and ROC curves are in `results/`, `results_cnn_baseline/`, and `results/comparison/`.

## 8. Error analysis

| Model | Correct FAKE | Missed FAKE | REAL false alarms | Correct REAL | Total errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Custom CNN | 392 | 108 | 169 | 331 | 277 |
| EfficientNet-B0 | 486 | 14 | 17 | 483 | 31 |

The CNN falsely labels 33.8% of REAL test images as FAKE and misses 21.6% of FAKE images. EfficientNet reduces these rates to 3.4% and 2.8%, respectively. REAL has slightly lower class recall for both models, making it the harder class by this measure. The saved aggregate results do not include per-image predictions, so they cannot identify the visual causes of individual errors.

The CNN ends at 74.55% training accuracy and 68.3% validation accuracy, a 6.25 percentage point gap. Its validation loss reaches a minimum around epoch 8 and then rises while training loss continues to fall. That pattern supports late-stage overfitting. EfficientNet reaches its minimum validation loss at epoch 23, where training and validation accuracy are 98.95% and 97.6%, a 1.35 point gap. At epoch 25 the gap is 2.06 points. This suggests mild late-stage overfitting, addressed in evaluation by choosing the best validation-loss checkpoint.

## 9. Image-upload demonstration

The Streamlit application in `app/demo.py` accepts JPG, PNG, or WebP images. It loads the saved EfficientNet checkpoint, applies the evaluation preprocessing, computes P(FAKE), and displays FAKE or REAL, confidence in the predicted class, P(FAKE), and P(REAL). Run it from the project root with `python -m streamlit run app/demo.py` in an environment containing the packages in `requirements.txt`. The uploaded image is an inference input; no retraining occurs.

## 10. Conclusion

On this 1,000-image held-out test set, EfficientNet-B0 substantially outperforms the custom CNN and meets the proposal's numerical success criteria. Transfer learning plus fine-tuning is the better of the two tested methods for this dataset. The saved checkpoint and demo complete the requested inference workflow.

## 11. Limitations and future work

The image dataset is not packaged with the code, so training reproduction requires the original archive. The saved artifacts establish results on one split and do not measure performance on other datasets, generators, compression levels, or cameras. The displayed probabilities are raw model outputs and have not been calibrated as confidence estimates. Future work should test separate external datasets, retain image-level error records for visual review, and assess probability calibration.

## 12. Reproducibility and evidence

The three notebooks in `ipynbs/` contain data checks, model definitions, training, evaluation, and comparison. The two checkpoint directories contain the saved models. Metric JSON files and figures in the results directories support the values above. `README.md` provides project layout and run instructions. `results/comparison/error_analysis.md` gives the concise error analysis used here.
