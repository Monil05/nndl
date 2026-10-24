# NNDL Deepfake Detection Project

This project compares a custom CNN trained from scratch with an ImageNet-pretrained EfficientNet-B0 for classifying face images as **FAKE** or **REAL**. Both saved checkpoints power the local image-upload demo.

## Project layout

| Path | Contents |
| --- | --- |
| `ipynbs/` | Dataset preparation, training, evaluation and final model comparison notebooks |
| `checkpoints/efficientnet_b0_best.pth` | Saved EfficientNet model used by the demo |
| `checkpoints_cnn_baseline/` | CNN baseline checkpoint |
| `results/` and `results_cnn_baseline/` | Metrics, training curves and confusion matrices |
| `results/comparison/` | Side-by-side metrics, figures and error analysis |
| `app/` | Local Streamlit image-upload demo with predictions from both models |
| `dataset/` | Dataset instructions; the image dataset is excluded from Git because of its size |
| `report/` | Final report in Markdown and editable Word format |
| `presentation/` | Final editable PowerPoint presentation |
| `tools/build_deliverables.py` | Rebuilds the Word report and slides from the saved project content |

## Run the demo locally

Open this directory in VS Code and use the Python environment where PyTorch and torchvision work. Install the remaining packages if needed:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app/demo.py
```

Open the local URL printed by Streamlit, upload a JPG, PNG or WebP face image, and select **Analyze image**. The interface shows each model's prediction, confidence, `P(FAKE)` and `P(REAL)`. It also displays the held-out EfficientNet metrics from `results/test_metrics.json`. The demo loads `checkpoints/efficientnet_b0_best.pth` and `checkpoints_cnn_baseline/cnn_baseline_best.pth`; keep both files in place. These checkpoints are trusted local training artifacts and are loaded with `weights_only=False` because they contain optimizer and random-number state as well as model weights. Do not replace them with untrusted files.

Inference matches both notebooks' test transform: convert to RGB, resize to 256 × 256, center crop to 224 × 224, convert to tensor, and normalize with ImageNet mean and standard deviation. The sigmoid of each model's single logit is `P(FAKE)` because training used **FAKE = 1** and **REAL = 0**. At a threshold of 0.50, confidence is the probability assigned to the predicted class.

The model was evaluated on face images from this project's dataset. Its score on an uploaded image is not proof that the image is authentic or manipulated, especially for images from other generators, cameras or domains.

## Held-out test results

| Model | Accuracy | Precision (FAKE) | Recall (FAKE) | F1 (FAKE) | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Custom CNN | 72.3% | 69.9% | 78.4% | 73.9% | 0.8018 |
| EfficientNet-B0 | 96.9% | 96.6% | 97.2% | 96.9% | 0.9963 |

Both models use the same 1,000-image held-out test set (500 FAKE, 500 REAL). See [the error analysis](results/comparison/error_analysis.md) for the confusion counts, class-specific failure rates and train/validation gaps. The original data split and training procedure are documented in the notebooks.

## Final deliverables

- [Final report](report/Final_Project_Report.docx) ([Markdown source](report/Final_Project_Report.md))
- [Final presentation](presentation/Final_Project_Presentation.pptx)

The saved checkpoint and evaluation artifacts are included. The original image dataset is not included; follow [dataset instructions](dataset/README.md) and keep the dataset-provided `8020/train`, `8020/valid`, and `8020/test` split to rerun the notebooks. A Python environment with PyTorch, torchvision, Pillow, and Streamlit is required to run the demo. Use the interpreter selected in VS Code rather than the Windows Store `python` alias if that alias is inactive.
