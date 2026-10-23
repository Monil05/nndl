# Final error analysis

**Positive class: FAKE.** Both models were evaluated on 500 FAKE and 500 REAL images in the held-out test set. The confusion matrices have true classes as rows and predicted classes as columns.

| Model | FAKE → FAKE | FAKE → REAL (missed fake) | REAL → FAKE (false alarm) | REAL → REAL |
| --- | ---: | ---: | ---: | ---: |
| Custom CNN | 392 | 108 | 169 | 331 |
| EfficientNet-B0 | 486 | 14 | 17 | 483 |

EfficientNet-B0 performs better on every reported metric. Its accuracy is **96.9%** versus **72.3%** for the CNN, an increase of **24.6 percentage points**. FAKE-class F1 rises from **0.739** to **0.969**. The total number of test mistakes falls from **277 to 31**.

The CNN's larger failure is on REAL images: it falsely labels **169/500 (33.8%)** as FAKE. It also misses **108/500 (21.6%)** FAKE images. EfficientNet makes both kinds of mistakes much less often: **17/500 (3.4%)** false alarms on REAL and **14/500 (2.8%)** missed FAKE images. By class recall, REAL is the slightly harder class for both models. The confusion counts show *how often* each error occurs, but the saved artifacts do not identify which visual features caused individual mistakes.

The CNN training history ends with **74.55% train accuracy** and **68.3% validation accuracy**, a **6.25 percentage point** gap. Validation loss reaches its minimum around epoch 8 and later rises while training loss continues to fall. This is evidence of overfitting in the later CNN epochs. EfficientNet's lowest validation loss is at epoch 23; at that epoch, train accuracy is **98.95%** and validation accuracy is **97.6%**, a **1.35 percentage point** gap. At the final logged epoch 25 the gap is **2.06 percentage points** (99.06% train, 97.0% validation), with validation loss above its epoch-23 minimum. This indicates mild late-stage overfitting, limited by selecting the best validation-loss checkpoint for testing and the demo.

These conclusions are limited to this held-out dataset. The current saved results provide aggregate errors, not image-level predictions or a separate cross-dataset evaluation; they cannot establish which manipulations, lighting conditions or image sources are most difficult.

Sources: `results/test_metrics.json`, `results_cnn_baseline/cnn_baseline_test_metrics.json`, the two `training_history.json` files, and the two held-out-test confusion matrix images in those directories.
