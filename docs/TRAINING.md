# Training and evaluation protocol

## Configurations

| Config | Training images | Validation images | Final evaluation | Epochs | Cosine T_max | Seed |
|---|---:|---:|---|---:|---:|---:|
| `mus.json` | 570 | 84 | 84 validation images | 1000 | 1000 | 0 |
| `deepsfp_development.json` | 212 | 24 | 24 development images | 400 | 400 | 42 |
| `deepsfp_final.json` | 236 | None | 21 official test images | 360 | 400 | 42 |

All use FP32, AdamW, batch size 16, initial learning rate 2e-4, minimum learning rate 2e-6, weight decay 1e-4, and gradient-norm clipping at 1.0. Bias parameters, one-dimensional parameters, and parameters explicitly marked to exclude decay are placed in a zero-decay group. The scheduler advances after each epoch. The final smaller batch is retained. No gradient accumulation or mixed precision is used.

The development checkpoint is the earliest epoch with the smallest validation image-macro MAE among evaluations every 20 epochs and the final epoch. DeepSfP final training uses a new initialization, never selects on the test set, and uses the fixed 360th-epoch checkpoint. Do not substitute `last.pt` for `best.pt` in a validation-based experiment.

## Architecture and losses

The encoder has four stages with 80, 160, 320, and 640 channels. A 192-channel feature pyramid feeds the progressive decoder. CRUs operate at quarter and half resolution with 8 × 8 windows and shifts of 0 and 4. UGF predicts one acceptance value per output pixel from decoder features, observation descriptors, and detached base/candidate cues. Final normals are normalized convex blends of the base and candidate.

The normal objective is masked per-image cosine loss, averaged across valid images. Weights are 1.0 for the final prediction, 0.2 for the half-resolution auxiliary normal, 0.1 for the quarter-resolution auxiliary normal, and 0.5 for the candidate. The utility objective has weight 0.01.

Let Δe be the base angular error minus the candidate angular error, in degrees. The detached utility target is sigmoid(Δe / 1°), and the detached importance is min(|Δe| / 0.25°, 1). Weighted BCE is normalized within each image before the batch mean. Ground truth is used to construct these training targets, never passed to `CRUFNet.forward`.

For epochs 1–10 the output uses the candidate. Learned blending is introduced with strength `(epoch − 10)/20` for epochs 11–30, then uses full strength. Utility supervision remains active throughout. Evaluation always uses the learned acceptance without the training ramp.

## Ablation interpretation

Each variant first constructs the same full model, then changes the selected components. This preserves the initial retained tensors across variants for a fixed seed in this package. `no_cru` replaces both reconstruction units with identity operations. `no_ugf` removes the learned fusion predictor and returns the candidate. `baseline` combines those two interventions. `fixed_blend` uses acceptance 0.5 after the same warmup/ramp. `no_utility_loss` keeps the learned predictor and sets only utility BCE weight to zero.

Half, quarter, and candidate supervision remain active. In candidate-only variants, final and candidate losses act on the same output with combined weight 1.5. 

## Inference and metrics

Inference uses native 256 × 256 patches with stride 128. A separable Hann weight, clamped to at least 1e-3, combines overlapping unit-normal predictions. The accumulated result is normalized again. No sharpening, smoothing, or prediction-dependent test tuning is applied.

Within each valid mask, predicted and reference vectors are normalized and angular errors are computed in degrees. For each image, evaluation reports MAE, median error, RMSE, and the percentage of pixels strictly below 11.25°, 22.5°, and 30°. The main dataset result is the arithmetic mean of each per-image metric, rather than a pooled-pixel score. MuS also exports the same metrics separately by turbidity suffix.

Base, candidate, and acceptance maps can be exported for visualization. These are accumulated independently across patches. Applying the averaged acceptance to averaged endpoints is not generally identical to the final map, because vector normalization and overlap fusion do not commute.

## Initialization and environment

The training script initializes the model on CPU before moving it to the selected device. Keep the constructor order and the two discarded random draws in `ContextEncoder` unchanged: they preserve the initialization sequence used for the experiments without creating unused modules. They add no trainable parameters or inference computation.

Dependencies are pinned in `requirements.txt`. Random-state settings are read directly from each configuration. Reproducibility across different PyTorch builds and devices depends on their numerical implementations.
