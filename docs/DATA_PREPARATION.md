# Data preparation

Obtain the original datasets from their authors. Dataset access is separate from this repository: [MuS-Polar3D dataset page](https://github.com/WangPuyun/MuS-Polar3D) and [DeepSfP project page](https://visual.ee.ucla.edu/deepsfp.htm/). Download the datasets separately, then prepare a cache with the commands in the README.

## MuS-Polar3D

The required source layout is:

```text
MuS-Polar3D_Mat/
  Bach_back_1.mat
  ...
MuS-Polar3D_Pol/
  Bach_back_1/
    0.png
    45.png
    90.png
    135.png
    mask.png
    Normal_gt.png
  ...
```

Each MAT file must provide `images`, `mask`, `I_Normal_gt`, `P`, `IrawD`, and `IrawS`. The four raw grayscale PNGs are read as float32 and divided by 255. `mask.png > 127` defines valid foreground pixels. `Normal_gt.png` is decoded as RGB/255 × 2 − 1 and normalized to unit length.

The loader checks that the PNG observations, mask, and encoded normal target match the MAT fields. The five auxiliary input channels are the supplied `P[..., 1:4]`, `IrawD`, and `IrawS`. These supplied fields are preserved, including their amplitude convention; `P3` must not be silently replaced by textbook DoLP.

Only the 570/84 samples listed in `splits/mus/` are used. Samples at turbidity level 0 and additional unlisted images are not automatically added. Training and validation are separated by object identifiers. The split lists must not be reshuffled when reproducing this protocol.

## DeepSfP

The loader accepts the `DeepSfPData`, `SurfaceNormals`, or `objects` directory:

```text
DeepSfPData/
  SurfaceNormals/
    objects/
      indoor/
        bird1_b.mat
        ...
      outdoor_cloudy/
        ...
      outdoor_sunny/
        ...
```

Each MAT file must contain `images` with shape 1024 × 1224 × 4, a binary `mask` of shape 1024 × 1224, and Cartesian `normals_gt` of shape 1024 × 1224 × 3. Raw intensities are retained at their supplied scale. The loader normalizes foreground target normals, then retains columns `[100:1124]`, producing a 1024 × 1024 field.

DeepSfP does not provide the MuS auxiliary fields. Its five input fields are derived from raw observations using the frozen adapter:

```text
q = I0 - I90
u = I45 - I135
a = hypot(q, u)
P1 = (q / a) × mask
P2 = (u / a) × mask
P3 = hypot(q, q + u)
IrawD = (mean(I0, I45, I90, I135) - a) × mask
IrawS = a × mask
```

The divisions return zero when `a <= 1e-8`. These expressions define the experiment's input convention, not a physical DoLP estimator. Ground-truth normals are never used to derive input features.

The included lists define 236 official training and 21 official test images. Within the official training set, the development protocol uses 212 training and 24 validation images. The supplemental 65-image collection is not treated as an independent test set. The test split is prepared only with `--include-test` and is evaluated separately after final training.

## Model input and sampling

The model input is a float32 tensor with channel order:

```text
I0, I45, I90, I135, P1, P2, P3, IrawD, IrawS, u_coordinate, v_coordinate, 1
```

Coordinate channels retain the original sensor location. For a pixel at sensor coordinate `(x, y)`, they are `(x − W/2)/(W/2)` and `(y − H/2)/(H/2)`. The DeepSfP horizontal offset of 100 pixels is retained after the central crop.

Each epoch samples one 256 × 256 crop per training image, uniformly from all eligible top-left positions. Foreground occupancy must be strictly greater than 50% for MuS or 40% for DeepSfP. Sampling is deterministic given the seed, epoch, and sample index. No rotations, flips, resizing, or normal-coordinate transformations are applied during training.

Evaluation uses the complete supplied MuS field or the 1024 × 1024 DeepSfP central field. It does not reuse the random training crop.

## Prepared cache

Run `prepare_data.py` once before training. It creates a separate `manifest.json` and one directory per sample under `records/`. MuS stores auxiliary fields and eligible crop indices; the original PNGs remain necessary during training and evaluation. DeepSfP stores raw intensities, masks, normalized normals, and eligible crop indices as memory-mapped NPY arrays.

Preparation resumes by checking existing committed records. Incomplete temporary directories are not used. Source roots and file metadata are recorded so accidental mixing of datasets is detected. If source files move or change, use a new cache directory. Do not place the cache inside the original dataset tree. Full preparation can require tens of gigabytes of disk space.
