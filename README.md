# Spall-to-Full

Official research code for the depth-completion component of the manuscript
**“Semantics Guided RGB-D Quantification of Fire-induced Concrete Spalling.”**

Spall-to-Full completes noisy or sparse metric depth maps of fire-damaged
concrete by combining RGB appearance, sparse depth observations, and a
three-class semantic mask (background, intact concrete, and spalled concrete).
The separate spalling measurement code is available in the
https://github.com/MikeyDong/Spall-Depth-and-area-measurement-from-Single-RGBD-image-arbitrary-view.

!\[Spall-to-Full overview](Image.svg)

## Repository contents

|Path|Purpose|
|-|-|
|`train\\\_spall2full.py`|Fine-tune Spall-to-Full from the released Any2Full ViT-L checkpoint.|
|`test\\\_spall\\\_to\\\_full.py`|Run inference and quantitative evaluation on one or more sparse-depth test conditions.|
|`model/ours/`|Model implementation and semantic-mask conversion utility.|
|`requirements.txt`|Python dependencies for the reported CUDA 12.1 environment.|

## Code, data, and weights

|Item|Location|Access code / expected local path|
|-|-|-|
|Source code|[GitHub repository](https://github.com/MikeyDong/Spall-to-Full)|Clone as described below.|
|Training and validation data|[Baidu Netdisk](https://pan.baidu.com/s/1CKtUdeUk33h2RJbShvrsAQ)|Code: `54hj`|
|Test data|[Baidu Netdisk](https://pan.baidu.com/s/1THFZ6zij-hYK9cplbhpjfA)|Code: `123h`|
|Final Spall-to-Full checkpoint|[Baidu Netdisk](https://pan.baidu.com/s/1lBauxafJxa16T1QPokWPlw)|Code: `yh8s`; place under `checkpoints/` and pass its filename to `--checkpoint`.|
|Any2Full ViT-L initialization checkpoint|[Official Any2Full repository](https://github.com/zhiyuandaily/Any2Full)|Save as `checkpoints/Any2Full\\\_vitl.pth.tar` for training.|



## Environment and installation

The experiments reported in the manuscript used Python 3.11.8, PyTorch 2.2.2,
CUDA 12.1, and NumPy 1.26.4. The reported training configuration used a ViT-L
model, 1456 × 1456 inputs, batch size 1. Training at
the resolution of 1456 × 1456 may not fit on lower-memory GPUs.

The commands below are for a Linux shell:

```bash
git clone https://github.com/MikeyDong/Spall-to-Full.git
cd Spall-to-Full

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt


## Input conventions

All modalities belonging to an image must have the same filename stem, for
example `sample\\\_001.jpg`, `sample\\\_001.png`, and `sample\\\_001.png` in their
respective directories.

|Input|Required representation|
|-|-|
|RGB|Three-channel images.|
|Metric depth|Single-channel depth image. The released data and default commands use millimetres, with zero denoting invalid or missing depth. Sixteen-bit PNG is recommended.|
|Training semantic mask|Three-channel PNG with channel order `\\\[background, intact concrete, spalled concrete]`.|
|Evaluation semantic mask|Single-channel label image with `0 = background`, `1 = intact concrete`, and `2 = spalled concrete`. The evaluation script converts it to one-hot form internally.|
|Evaluation ROI mask|Single-channel image; every nonzero pixel is included in the foreground ROI.|

If the training annotations are single-channel class-label PNGs, convert them
to three-channel one-hot masks with:

```bash
python model/ours/build\\\_onehot\\\_masks.py \\\\
  --input-dir data/mask\\\_labels \\\\
  --output-dir data/mask\\\_onehot
```

## Training

### 1\. Arrange the data and initialization checkpoint

The default paths in `train\\\_spall2full.py` expect:

```text
Spall-to-Full/
├── checkpoints/
│   └── Any2Full\\\_vitl.pth.tar
└── data/
    ├── rgb/
    ├── Depth/
    ├── mask\\\_onehot/
    └── Val/
        ├── rgb/
        ├── Depth/
        └── mask\\\_onehot/
```

The published split contains 553 training images and 138 validation images
(691 images in total). If all 691 triplets are initially placed in the three
top-level training directories and the `data/Val/` directories are empty, the
script uses seed 42 to select 20% for validation. **This operation physically
moves the selected RGB, depth, and mask files into `data/Val/`.** Keep a backup
of the downloaded archive. If a validation split is already present, retain the
published 553/138 split and do not repartition it.

Training images must be at least `EXPECTED\\\_SIZE × EXPECTED\\\_SIZE` because the
default pipeline samples four random crops per image per epoch. Validation
images are resized to that size.

### 2\. Review the configuration

Paths and hyperparameters are constants near the beginning of
`train\\\_spall2full.py`; the script does not currently expose them as command-line
arguments. The manuscript configuration is already the default:

|Setting|Default|
|-|-:|
|Input size|1456 × 1456|
|Batch size|1|
|Epochs|30|
|Random crops per training image|4|
|Training/validation ratio|80/20|
|Random seed|42|
|Optimizer / learning rate|Adam / 5 × 10⁻⁵|
|Maximum supervised depth|3000 mm|
|Synthetic training deletion|60% of all image positions|

### 3\. Start training

Run the command from the repository root:

```bash
python train\\\_spall2full.py
```

The script writes the following files to `outputs/`:

* `train\\\_val\\\_log.csv`: per-epoch training and validation losses;
* `last\\\_model.pth`: checkpoint from the most recent epoch;
* `epoch\\\_15\\\_model.pth`: additional epoch-15 checkpoint;
* `best\\\_model.pth`: checkpoint with the lowest validation total loss.

The saved checkpoint dictionaries contain `model\\\_state`, `optimizer\\\_state`,
`epoch`, `best\\\_val`, and the training/validation loss summaries. The evaluation
script accepts this format directly.

Please use the Last model as the final model and conduct testing.

## Evaluation and testing

### 1\. Arrange the test inputs

Extract the released test archive. The folder names may be different from the
example below; map the actual folders to the corresponding command-line
arguments:

```text
data/test/
├── rgb/
├── depth\\\_inputs/
│   ├── delete\\\_40/
│   ├── delete\\\_60/
│   └── delete\\\_80/
├── semantic\\\_masks/
├── gt\\\_depth/
└── roi\\\_masks/
```

Each sparse-depth directory is treated as a separate test case, and its folder
name is used in the output tables. The released quantitative test set contains
26 images. All input types are paired by filename stem; an image with a missing
pair is skipped and reported in the terminal.

### 2\. Run the released evaluation

Replace the checkpoint filename and, if necessary, the extracted folder names:

```bash
python test\\\_spall\\\_to\\\_full.py \\\\
  --rgb\\\_dir data/test/rgb \\\\
  --depth\\\_input\\\_dirs \\\\
    data/test/depth\\\_inputs/delete\\\_40 \\\\
    data/test/depth\\\_inputs/delete\\\_60 \\\\
    data/test/depth\\\_inputs/delete\\\_80 \\\\
  --mask\\\_dir data/test/semantic\\\_masks \\\\
  --gt\\\_depth\\\_dir data/test/gt\\\_depth \\\\
  --roi\\\_mask\\\_dir data/test/roi\\\_masks \\\\
  --checkpoint checkpoints/spall\\\_to\\\_full.pth \\\\
  --out\\\_root outputs/evaluation \\\\
  --encoder vitl \\\\
  --model\\\_input\\\_size 1456 \\\\
  --depth\\\_scale 1.0 \\\\
  --save\\\_npy
```

`--depth\\\_scale` is the number by which values read from depth images are divided.
Use the default `1.0` when the files already store millimetres. For example, use
`1000` only if an integer image stores 1000 units per millimetre. Run
`python test\\\_spall\\\_to\\\_full.py --help` for all optional arguments.

### 3\. Expected outputs

For each sparse-depth case, the script creates:

```text
outputs/evaluation/<case>/
├── Spall\\\_to\\\_Full/
│   ├── 01\\\_predictions/
│   ├── 02\\\_colorized\\\_predictions/
│   ├── 03\\\_gt\\\_prediction\\\_comparisons/
│   └── 04\\\_metrics/
├── <case>\\\_metrics.xlsx
├── <case>\\\_per\\\_image\\\_metrics.csv
└── <case>\\\_summary.csv
```

It also creates cross-case files in `outputs/evaluation/`:

* `all\\\_cases\\\_metrics.xlsx`;
* `all\\\_cases\\\_per\\\_image\\\_metrics.csv`;
* `all\\\_cases\\\_summary.csv`.

The primary manuscript metrics are MAE and RMSE in millimetres, evaluated over
the supplied foreground ROI containing intact and spalled concrete. With the
released 26-image quantitative test set, the exact released sparse-depth maps,
ROI masks, and final 1456 checkpoint, the manuscript reports:

|Valid-depth deletion ratio|MAE (mm)|RMSE (mm)|
|-:|-:|-:|
|40%|4.4|9.3|
|60%|6.0|13.0|
|80%|7.8|15.9|

These values are reference aggregate results. Small numerical differences may
occur across GPU models and software builds; large differences usually indicate
an input-unit, filename-pairing, semantic-label, ROI, resolution, or checkpoint
mismatch.

## Using another model input size

The published model uses 1456. Lower-resolution experiments can be run by
changing `EXPECTED\\\_SIZE` in `train\\\_spall2full.py`, training a new checkpoint,
and evaluating that checkpoint with the same value passed to
`--model\\\_input\\\_size`. For example, a 1036 model requires:

```python
# train\\\_spall2full.py
EXPECTED\\\_SIZE = 1036
```

```bash
python test\\\_spall\\\_to\\\_full.py ... --model\\\_input\\\_size 1036
```

The input size and checkpoint must match. Do not change the model patch size.

## Citation

The manuscript is currently under assessment. Bibliographic citation details
will be added after publication.

