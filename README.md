# ProtoPlanetary Disk AutoEncoders
An AutoEncoder model to reconstruct and generate new images of edge-on Proto Planetary disks using physical parameters as input.

The model architecture is the following:

![alt text](https://github.com/jorgemarpa/PPDAE/blob/main/figures/PPDAE_arch_V2.png)

## Image Samples

The edge-on disk images used for training were generated using the MCFOST Radiative Transfer code. The training set looks like this:

![alt text](https://github.com/jorgemarpa/PPDAE/blob/main/figures/image_wall.png)

with a wide variety of shapes and sizes.

## Physical Parameters
Each image is associated with the physical parameters used to simulate the source. We used 8 physical parameters to as inputs to MCFOST:

```
    m_dust = 'mass of the dust'
    Rc     = 'critical radius when exp drops(size)'
    f_exp  = 'flare exponent'
    H0     = 'scale hight'
    Rin    = 'inner raidus'
    sd_exp = 'surface density exponent'
    alpha  = 'dust stettling'
    inc    = 'inclination'
```
All physical parameters where sampled from a evenly spaced grid and a random sampling between the range of possible values in order to fill up the gaps.

![alt text](https://github.com/jorgemarpa/PPDAE/blob/main/figures/phy_params.png)

## Usage

Use `ae_main.py` to train a AE model with the following parameters:
```
  -h, --help            show this help message and exit
  --dry-run             Load data and initialize models [False]
  --machine MACHINE     were to is running (local, colab, [exalearn])
  --data DATA           data used for training ([MNIST], ProtoPD)
  --lr LR               learning rate [1e-4]
  --lr-sch LR_SCH       learning rate shceduler ([None], step,
                        exp,cosine, plateau)
  --batch-size BATCH_SIZE
                        batch size [128]
  --num-epochs NUM_EPOCHS
                        total number of training epochs [100]
  --early-stop          Early stoping
  --cond COND           label conditional VAE (F,[T])
  --latent-dim LATENT_DIM
                        dimension of latent space [6]
  --dropout DROPOUT     dropout for lstm/tcn layers [0.2]
  --kernel-size KERNEL_SIZE
                        kernel size for tcn conv, use odd ints [5]
  --comment COMMENT     extra comments
```

## Recontruction examples
Edge-on images (upper row), AE reconstruction (middle), and residuals (lower row).

![alt text](https://github.com/jorgemarpa/PPDAE/blob/main/figures/Test_Recon_106050_52a73755.png)

Training logs and models https://app.wandb.ai/jorgemarpa/PPD-AE/overview

## Dependencies
We use poetry to manage dependencies and environment. First install poetry:

```
pip install poetry
```

Then install the necessary libraries listed in poetry.toml using the install command from poetry inside the repo directory:

```
poetry install
```

More info on how to use poetry -> https://python-poetry.org/docs/

## Sources and inspiration

* https://www.jeremyjordan.me/variational-autoencoders/
* https://github.com/wiseodd/generative-models
