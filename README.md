# NeuroTranslate

# Surface Vision Transformers for MRI brain representation translations

This repo is contains scripts to apply various deep learning models that attempt to translate from one brain representaion to another. **So far, only the trasnlation from MRI brain meshes to functional connectivity matrices is actively being worked on**. As such, the current README will only walk through this translation process.

<!-- 
<img src="./docs/sit_gif.gif"
     alt="Surface Vision Transformers"
     style="float: left; margin-right: 10px;" /> -->

# Updates
<details>
    <summary><b> V.1.0.1</b></summary>
    Initial commits - 04.03.24
    <ul type="circle">
        <li> Changing the patch embedding for each sphere, prior way did not 
        embed each sphere correctly for the ICA brain map. </li>
    </ul>
</details>

<details>
    <summary><b> V.1.0.0 - 04.03.24</b></summary>
    Major codebase update - 04.03.24
    <ul type="circle">
        <li> Adding ICA mesh -> Schaeffer netmat translation</li>
        <li> Modifying preprocessing script (curr no normalization of ico sphere data) </li>
        <li> ingle config file tasks (surf2mat) and data configurations (template because we use HCP mesh template)</li>
        <li> adding mesh indices to extract non-overlapping triangular patches from a cortical mesh ico 6 sphere representation</li>
        <li> training script </li>
        <li> README </li>
        <li> config file for training </li>
    </ul>
</details>

# Installation & Set-up

This follows the a similat set up as the original repo that this is based off of. Namely, follow these same steps:

**Python and Conda**

Make sure you have python and conda in your OS. Create a conda env:
```
conda env create -f environment.yml
```
You can call it what you want. To do so, change the name in the `environment.yml` file. Also, if `-f` fails, you can try `--file=environment.yml` instead.

# Brain Mesh Data

Data in this projects is expexted to come from some MRI brain representation. For now, its only MRI brain meshes -> network matrices. So, the data path you need should be HCP styled data where each subject should have `subj_ID.shape.gii` file that represents that geometric mesh data. 

You need to first resample the brain_maps into a tessellated sphere with batches of equal size. Your brain maps should contained within the cifti file dtseries.nii
which is a 'dense time series' cifti file - don't be confused by the nii, its NOT a volume file in the way regular nii or nii.gz files are. The `./data/chpc_resample_sphere_fs_space.sh` script is meant to be used to conver brain maps of contained in the brain_representation folder into left and right spheres that represent left and right cortex, respectively. For us, these are ICA brain maps.
info can be found in:
1. https://manpages.debian.org/testing/connectome-workbench/wb_command.1.en.html#Maps
2. https://www.humanconnectome.org/software/workbench-command/-surface-create-sphere
3. https://www.humanconnectome.org/software/workbench-command/-metric-resample

Note that the `./data/chpc_resample_sphere_fs_space.sh` uses commands from work bench, so it cannot run if you don't have workbench. See above to figure out how to install it into your machine or, ideally, you work off of a High Performance Computer cluster that already has the software.

Once the brain meshes are converted into two spheres (one for each hemisphere), we can move on to preprocess the data.

## Preprocessing spheres (tesselating into ico-N spheres)

As in the original paper(s), we will be resampling these large spheres into ico-2 spheres of 320 patches with 153 veteces each. the `./tools/preprocessing.py` script is meant to do this, given a setting `.yml` file. Example:
```
python3 preprocessing.py ../config/preprocessing/ICAd25_schfd200.yml 
```
The above code will preprocess the ICA25 spheres to prep them for a Schaeffer 200 translation. In that script, there is a way to loop them, or you can choose the manual way like I show above.

## 1. Accessing processed data

Cortical surface metrics already processed as in [S. Dahan et al 2021](https://arxiv.org/abs/2203.16414) and [A. Fawaz et al 2021](https://www.biorxiv.org/content/10.1101/2021.12.01.470730v1) are available upon request. 

<!-- <details>
    <summary><b> How to access the processed data?</b></summary>
    <p>
    To access the data please:
    <br>
        <ul type="circle">
            <li>Sign in <a href="https://data.developingconnectome.org/app/template/Login.vm">here</a> </li>
            <li>Sign the dHCP open access agreement </li>
            <li> Forward the confirmation email to <b> slcn.challenge@gmail.com</b>  </li>
        </ul>
    </br>
    </p>
</details> -->

# Training & Inference

## Training SiT

For training a SiT model, use the following command:

```
python train.py ../config/SiT/training/hparams.yml
```
Where all hyperparameters for training and model design models are to be set in the yaml file `config/preprocessing/hparams.yml`, such as: 

- Transformer architecture
- Training strategy: from scratch, ImageNet or SSL weights
- Optimisation strategy
- Patching configuration
- Logging

## Testing SiT

For testing a SiT model, please put the path of the SiT weights in /testing/hparams.yml and use the following command: 

```
python test.py ../config/SiT/training/hparams.yml
```


# References 

This repo is largely based off of previous work from others, namely the Surface image Transformer code from Dahan et al., 2022 (https://arxiv.org/abs/2203.16414 & https://arxiv.org/abs/2204.03408) with original repo at: https://github.com/metrics-lab/surface-vision-transformers

# Citation

Please cite these works if you found it useful:

[Surface Vision Transformers: Attention-Based Modelling applied to Cortical Analysis](https://arxiv.org/abs/2203.16414)

```
@article{dahan2022surface,
  title={Surface Vision Transformers: Attention-Based Modelling applied to Cortical Analysis},
  author={Dahan, Simon and Fawaz, Abdulah and Williams, Logan ZJ and Yang, Chunhui and Coalson, Timothy S and Glasser, Matthew F and Edwards, A David and Rueckert, Daniel and Robinson, Emma C},
  journal={arXiv preprint arXiv:2203.16414},
  year={2022}
}
```
[Surface Vision Transformers: Flexible Attention-Based Modelling of Biomedical Surfaces](https://arxiv.org/abs/2204.03408)

```
@article{dahan2022surface,
  title={Surface Vision Transformers: Flexible Attention-Based Modelling of Biomedical Surfaces},
  author={Dahan, Simon and Xu, Hao and Williams, Logan ZJ and Fawaz, Abdulah and Yang, Chunhui and Coalson, Timothy S and Williams, Michelle C and Newby, David E and Edwards, A David and Glasser, Matthew F and others},
  journal={arXiv preprint arXiv:2204.03408},
  year={2022}
}
```

This project is still in progress.
