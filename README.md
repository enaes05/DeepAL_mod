# DeepAL modification to study Head and Neck cancer

## Description
This is a modification of DeepAL, the original project can be found [here](https://github.com/ej0cl6/deep-active-learning).

This work is focused on the study of Head and Neck cancer. The file ```VAE_HeadAndNeck.ipynb``` is included, made in **Google Colab**, which details the
reduction of a dataset with dimensionality *564 x 60498*.

## Other necessary files
There are files that must be downloaded for trying some experiments:
1. [This](https://drive.google.com/file/d/1mBbC-spuR0JACDrIhL-ZJt-8KcijC4x9/view?usp=sharing) one is needed to run the ```VAE_HeadAndNeck.ipynb``` file. It is a dataset with negative (non-cancer) samples. The size is **13.9MB**.
2. [This](https://drive.google.com/file/d/1G93TilBkH7n4Gj83gjXTIKNghdOA_mFJ/view?usp=sharing) dataset is also needed for the ```VAE_HeadAndNeck.ipynb``` file. It is a dataset with positive (Head and Neck cancer) samples. The size is **162MB**.
3. [This](https://drive.google.com/file/d/1-3MtkhbSpSBe16_cyh50vJJQ1qlQyFZZ/view?usp=sharing) is the reduced dataset after using the Variational Autoencoder. It is necessary for executing *Deep Active Learning* experiments.
4. [This](https://drive.google.com/file/d/1XSXfeDr0givI04xVi3ankXvlyS3akYy6/view?usp=sharing) is an artificial dataset, with the same dimensionality as the previous one. The values are randomly generated. It can be used to check how the
algorithm behaves.

## Install
*DeepAL* can be tried in your computer or uploading the files to **Google Drive** and then using **Google Colab** to run the demo. If using your computer, you will
need these prerequisites:
- numpy 1.21.2
- scipy 1.7.1
- pytorch 1.10.0
- torchvision 0.11.1
- scikit-learn 1.0.1
- tqdm 4.62.3
- ipdb 0.13.9

You can also install a conda environment:
```conda env create -f environment.yml```

## Demo
To try with the Head and Neck cancer dataset:

```python demo.py --dataset_name HeadAndNeck --n_init_labeled 10 --n_query 6 --n_round 10 --strategy_name RandomSampling```

## Citing
Please do not forget to cite the original project, this is just a modification to study a concrete case:

```
@article{Huang2021deepal,
    author    = {Kuan-Hao Huang},
    title     = {DeepAL: Deep Active Learning in Python},
    journal   = {arXiv preprint arXiv:2111.15258},
    year      = {2021},
}
```
