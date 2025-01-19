# Learning Video Representations without Natural Videos
<div style="text-align: center;">
  <p>
    <a href="https://scholar.google.com/citations?user=AIm87GIAAAAJ&hl=en">Xueyang Yu</a>, 
    <a href="https://xinleic.xyz/">Xinlei Chen</a>, 
    <a href="https://yossigandelsman.github.io/">Yossi Gandelsman</a>
  </p>
</div>

[[`Project Page`](https://unicorn53547.github.io/video_syn_rep/)] [[`arXiv`](https://arxiv.org/abs/2410.24213)]


[![Project](https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green)](https://unicorn53547.github.io/video_syn_rep/) 
[![arXiv](https://img.shields.io/badge/arXiv-2410.24213-A42C25?style=flat&logo=arXiv&logoColor=A42C25)](https://arxiv.org/abs/2410.24213)


<div align=center>
<img src="assert/datasets.png" style="width:100%;">
</div>


# TODO
- [ ] Robustness Eval code and scripts
- [ ] Full pretrain and finetune scripts
- [x] Code Release

# Requirements

To install all the requirements, simply do:

```
pip intall -r requirements.txt
```

This provided torch version in our training, other versions of torch and torchvision are likely to work.

# Dataset Generation and Preparation
We provide code and scripts for generating offline data in `generate_code` folder. Please refer to [Generation Scripts](generation_code/README.md) for detailed instructions.

To use offline generated dataset and other downloaded data for training, generate csv file and put them in `train_code/Annotations` folder. Example for generated dataset and real dataset are [synthetic.csv](train_code/Annotations/ucf101/synthetic.csv) and [ucf101_train.csv](train_code/Annotations/ucf101/train.csv).

# VideoMAE Pretrain

## Train VideoMAE on the fly
For simple progression, we support training while generating data on the fly. For example, to train with moving circles, do

```
cd train_code
sh scripts/ucf101/moving_circle.sh
```
Note you should first fill in the bash scripts with your log path and change parameters to your preference.

## Train VideoMAE with offline data
For more complex data (e.g. w/ affine transform, moving textures and image clips), we suggest generating offline dataset. The training process strictly follows the VideoMAE. Prepare datasets and place the csv, fill in the bash scripts and then run
```
cd train_code
sh scripts/ucf101/pretrain/train_dataset.sh
```

# VideoMAE Finetune
For finetuning with downstream dataset, prepare datasets following [VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/DATASET.md) and place csv in `train_code/Annotations` folder. Fill in corresponding bash scripts and run
```
cd train_code
sh scripts/ucf101/finetune/ft_hmdb.sh
```
# More Representation Evaluation
Additionaly, we use linear probe and corrupted pertubation to eval the quality and robustness of learned representation.

## Linear Probe
preparation is same as finetune process. Then run
```
cd train_code
sh scripts/ucf101/finetune/LP.sh
```
## Robustness Eval
Scripts to be released soon!


# Acknowledgements
We thank Amil Dravid and
Ren Wang for their valuable comments and feedback
on our paper; and thank UC Berkeley for the computational support to perform data processing and experiments. YG is supported by the Google Fellowship.

We thank the contributors to the following open-source projects.  Our project is impossible without the inspirations from these excellent researchers.
* [VideoMAE](https://github.com/MCG-NJU/VideoMAE)
* [Learning_with_Noise](learning_with_noise
)
* [Action Recognition Robustness Eval
](https://github.com/Maddy12/ActionRecognitionRobustnessEval)
# Citation
If you think this project is helpful, please feel free to leave a star⭐️ and cite our paper:
```
@article{yu2024learning,
  title={Learning Video Representations without Natural Videos},
  author={Yu, Xueyang and Chen, Xinlei and Gandelsman, Yossi},
  journal={arXiv e-prints},
  pages={arXiv--2410},
  year={2024}
}
```