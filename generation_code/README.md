# Dataset generation
We provide scripts for generating the video dataset progression in our paper.
For generation, default storage space is `../data`. Feel free to modify in Line 336 of gen_syn_video.py and parse your own path.

## Video Generation
For a) Statical Objects:

`python gen_syn_video.py --large-scale --shape-model circle --static`

For b) Moving Circles:

`python gen_syn_video.py --large-scale --shape-model circle`

For c) Moving Shapes:

`python gen_syn_video.py --large-scale --shape-model mixed`

For d) Moving Transforming Shapes:

`python gen_syn_video.py --large-scale --shape-model mixed --affine`

For e) Accelerating Transforming Shapes:

`python gen_syn_video.py --large-scale --shape-model mixed --affine --acc`

For h) Accelerating Transforming Crops (texture / stylegan / imagenet)

`python gen_syn_video.py --large-scale --shape-model mixed --affine --acc --textured --texture-folder PATH_TO_YOUR_TEXTURES`


## StyleGAN & Synthetic textures
We borrowed the texture data from [learning_with_noise](https://github.com/mbaradad/learning_with_noise). (Download from their release)

## Other parameters
Video related parameters: `--fps`, `--duration`

## Resources and acceleration
The generation only reply on cpu now. For progression without textures, use `--parallel True` to accelerate the process. For dataset with textures added, we simply launch multiple process. Feel free to modify the parallel producure and accelerate the generation.

We will try to release larger (kinetics-400 size) dataset with textures added.



