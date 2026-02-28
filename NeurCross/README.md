# NeurCross: A Neural Approach to Computing Cross Fields for Quad Mesh Generation

### [Project](https://qiujiedong.github.io/publications/NeurCross/) | [Paper](https://arxiv.org/pdf/2405.13745)

**This repository is the official PyTorch implementation of our
paper,  *NeurCross: A Neural Approach to Computing Cross Fields for Quad Mesh Generation*, ACM Transactions on Graphics (SIGGRAPH 2025).**

<img src='./assets/NeurCross.jpg'>

## Requirements

- python 3.7
- CUDA 11.7
- Pytorch 1.13.1

## Installation

```
git clone https://github.com/QiujieDong/NeurCross.git
cd NeurCross
```

## Overfitting

```
cd quad_mesh
python train_quad_mesh.py
```

All parameters are set in the ```quad_mesh_args.py```.


## Extraction

The extractor is from [libigl](https://libigl.github.io/)
and [libQEx](https://github.com/hcebke/libQEx).


## Cite

If you find our work useful for your research, please consider citing our paper.

```bibtex
@article{Dong2025NeurCross,
author={Dong, Qiujie and Wen, Huibiao and Xu, Rui and Chen, Shuangmin and Zhou, Jiaran and Xin, Shiqing and Tu, Changhe and Komura, Taku and Wang, Wenping},
title={NeurCross: A Neural Approach to Computing Cross Fields for Quad Mesh Generation},
journal={ACM Trans. Graph.},
publisher={Association for Computing Machinery},
address={New York, NY, USA},
year={2025},
volume={44},
number={4},
url={https://doi.org/10.1145/3731159},
doi={10.1145/3731159}
}
```



## Acknowledgments

Our code is inspired by [NeurCADRecon](https://github.com/QiujieDong/NeurCADRecon)
and [SIREN](https://github.com/vsitzmann/siren).
We would like to thank the authors
of [libigl](https://libigl.github.io/)
and [libQEx](https://github.com/hcebke/libQEx).

