# **[ICLR 2024] Towards Elminating Hard Label Constraints in Gradient Inverision Attacks**

This is the official repositaory for [ICLR 2024] accepted poster: Towards Elminating Hard Label Constraints in Gradient Inverision Attacks

arxiv link: 

## Prerequisites:
- python 3.9.13
- pytorch 1.12.1
- torchvision 0.13.1

## Code running:
It is **recommended** to try the recovery in jupyter notebooks presented in the files. **Core algorithms is coded in** `recovering.py`, to be specific, in `PSO` and `label_reco` functions. 

In `label_reco_demo.ipynb` we present a tiny example of label recovery, showing in details How to run the algorithm and get recovered labels.

In `label_rec_exp.ipynb` We present a full example of label recovery, where the results are shown in block outputs.

In `draw_pic_demo.ipynb` we show how to generate the pics in Figure 1.

In `reconstruct_demo.ipynb` we show how to reconstruct images from FCN-4 with well-recovered last-layer features analytically.
## Citations
If you find this code useful for your research, please cite our papers.
```
@inproceedings{
wang2024towards,
title={Towards Eliminating Hard Label Constraints in Gradient Inversion Attacks},
author={Yanbo Wang, Jian Liang, Ran He},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024}
}
``` 
