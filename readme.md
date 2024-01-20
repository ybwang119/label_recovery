# **[ICLR 2024] Towards Elminating Hard Label Constraints in Gradient Inverision Attacks**
This is the official repositaory for [ICLR 2024] accepted poster.

arxiv link: 

### Abstract: 
Gradient inversion attacks aim to reconstruct local training data from intermediate gradients exposed in the federated learning framework. Despite successful attacks, all previous methods, starting from reconstructing a single data point and then relaxing the single-image limit to batch level, are only tested under hard label constraints. Even for single-image reconstruction, we still lack an analysis-based algorithm to recover augmented soft labels. In this work, we change the focus from enlarging batchsize to investigating the hard label constraints, considering a more realistic circumstance where label smoothing and mixup techniques are used in the training process. In particular, we are the first to initiate a novel algorithm to simultaneously recover the ground-truth augmented label and the input feature of the last fully-connected layer from single-input gradients, and provide a necessary condition for any analytical-based label recovery methods. Extensive experiments testify to the label recovery accuracy, as well as the benefits to the following image reconstruction. We believe soft labels in classification tasks are worth further attention in gradient inversion attacks.
## Prerequisites:
- python 3.9.13
- pytorch 1.12.1
- torchvision 0.13.1

## Code running:
It is **recommended** to try the recovery algorithm in jupyter notebooks presented in the files. 

**Core algorithms is coded in** `recovering.py`, to be specific, in `PSO` and `label_reco` functions. 

In `label_reco_demo.ipynb` we present a tiny example of label recovery, showing in details How to run the algorithm and get recovered labels.

In `label_rec_exp.ipynb` We present a full example of label recovery, where the results are shown in block outputs.

In `draw_pic_demo.ipynb` we show how to generate the pics in Figure 1.

In `reconstruct_demo.ipynb` we show how to reconstruct images from FCN-4 with well-recovered last-layer features analytically. Pics are identical to those in Figure 3.

Our codes are developed based on [IG Repository](https://github.com/JonasGeiping/invertinggradients) and [DLG Repository](https://github.com/mit-han-lab/dlg). Sincerely thanks for their contributions to the community!
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
