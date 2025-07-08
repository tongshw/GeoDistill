<h1 align="center"><strong>GeoDistill: Geometry-Guided Self-Distillation for Weakly Supervised Cross-View Localization</strong></h1>

<p align="center">
  <a href="https://arxiv.org/abs/2308.16906" target='_blank'>
    <img src="https://img.shields.io/badge/arXiv-2308.16906-orange?">
  </a> 
  <a href="https://arxiv.org/pdf/2308.16906.pdf" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-📖-green?">
  </a>
  <a href="#citation">
    <img src="https://img.shields.io/badge/Citation-🔗;-blue">
  </a>
</p>

## 🏠 About

![image-20230831214545912](./assests/architecture.png)

We introduce a novel approach to fine-grained cross-view geo-localization. Our method **aligns a warped ground image with a corresponding GPS-tagged satellite image covering the same area using homography estimation.** We first employ a differentiable **spherical transform**, adhering to geometric principles, to **accurately align the perspective of the ground image with the satellite map.** To address challenges such as occlusion, small overlapping range, and seasonal variations, we propose a robust correlation-aware homography estimator to align similar parts of the transformed ground image with the satellite image. Our method achieves **sub-pixel resolution and meter-level GPS accuracy** by mapping the center point of the transformed ground image to the satellite image using a homography matrix and determining the orientation of the ground camera using a point above the central axis. Operating at a speed of 30 FPS, our method outperforms state-of-the-art techniques, **reducing the mean metric localization error by 21.3% and 32.4%** in same-area and cross-area generalization tasks on the VIGOR benchmark, respectively, and by **34.4% on the KITTI benchmark in same-area evaluation.**

## 📦 Checkpoints
📁 [**Download pretrained models**](https://drive.google.com/drive/folders/1pPaECfpH3H1_hPc7bDbT2X7oZ5-_HyH9?usp=drive_link)


## 🚀 Training

### Set up

We train and test our codes under the following environment:

- Ubuntu 18.04
- CUDA 12.0
- Python 3.8.16
- PyTorch 1.13.0

To get started, follow these steps: 

1. Clone this repository.

```bash
git clone https://github.com/tongshw/GeoDistill.git
cd GeoDistill
```

2. Install the required packages.

```bash
conda create -n geodistill python=3.9 -y
conda activate geodistill
pip install -r requirements.txt
```

### Training
We released our implementation of G2SWeakly and GeoDistill with G2SWeakly both VGG and DINO variants. 

**We apply mask in both ground image and feature maps when base model is G2SWeakly, and when the base model is CCVPE, we apply mask to the descriptor.**
```bash
# to train our implemented G2SWeakly in VIGOR cross area
python -u train_vigor.py --train True --train_g2sweakly True --cross_area True


# to train orientation estimator
python -u train_orientation.py --train True --cross_area True
```

## 🎉 Evaluation

### 2-DoF Evaluation

To evaluate the Geodistill model, follow these steps:

1. Download the [VIGOR](https://github.com/Jeff-Zilence/VIGOR) dataset and set its path to '/home/< usr >/Data/VIGOR'.
2. Download the [pretrained models](https://drive.google.com/drive/folders/1pPaECfpH3H1_hPc7bDbT2X7oZ5-_HyH9?usp=drive_link) and place them in the './checkpoints/VIGOR '.
3. Run the following command:

````bash
chmod +x val.sh
# Usage: val.sh [same|cross]
# For same-area in VIGOR
./val.sh same 0
# For cross-area in VIGOR
./val.sh cross 0
````


### 3-DoF Evaluation

````bash
chmod +x val.sh
# Usage: val.sh [same|cross]
# For same-area in VIGOR
./val.sh same 0
# For cross-area in VIGOR
./val.sh cross 0
````



<h2 id="citation">🔗 Citation</h2>

If you find our work helpful, please cite:

```bibtex
@article{wang2024fine,
  title={Fine-Grained Cross-View Geo-Localization Using a Correlation-Aware Homography Estimator},
  author={Wang, Xiaolong and Xu, Runsen and Cui, Zhuofan and Wan, Zeyu and Zhang, Yu},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

## 👏 Acknowledgements

- This work is based on [G2SWeakly](https://github.com/yujiaoshi/g2sweakly) and [CCVPE](https://github.com/tudelft-iv/CCVPE), we thank the authors for the contribution.
