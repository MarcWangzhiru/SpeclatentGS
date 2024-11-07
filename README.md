# SpeclatentGS
This repo contains the official implementation of the **ACM MM 2024** [paper](https://arxiv.org/abs/2409.05868) :

<div align="center">
<h1>
<b>
SpecGaussian with latent features: A high-quality modeling of the view-dependent appearance for 3D Gaussian Splatting
</b>
</h1>
<h4>
<b>
Zhiru Wang，Shiyun Xie，Chengwei Pan and Guoping Wang
</b>
</h4>
</div>

### PIPELINE 
![pipeline](/assets/pipeline.png)

## Environment Installation
You can install the base environment using:
```shell
git clone https://github.com/MarcWangzhiru/SpeclatentGS.git
cd SpeclatentGS
conda env create --file environment.yml
```
For the installation of **submodules**, you can use the following command: 
```shell
cd submodules/diff-gaussian-rasterization
python stup.py install
```
and
```shell
cd submodules/simple-knn
python stup.py install
```
You also need to install the **tinycudann** library. In general, you can use the following command:
```shell
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## Dataset Preparation
The dataset used in our method is in the same format as the dataset in Gaussian Splatting. If you want to use your custom dataset, follow the process of in [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting.git). We obtained our own [shiny_dataset]( https://drive.google.com/file/d/1mmBmptl9Pd8crLfO9y2E51F_R4Ecy0R1/view?usp=sharing) by resize the images of original [Shiny Dataset](https://vistec-my.sharepoint.com/:f:/g/personal/pakkapon_p_s19_vistec_ac_th/EnIUhsRVJOdNsZ_4smdhye0B8z0VlxqOR35IR3bp0uGupQ?e=TsaQgM) and recolmap.


## Trainng
For training, you can use the following command:
```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval
```
## Evalution
For evalution, you can use the following command:
```shell
python render.py -m <path to trained model> --eval
```


## Citation
