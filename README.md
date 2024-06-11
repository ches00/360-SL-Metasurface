# 360° Structured Light with Learned Metasurfaces 

### [Project Page](https://eschoi.com/360-SL-Metasurface/) | [Paper](https://www.nature.com/articles/s41566-024-01450-x) | [Data](https://doi.org/10.5281/zenodo.11518075) | [Arxiv](https://arxiv.org/abs/2306.13361)

[Eunsue Choi*](https://eschoi.com), [Gyeongtae Kim*](https://scholar.google.co.kr/citations?user=0rZekfsAAAAJ), [Jooyeong Yun](https://scholar.google.com/citations?user=iw2cTTYAAAAJ), [Yujin Jeon](https://scholar.google.com/citations?user=M9ZnHHoAAAAJ), [Junsuk Rho+](https://sites.google.com/site/junsukrho/), [Seung-Hwan Baek+](https://www.shbaek.com/)

dataset: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11518075.svg)](https://doi.org/10.5281/zenodo.11518075)



This repository contains the implementation and supplementary files of the paper:

**360° Structured Light with Learned Metasurfaces**

Eunsue Choi *, Gyeongtae Kim *, Jooyeong Yun, Yujin Jeon, Junsuk Rho+, Seung-Hwan Baek+

***Nature Photonics, 2024***



This code implements a 180° full-space light propagation model for simulating 360° structured light, differentiable fish-eye camera rendering and 360° depth reconstructor in pytorch. These components are optimized end-to-end using machine learning optimizers. 



## Dataset 

We have provided synthetic fish-eye images rendered in blender. 

You can download the train and test datasets from [this link](https://zenodo.org/records/5637679) and place them in the corresponding folders.

We have also included the fish-eye camera configuration for this dataset in a file named "calib_results.txt" within this repository.

If you wish to use your own dataset, please replace the configuration file with your calibration results.



## Training 

To perform end-to-end training (of metasurface and 3D reconstructor) execute the 'train.py':

```bash
python train.py
```

Please, refer to the details of the arguments in utils/ArgParser.py 

We have provided several supplementary files for training: 

- synthetic fisheye dataset 
- __checkpoint/pattern.mat__: Optimized meta-surface phase map 
- __calib_results.txt__ : Calibration result of the dataset 
- __fisheye_mask.npy__: Validation mask of the fisheye-camera dataset for the given camera-lens 

If you perform your own end-to-end training, please replace those files with your own. 



## Testing 

Our implementation performs two types of inference: for synthetic images and real-world captures. To perform inference on your images, execute 'test.py':

```bash
python test.py
```

Please, refer to the details of the arguments in utils/ArgParser.py 



## Requirements

This code has been trained and tested on Linux with an Nvidia A6000 GPU with 48GB of VRAM.

We recommend using a Conda environment. Build a virtual environment based on the YAML configuration file provided in this repository.

```bash 
conda env create --file environment.yaml 
```



## Citation

If you find our work useful in your research, please cite:

```
TBU
```



## Acknowledgement

Part of our code is based on the other works: [sphere-stereo](https://github.com/KAIST-VCLAB/sphere-stereo), [OcamCalib](https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab), [Omnimvs](https://github.com/hyu-cvlab/omnimvs-pytorch), and [polka_lines](https://openaccess.thecvf.com/content/CVPR2021/html/Baek_Polka_Lines_Learning_Structured_Illumination_and_Reconstruction_for_Active_Stereo_CVPR_2021_paper.html).

Our dataset was rendered with 3D object from [ShapeNet](https://shapenet.org/)
