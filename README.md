# NestConvNeXt-IQA

Pytorch implementation of NestConvNeXt-IQA paper "No-Reference LWIR Image Quality Assessment via NestConvNeXt and Lightweight Dual Attention Networks"



## Datasets

In this work, two LWIR datasets are used: `LWIR-IQA` and `TIIQAD`, which represent synthetically and authentically distorted IQA databases. Two evaluation setups are considered, corresponding to authentic and synthetic distortions. For the synthetic setup, both global and per-distortion-type evaluations are performed, including white noise, blurring, non-uniformity, JPEG, and JPEG2000 compression.

1. The [LWIR-IQA](https://doi.org/10.2478/jee-2022-0011) image dataset used during the current study is available in the GitHub Repository: https://github.com/azedomar/LWIR_IQA_dataset. 
2. The thermal dataset [TIIQAD](https://doi.org/10.1109/ICIP51287.2024.10648145) employed in this study is also publicly available on GitHub: https://github.com/cheunglaihip/TIIQAD.


## IQA algorithms

To validate the performance of the proposed method, 16 state-of-the-art IQA methods were used for comparison, including traditional full-reference, deep learning full-reference, traditional no-reference, and deep learning no-reference 
### Traditional full-reference (TR-FR) metrics:
1. `VIF:`
   [H. R. Sheikh and A. C. Bovik, “Image information and visual quality,” IEEE Trans. Image Process. vol. 15, no 2, p. 430-444, 2006, doi: 10.1109/TIP.2005.859378](https://doi.org/10.1109/TIP.2005.859378)

2. `IW SSIM:`
   [Z. Wang and Q. Li, “Information content weighting for perceptual image quality assessment,” IEEE Trans. Image Process. vol. 20, no. 5, pp. 1185–1198, 2011, doi: 10.1109/TIP.2010.2092435.](https://doi.org/10.1109/TIP.2010.2092435)
4. `IW PSNR:`
   [Z. Wang and Q. Li, “Information content weighting for perceptual image quality assessment,” IEEE Trans. Image Process. vol. 20, no. 5, pp. 1185–1198, 2011, doi: 10.1109/TIP.2010.2092435.](https://doi.org/10.1109/TIP.2010.2092435)

5. `GMSD:`
   [W. Xue, L. Zhang, X. Mou, and A. C. Bovik, “Gradient magnitude similarity deviation: A highly efficient perceptual image quality index,” IEEE Trans. Image Process. vol. 23, no. 2, pp. 684–695, 2014, doi: 10.1109/TIP.2013.2293423.](https://doi.org/10.1109/TIP.2013.2293423)

6. `FSIM:`
   [L. Zhang, L. Zhang, X. Mou, and D. Zhang, “FSIM: A feature similarity index for image quality assessment,” IEEE Trans. Image Process. vol. 20, no. 8, pp. 2378–2386, 2011, doi: 10.1109/TIP.2011.2109730.](https://doi.org/10.1109/TIP.2011.2109730)

### Deep learning full-reference (DL-FR) metric:
6. `SCIQA:` 
   [W. Xian et al., “A style transfer-based fast image quality assessment method for image sensors,” Sensors, vol. 25, no. 16, art. no. 5121, Aug. 2025, doi: 10.3390/s25165121.](https://doi.org/10.3390/s25165121)

### Traditional no-reference (TR-NR) metrics:
7. `NIQE:`
   [A. Mittal, R. Soundararajan, and A. C. Bovik, “Making a ‘completely blind’ image quality analyzer,” IEEE Signal Process. Lett. vol. 20, no. 3, pp. 209–212, 2013, doi: 10.1109/LSP.2012.2227726.](https://doi.org/10.1109/LSP.2012.2227726)

8. `BRISQUE:`
   [A. Mittal, A. K. Moorthy, and A. C. Bovik, “No-reference image quality assessment in the spatial domain,” IEEE Trans. Image Process. vol. 21, no. 12, pp. 4695–4708, 2012, doi: 10.1109/TIP.2012.2214050.](https://doi.org/10.1109/TIP.2012.2214050)

9. `OG-IQA:`
   [L. Liu, Y. Hua, Q. Zhao, H. Huang, and A. C. Bovik, “Blind image quality assessment by relative gradient statistics and adaboosting neural network,” Signal Process. Image Commun. vol. 40, pp. 1–15, 2016, doi: 10.1016/j.image.2015.10.005.](https://doi.org/10.1016/j.image.2015.10.005)

10. `GM-LOG:`
    [W. Xue, X. Mou, L. Zhang, A. C. Bovik, and X. Feng, “Blind image quality assessment using joint statistics of gradient magnitude and Laplacian features,” IEEE Trans. Image Process. vol. 23, no. 11, pp. 4850–4862, 2014, doi: 10.1109/TIP.2014.2355716.](https://doi.org/10.1109/TIP.2014.2355716)

11. `GWH-GLBP:`
    [Q. Li, W. Lin, and Y. Fang, “No-reference quality assessment for multiply-distorted images in gradient domain,” IEEE Signal Process. Lett. vol. 23, no. 4, pp. 541–545, Apr. 2016, doi: 10.1109/LSP.2016.2537321.](https://doi.org/10.1109/LSP.2016.2537321)

### Deep learning no-reference (DL-NR) metrics:
12. `HyperNet:`
    [S. Su et al., “Blindly assess image quality in the wild guided by a self-adaptive hyper network,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 3664–3673. doi: 10.1109/CVPR42600.2020.00372.](https://doi.org/10.1109/CVPR42600.2020.00372)

13. `TReS:`
    [S. A. Golestaneh, S. Dadsetan, and K. M. Kitani, “No-reference image quality assessment via transformers, relative ranking, and Self-consistency,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), Waikoloa, HI, USA, 2022, pp. 3989–3999. doi: 10.1109/WACV51458.2022.00404.](https://doi.org/10.1109/WACV51458.2022.00404)

14. `DACNN:`
    [Z. Pan et al., “DACNN: Blind image quality assessment via a distortion-aware convolutional neural network,” IEEE Trans. Circuits Syst. Video Technol. vol. 32, no. 11, pp. 7518–7531, 2022, doi: 10.1109/TCSVT.2022.3188991.](https://doi.org/10.1109/TCSVT.2022.3188991)

15. `VCRNet:`
    [Z. Pan, F. Yuan, J. Lei, Y. Fang, X. Shao, and S. Kwong, “VCRNet: Visual compensation restoration network for no-reference image quality assessment,” IEEE Trans. Image Process. vol. 31, pp. 1613–1627, 2022, doi: 10.1109/TIP.2022.3144892.](https://doi.org/10.1109/TIP.2022.3144892)

16. `SaTQA:`
    [J. Shi, P. Gao, and J. Qin, “Transformer-based no-reference image quality assessment via supervised contrastive learning,” Proc. AAAI Conf. Artif. Intell. vol. 38, no. 5, pp. 4829–4837, 2024, doi: 10.1609/aaai.v38i5.28285.](https://doi.org/10.1609/aaai.v38i5.28285)


## Usage Instructions

### For Training and Testing
- Download the LWIR-IQA and/or the TIIQAD datasets.
- Open the file: `RunTrain.ipynb`

#### For the LWIR-IQA dataset, run:
```
%run TrainTest.py --batch_size 40 --svpath '/path_where_to_save_results/' --epochs 20 --lr 2e-5 --gpunum 0 --datapath ./Path_to_LWIR-IQA_dataset/ --dataset LWIR_IQA --version 1 --seed 2030
```

##### For the TIIQAD dataset, run:
```
%run TrainTest.py --batch_size 40 --svpath '/path_where_to_save_results/' --epochs 20 --lr 2e-5 --gpunum 0 --datapath ./Path_to_TIIQAD_dataset/ --dataset TIIQAD --version 1 --seed 2030
```

### Testing on a single image

- Open the file `RunTest_predict_one_image.ipynb` 
- Set the path to the trained model. You may also use the provided pretrained models for `LWIR-IQA` and `TIIQAD`.
- Set the path to the image to be predicted.
- Run the notebook.

## Results

Some of our results are provided in the `./Results/` folder.

This folder contains spider diagrams summarizing the performance comparison under distortion-specific analysis. These diagrams are presented for:
- all IQA algorithms,
- full-reference (FR) algorithms,
- traditional no-reference (TR-NR) algorithms,
- deep learning-based no-reference (DL-NR) algorithms.
   
In addition, the `Examples.docx` file includes sample images from the `LWIR-IQA` and `TIIQAD` datasets. For each example, the corresponding subjective score (MOS) and the objective score predicted by `NestConvNeXt-IQA` are provided.


### Examples from LWIR-IQA and TIIQAD datasets
The following images illustrate the performance of NestConvNeXt-IQA on the LWIR-IQA and TIIQAD datasets.

1) LWIR-IQA dataset (Synthetic Distortions)
This dataset includes five types of synthetic distortions: white noise (AWGN), blur (BLU), non-uniformity (NU), JPEG (JPG), and JPEG2000 (J2K).

#### Non-uniformity

| **Visual Sample** | ![NU 1](Results/images/IMG_09_NU%20(1).bmp) | ![NU 2](Results/images/IMG_09_NU%20(2).bmp) | ![NU 4](Results/images/IMG_09_NU%20(4).bmp) |
| **Filename** | `IMG_09_NU (1).bmp` | `IMG_09_NU (2).bmp` | `IMG_09_NU (4).bmp` |
| **MOS** | 4.645 | 4.516 | 2.226 |
| **NestConvNeXt** | **4.691** | **4.435** | **2.344** |


| **Visual Sample** | ![NU 1](Results/images/IMG_20_NU%20(1).bmp) | ![NU 3](Results/images/IMG_20_NU%20(3).bmp) | ![NU 4](Results/images/IMG_20_NU%20(4).bmp) |
| **Filename** | `IMG_20_NU (1).bmp` | `IMG_20_NU (3).bmp` | `IMG_20_NU (4).bmp` |
| **MOS** | 3.742 | 2.613 | 2.129 |
| **NestConvNeXt** | **3.749** | **2.681** | **2.162** |

#### White Noise (AWGN)

| Metric | Sample 1 | Sample 2 | Sample 3 |
| :--- | :---: | :---: | :---: |
| **Visual Sample** | ![AWGN 1](Results/images/IMG_12_AWGN%20(1).bmp) | ![AWGN 2](Results/images/IMG_12_AWGN%20(2).bmp) | ![AWGN 5](Results/images/IMG_12_AWGN%20(5).bmp) |
| **Filename** | `IMG_12_AWGN (1).bmp` | `IMG_12_AWGN (2).bmp` | `IMG_12_AWGN (5).bmp` |
| **MOS** | 4.677 | 4.484 | 2.290 |
| **NestConvNeXt** | **4.442** | **4.416** | **2.618** |

#### Blur

| Metric | Sample 1 | Sample 2 |
| :--- | :---: | :---: |
| **Visual Sample** | ![BLU 2](Results/images/IMG_10_BLU%20(2).bmp) | ![BLU 4](Results/images/IMG_10_BLU%20(4).bmp) |
| **Filename** | `IMG_10_BLU (2).bmp` | `IMG_10_BLU (4).bmp` |
| **MOS** | 4.226 | 2.871 |
| **NestConvNeXt** | **4.376** | **2.708** |

#### JPEG2000

| Metric | Sample 1 | Sample 2 |
| :--- | :---: | :---: |
| **Visual Sample** | ![J2K 1](Results/images/IMG_07_J2K%20(1).bmp) | ![J2K 5](Results/images/IMG_07_J2K%20(5).bmp) |
| **Filename** | `IMG_07_J2K (1).bmp` | `IMG_07_J2K (5).bmp` |
| **MOS** | 4.194 | 1.548 |
| **NestConvNeXt** | **4.205** | **1.635** |

#### JPEG Compression

| Metric | Sample 1 | Sample 2 |
| :--- | :---: | :---: |
| **Visual Sample** | ![JPG 1](Results/images/IMG_18_JPG%20(1).bmp) | ![JPG 4](Results/images/IMG_18_JPG%20(4).bmp) |
| **Filename** | `IMG_18_JPG (1).bmp` | `IMG_18_JPG (4).bmp` |
| **MOS** | 3.161 | 1.419 |
| **NestConvNeXt** | **3.223** | **1.658** |

2) TIIQAD dataset (authentic Distortions)

| Metric | Sample 1 | Sample 2 | Sample 3 |
| :--- | :---: | :---: | :---: |
| **Visual Sample** | ![Sample 138](Results/images/138.jpg) | ![Sample 1508](Results/images/1508.jpg) | ![Sample 952](Results/images/952.jpg) |
| **Filename** | `138.jpg` | `1508.jpg` | `952.jpg` |
| **MOS** | 2.637 | 2.837 | 2.947 |
| **NestConvNeXt** | **2.677** | **2.923** | **2.913** |


| Metric | Sample 1 | Sample 2 | Sample 3 |
| :--- | :---: | :---: | :---: |
| **Visual Sample** | ![Sample 1445](Results/images/1445.jpg) | ![Sample 1022](Results/images/1022.jpg) | ![Sample 1002](Results/images/1002.jpg) |
| **Filename** | `1445.jpg` | `1022.jpg` | `1002.jpg` |
| **MOS** | 3.034 | 3.094 | 3.102 |
| **NestConvNeXt** | **3.005** | **3.064** | **3.098** |


| Metric | Sample 1 | Sample 2 | Sample 3 |
| :--- | :---: | :---: | :---: |
| **Visual Sample** | ![Sample 248](Results/images/248.jpg) | ![Sample 253](Results/images/253.jpg) | ![Sample 413](Results/images/413.jpg) |
| **Filename** | `248.jpg` | `253.jpg` | `413.jpg` |
| **MOS** | 2.516 | 2.898 | 3.449 |
| **NestConvNeXt** | **2.614** | **2.876** | **3.254** |


| Metric | Sample 1 | Sample 2 | Sample 3 |
| :--- | :---: | :---: | :---: |
| **Visual Sample** | ![Sample 450](Results/images/450.jpg) | ![Sample 2248](Results/images/2248.jpg) | ![Sample 1039](Results/images/1039.jpg) |
| **Filename** | `450.jpg` | `2248.jpg` | `1039.jpg` |
| **MOS** | 2.514 | 3.034 | 3.111 |
| **NestConvNeXt** | **2.750** | **2.942** | **3.053** |

| Metric | Sample 1 | Sample 2 | Sample 3 |
| :--- | :---: | :---: | :---: |
| **Visual Sample** | ![Sample 421](Results/images/421.jpg) | ![Sample 224](Results/images/224.jpg) | ![Sample 2289](Results/images/2289.jpg) |
| **Filename** | `421.jpg` | `224.jpg` | `2289.jpg` |
| **MOS** | 2.766 | 2.903 | 3.034 |
| **NestConvNeXt** | **2.768** | **2.800** | **2.948** |

