# NestConvNeXt-IQA

Pytorch implementation of NestConvNeXt-IQA paper "No-Reference LWIR Image Quality Assessment via NestConvNeXt and Lightweight Dual Attention Networks"



## Datasets

In this work, two LWIR datasets are used: LWIR-IQA and TIIQAD, which represent synthetically and authentically distorted IQA databases. Two evaluation setups are considered, corresponding to authentic and synthetic distortions. For the synthetic setup, both global and per-distortion-type evaluations are performed, including white noise, blurring, non-uniformity, JPEG, and JPEG2000 compression.

1. The LWIR-IQA image dataset used during the current study is available in the GitHub Repository: https://github.com/azedomar/LWIR_IQA_dataset. 
2. The thermal dataset TIIQAD employed in this study is also publicly available on GitHub: https://github.com/cheunglaihip/TIIQAD.


## IQA algorithms

To validate the performance of the proposed method, 16 state-of-the-art IQA methods were used for comparison, including traditional full-reference, deep learning full-reference, traditional no-reference, and deep learning no-reference 
### Traditional full-reference (TR-FR) metrics:
1.  VIF
    H. R. Sheikh and A. C. Bovik, “Image information and visual quality,” IEEE Trans. Image Process. vol. 15, no 2, p. 430-444, 2006, doi: 10.1109/TIP.2005.859378.

2. IW SSIM
3. IW PSNR
   Z. Wang and Q. Li, “Information content weighting for perceptual image quality assessment,” IEEE Trans. Image Process. vol. 20, no. 5, pp. 1185–1198, 2011, doi: 10.1109/TIP.2010.2092435.


4. GMSD
   W. Xue, L. Zhang, X. Mou, and A. C. Bovik, “Gradient magnitude similarity deviation: A highly efficient perceptual image quality index,” IEEE Trans. Image Process. vol. 23, no. 2, pp. 684–695, 2014, doi: 10.1109/TIP.2013.2293423.

5. FSIM
   L. Zhang, L. Zhang, X. Mou, and D. Zhang, “FSIM: A feature similarity index for image quality assessment,” IEEE Trans. Image Process. vol. 20, no. 8, pp. 2378–2386, 2011, doi: 10.1109/TIP.2011.2109730.

### Deep learning full-reference (DL-FR):
6. SCIQA 
   W. Xian et al., “A style transfer-based fast image quality assessment method for image sensors,” Sensors, vol. 25, no. 16, art. no. 5121, Aug. 2025, doi: 10.3390/s25165121.

### Traditional no-reference (TR-NR) metrics:
7. NIQE
   A. Mittal, R. Soundararajan, and A. C. Bovik, “Making a ‘completely blind’ image quality analyzer,” IEEE Signal Process. Lett. vol. 20, no. 3, pp. 209–212, 2013, doi: 10.1109/LSP.2012.2227726.

8. BRISQUE
   A. Mittal, A. K. Moorthy, and A. C. Bovik, “No-reference image quality assessment in the spatial domain,” IEEE Trans. Image Process. vol. 21, no. 12, pp. 4695–4708, 2012, doi: 10.1109/TIP.2012.2214050.

9. OG-IQA
   L. Liu, Y. Hua, Q. Zhao, H. Huang, and A. C. Bovik, “Blind image quality assessment by relative gradient statistics and adaboosting neural network,” Signal Process. Image Commun. vol. 40, pp. 1–15, 2016, doi: 10.1016/j.image.2015.10.005.

10. GM-LOG
    W. Xue, X. Mou, L. Zhang, A. C. Bovik, and X. Feng, “Blind image quality assessment using joint statistics of gradient magnitude and Laplacian features,” IEEE Trans. Image Process. vol. 23, no. 11, pp. 4850–4862, 2014, doi: 10.1109/TIP.2014.2355716.

11. GWH-GLBP
    Q. Li, W. Lin, and Y. Fang, “No-reference quality assessment for multiply-distorted images in gradient domain,” IEEE Signal Process. Lett. vol. 23, no. 4, pp. 541–545, Apr. 2016, doi: 10.1109/LSP.2016.2537321.

### Deep learning no-reference (DL-NR):
12. HyperNet
    S. Su et al., “Blindly assess image quality in the wild guided by a self-adaptive hyper network,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 3664–3673. doi: 10.1109/CVPR42600.2020.00372.

13. TReS
    S. A. Golestaneh, S. Dadsetan, and K. M. Kitani, “No-reference image quality assessment via transformers, relative ranking, and Self-consistency,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), Waikoloa, HI, USA, 2022, pp. 3989–3999. doi: 10.1109/WACV51458.2022.00404.

14. DACNN
    Z. Pan et al., “DACNN: Blind image quality assessment via a distortion-aware convolutional neural network,” IEEE Trans. Circuits Syst. Video Technol. vol. 32, no. 11, pp. 7518–7531, 2022, doi: 10.1109/TCSVT.2022.3188991.

15. VCRNet
    Z. Pan, F. Yuan, J. Lei, Y. Fang, X. Shao, and S. Kwong, “VCRNet: Visual compensation restoration network for no-reference image quality assessment,” IEEE Trans. Image Process. vol. 31, pp. 1613–1627, 2022, doi: 10.1109/TIP.2022.3144892.

16. SaTQA
    J. Shi, P. Gao, and J. Qin, “Transformer-based no-reference image quality assessment via supervised contrastive learning,” Proc. AAAI Conf. Artif. Intell. vol. 38, no. 5, pp. 4829–4837, 2024, doi: 10.1609/aaai.v38i5.28285.


## Usage Instructions

### For Training and Testing
Download the LWIR-IQA and/or the TIIQAD datasets.
Open the file " RunTrain.ipynb "

For the LWIR-IQA dataset, run
%run run.py --batch_size 40 --svpath '/path_where_to_save_results/' --epochs 20 --lr 2e-5 --gpunum 0 --datapath ./Path_to_LWIR-IQA_dataset/ --dataset lwir_iqa --version 1 --seed 2022

For the TIIQAD dataset, run
%run run.py --batch_size 40 --svpath '/path_where_to_save_results/' --epochs 20 --lr 2e-5 --gpunum 0 --datapath ./Path_to_TIIQAD_dataset/ --dataset TIIQAD --version 1 --seed 2022

### Testing on a single image

Open the file " RunTest_predict_one_image.ipynb" 
Set the path to the trained model. You may also use the provided pretrained models for LWIR-IQA and TIIQAD.
Set the path to the image to be predicted.
Run the notebook.


