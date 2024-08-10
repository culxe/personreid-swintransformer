# Person Re-Identification on Multi-Modal Data Using Swin Transformer Classifier
published on Institut Teknologi Sepuluh Nopember [repository](https://repository.its.ac.id/109264/)
## Background

The increasing crime rates in Indonesia are alarming. Surveillance cameras are used to prevent crimes, but manual monitoring is still insufficient. A deep learning technology called Reidentification is used to reidentify individuals who have already been identified in a surveillance camera system. Field conditions can limit reidentification activities. Three spectrums of light can assist in this process: RGB, NI, and TI. Transformers, initially used in NLP, have shown potential in computer vision tasks. Swin Transformer, which incorporates a transformer hierarchy and shifted windows, is effective for complex vision tasks.

## Objective

The objective is to develop a deep learning model using the Swin Transformer classifier to reidentify individuals using RGB, NI, and TI modalities.

## Method

Using [RGBNT201 dataset](https://drive.google.com/drive/folders/1EscBadX-wMAT56_It5lXY-S3-b5nK1wH?usp=sharing), consisting of 201 identities, with 171 identities for training and 30 identities for testing.

### Data implementation on Swin Transformer
refer from [Swin transformer: Hierarchical vision transformer using shifted windows](https://arxiv.org/pdf/2103.14030), this is impelementation using multi-modal dataset on Swin Transformer.

#### Patch Embedding
<img src="/result/bab3_patchembedding.png" alt="Alt text" width="800"/>

#### Windows Multi Head Attention
<img src="/result/bab3_wmsa.png" alt="Alt text" width="700"/>

#### Shifted Windows Multi Head Attention
<img src="/result/bab3_swmsa.png" alt="Alt text" width="800"/>


### Training and Testing Models

| Model | Circle | Contrast | Triplet |
|-------|--------|----------|---------|
|   A |   ✔    |    ✘     |    ✔    |
|   B   |   ✔    |    ✔     |    ✔    |
|   C  |   ✔    |    ✘     |    ✘    |
|   D  |   ✔    |    ✔     |    ✘    |

*AutoAugment is applied to each loss variation.*

## Results

### Rank and mAP Results

The best performance was achieved using the Swin Transformer Model D.
| Model            | mAP  | rank@1 |
|------------------|------|--------|
| Loss Var. A (circle + triplet)   | 24.55| 48.33  |
| Loss Var. B (circle + contrast + triplet)   | 29.66| 56     |
| AutoAugment C (circle + autoaugment)    | 31.26| 60.33  |
| Loss Var. C (circle)   | 31.55| 57.33  |
| Loss Var. D (cirle + contrast)   | 31.55| 57.33  |
| AutoAugment A (circle + triplet + autoaugment)    | 31.79| 57.66  |
| AutoAugment B (circle + contrast + triplet + autoaugment)    | 32.55| 60     |
| **AutoAugment D (cirle + contrast + autoaugment)**|**32.79**|**63**|

### Prediction Visualization Results

- **Query Performance:**
got the best model with Model Swin AutoAugment B Circle + Triplet Loss, with RGB: 10/10, TI: 10/10, NI: 9/10
<img src="/result/SwinAugmentedB-Visualize.png" alt="Alt text" width="800"/>
with detailed prediction
<img src="/result/SwinAutoaugment-predictdetail.png" alt="Alt text" width="800"/>

### Grad-CAM Results

Grad-CAM analysis showed that the model focuses well on the human body for predictions. However, the Auto-Augment model provided better-focused results on relevant image areas.
<img src="/result/query_visualization_allmodels.png" alt="Alt text" width="800"/>

## References

1. [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch/tree/master)
2. [IEEE](https://github.com/ziwang1121/IEEE) for multi-modal dataset
3. A. Zheng, Z. Wang, Z. Chen, L. Li, and J. Tang, “Robust multi-modal person re-identification,” vol. 35, pp. 3529–3537, 2021. doi: 10.1609/aaai.v35i4.16497. Available: https://ojs.aaai.org/index.php/AAAI/article/view/16497. [Accessed on 2024]
4. Z. Liu et al., “Swin transformer: Hierarchical vision transformer using shifted windows,” In Proc. IEEE Int. Conf. Comput. Vis., 2021, pp. 9992–10002. doi: 10.1109/ICCV48922.2021.00988.
5. E. D. Cubuk, B. Zoph, D. Mane, V. Vasudevan, and Q. V. Le, “Autoaugment: Learning augmentation policies from data, 2019. arXiv: 1805.09501 [cs.CV].
6. R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-cam: Visual explanations from deep networks via gradient-based localization,” International Journal of Computer Vision, vol. 128, no. 2, pp. 336–359, Oct. 2019, issn: 1573-1405. doi: 10.1007/s11263-019-01228-7. [Online]. Available: http://dx.doi.org/10. 1007/s11263-019-01228-7 (visited on 2024).


## Disclaimer

We apologize for any errors or inaccuracies in this repository. Please feel free to raise an issue or contact us if you notice something that needs correction or improvement. Your feedback is highly appreciated!

