# Person Re-Identification on Multi-Modal Data Using Swin Transformer Classifier

## Background

The increasing crime rates in Indonesia are alarming. Surveillance cameras are used to prevent crimes, but manual monitoring is still insufficient. A deep learning technology called Reidentification is used to reidentify individuals who have already been identified in a surveillance camera system. Field conditions can limit reidentification activities. Three spectrums of light can assist in this process: RGB, NI, and TI. Transformers, initially used in NLP, have shown potential in computer vision tasks. Swin Transformer, which incorporates a transformer hierarchy and shifted windows, is effective for complex vision tasks.

## Objective

The objective is to develop a deep learning model using the Swin Transformer classifier to reidentify individuals using RGB, NI, and TI modalities.

## Method

The RGBNT201 dataset was used, consisting of 201 identities, with 171 for training and 30 for testing. In model development, the Swin Transformer architecture was used with Windows Multi Head Attention (W-MSA) and Shifted Windows Multi Head Attention (SW-MSA). The model was tested using various Loss Functions (Circle, Triplet, Contrastive) and an Auto-Augment method was added to the model. The Auto-Augment policy was sourced from ImageNet.

### Testing Models

- **Model A:** Circle + Triplet + Contrast Loss
- **Model B:** Circle + Triplet Loss
- **Model C:** Circle Loss
- **Model D:** Circle + Contrast Loss
- **AutoAugment:** Applied to each loss variation
<img src="/result/modelreid-applied.png" alt="Alt text" width="800"/>

**Evaluation:**  
- Rank and mAP (mean Average Precision) performance metrics.
- Prediction visualization (top 10 images of predicted identities).
- Model visualization using Grad-CAM (Visual Explanations from Deep Networks via Gradient-Based Localization).

## Results

### Rank and mAP Results

The best performance was achieved using the Swin Transformer Model D.
<img src="/result/result-rankMAP.png" alt="Alt text" width="800"/>

### Prediction Visualization Results

- **Query Performance:**
got the best model with Model Swin AutoAugment B Circle + Triplet Loss, with RGB: 10/10, TI: 10/10, NI: 9/10
<img src="/result/SwinAugmentedB-Visualize.png" alt="Alt text" width="800"/>

### Grad-CAM Results

Grad-CAM analysis showed that the model focuses well on the human body for predictions. However, the Auto-Augment model provided better-focused results on relevant image areas.
<img src="/result/query_visualization_allmodels.png" alt="Alt text" width="800"/>

## Conclusion

1. This research successfully developed a deep learning model using the Swin Transformer classifier for person re-identification on RGB, NI, and TI modalities.
2. The best performance was achieved by Swin Transformer Model D with AutoAugment (using Circle Loss and Contrast Loss), achieving a mAP of 32.79% and rank@1 of 63%.

## References

1. A. Zheng, Z. Wang, Z. Chen, L. Li, and J. Tang, “Robust multi-modal person re-identification,” vol. 35, pp. 3529–3537, 2021. doi: 10.1609/aaai.v35i4.16497. Available: https://ojs.aaai.org/index.php/AAAI/article/view/16497. [Accessed on 2024]
2. Z. Liu et al., “Swin transformer: Hierarchical vision transformer using shifted windows,” In Proc. IEEE Int. Conf. Comput. Vis., 2021, pp. 9992–10002. doi: 10.1109/ICCV48922.2021.00988.
3. E. D. Cubuk, B. Zoph, D. Mane, V. Vasudevan, and Q. V. Le, “Autoaugment: Learning augmentation policies from data, 2019. arXiv: 1805.09501 [cs.CV].
