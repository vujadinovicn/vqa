# Model Card: VQA Models

## Overview

This document provides detailed descriptions of the different models developed for Visual Question Answering (VQA) tasks. Each model uses a distinct method for combining image and text embeddings, trained with different hyperparameters.

---

### 1. VQA-Stack I

- **Fusion Method**: **Stacking** - Image and text embeddings are concatenated into a single vector.
- **Learning Rate**: 1 × 10<sup>-5</sup>
- **Batch Size**: 3072
- **Training Epochs**: 1000
- **Best Epoch**: 841
- **Training Accuracy**: 60.1%
- **Validation Accuracy**: 51%
- **Download**: [VQA-Stack I Model](https://drive.google.com/file/d/1_1nM1gtF0W4nUXSEiUNQ8UhD6XkXbfZ6/view?usp=sharing)

---

### 2. VQA-Stack II

- **Fusion Method**: **Stacking** - Same as VQA-Stack I, but with a larger batch size.
- **Learning Rate**: 1 × 10<sup>-5</sup>
- **Batch Size**: 16384
- **Training Epochs**: 2000
- **Best Epoch**: 1965
- **Training Accuracy**: 59.4%
- **Validation Accuracy**: 50.7%
- **Download**: [VQA-Stack II Model](https://drive.google.com/file/d/1mbMHEQBOqcMQXmQ2DL3u9b-zLxBedptT/view?usp=sharing)

---

### 3. VQA-Mul

- **Fusion Method**: **Multiplication** - Point-wise multiplication of the image and text embedding vectors.
- **Learning Rate**: 1 × 10<sup>-5</sup>
- **Batch Size**: 4096
- **Training Epochs**: 2000
- **Best Epoch**: 1616
- **Training Accuracy**: 49.8%
- **Validation Accuracy**: 48.2%
- **Download**: [VQA-Mul Model](https://drive.google.com/file/d/1mX-Wz_PEsb7XggT2hBdLeTTDBAq2lj_h/view?usp=sharing)

---

### 4. VQA-Attention I

- **Fusion Method**: **Attention Mechanism** - Utilizing attention to focus on key parts of the input embeddings.
- **Learning Rate**: 1 × 10<sup>-5</sup>
- **Batch Size**: 3072
- **Training Epochs**: 1000
- **Best Epoch**: 634
- **Training Accuracy**: 48.3%
- **Validation Accuracy**: 44.6%
- **Download**: [VQA-Attention I Model](https://drive.google.com/file/d/1Gx_jeAT3PMKgnR9NQS20b1EajwBpGPN0/view?usp=sharing)

---

### 5. VQA-Attention II

- **Fusion Method**: **Attention Mechanism** - Same as VQA-Attention I, but with a larger learing rate parameter.
- **Learning Rate**: 3 × 10<sup>-4</sup>
- **Batch Size**: 3072
- **Training Epochs**: 1000
- **Best Epoch**: 51
- **Training Accuracy**: 43.2%
- **Validation Accuracy**: 41.4%
- **Download**: [VQA-Attention II Model](https://drive.google.com/file/d/1X3gqt3bfQDYI2Y0alNYb1klEOIe0jNNq/view?usp=sharing)
