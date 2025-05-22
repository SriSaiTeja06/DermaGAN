
# DermaGAN

DermaGAN is a deep learning project that uses Generative Adversarial Networks (GANs) to generate synthetic dermatology images and enhance the performance of skin lesion classification models. Designed to tackle the challenge of limited and imbalanced dermatological datasets, DermaGAN contributes a novel dual-model pipeline combining GAN-based data augmentation with a custom CNN classifier. The model is trained and classifies images into four different classes: Melanoma, Basal Cell Carcinoma, Actinic Keratosis, and Benign Keratosis.

---

## 🧠 Motivation

Accurate skin lesion diagnosis through AI often suffers due to insufficient or imbalanced image datasets. To bridge this gap, DermaGAN introduces synthetic data generation using GANs to augment underrepresented classes and improve classifier robustness. The goal is to enable earlier detection of skin conditions, particularly skin cancer, through better-trained diagnostic models.

---

## 🧰 Features

- Custom GAN model trained on curated dermatology datasets
- High-quality synthetic image generation to balance class distribution
- Custom CNN model for skin lesion classification
- Evaluation metrics including accuracy, loss, and confusion matrix
- Modular and extendable codebase using TensorFlow and Keras

---

## 🏗️ Architecture

**1. GAN Module:**
- Generator and Discriminator trained adversarially to produce realistic synthetic skin lesion images.
- Refined training pipeline to improve image quality over time.

**2. CNN Classifier:**
- Custom convolutional neural network trained on a mix of real and synthetic images.
- Predicts lesion types across multiple classes with enhanced accuracy.

**Workflow:**

```
Raw Dataset → GAN → Synthetic Images → Augmented Dataset → CNN → Classification Results
```

---

## 🖼️ Results

- Achieved improved classification accuracy by augmenting real data with high-fidelity GAN-generated images.
- The combined pipeline showed a **significant increase in diagnostic performance** over baseline CNN models trained on non-augmented data.

---

## 📊 Technologies Used

- Python 3.x
- TensorFlow & Keras
- NumPy, OpenCV, Matplotlib
- Jupyter Notebook / Colab
- GANs for synthetic data generation
- CNNs for image classification

---

## 🚀 How to Run

1. **Clone the repository**  
   ```
   git clone https://github.com/SriSaiTeja06/DermaGAN
   cd DermaGAN
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks in order**
   - Start with `FinalGANImplementation.ipynb` to generate synthetic images.
   - Train CNNs with and without GAN images using:
     - `Model_With_GAN_Images.ipynb`
     - `Model_without_GAN_images.ipynb`

4. **Use pre-trained model**
   - You can load `predictin_with_GAN_model.h5` in `prediction.ipynb` for testing on new images.

5. **Explore additional architectures**
   - Try alternative models like ResNet50 or MobileNetV2 in their respective notebooks.

---

## 📦 Data

Due to the large size of the image dataset, it is not included in this repository. You can download the data from the following Google Drive link:

[Google Drive Data Link](https://drive.google.com/drive/u/0/folders/1sEhUQS-lSO3g_Czgb6GrmWDg8SKbd5LF)

---

## � Future Work

- Integrate pretrained models (e.g., ResNet, EfficientNet) for comparison with custom CNN
- Apply advanced GAN architectures (StyleGAN, Pix2Pix)
- Evaluate generalization on larger and more diverse dermatology datasets
- Deploy as a web application for real-time lesion classification support

---

## ✍️ Authors

- Sri Sai Teja M S  
- Ullas P  
- Vikas K R  

---

## 📜 License

This project is developed as part of an academic requirement and is intended for educational and research purposes only.

