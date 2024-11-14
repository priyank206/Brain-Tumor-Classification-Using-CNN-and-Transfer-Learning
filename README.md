
# Brain Tumor Classification Using CNN and Transfer Learning

## Overview

This repository contains code, models, and resources for classifying brain tumors using Convolutional Neural Networks (CNNs) and transfer learning techniques. The goal of this project is to build and compare models that can identify brain tumor types from MRI images with high accuracy.

## Project Structure

- **Notebooks**:
  - `deep_learning_da.ipynb`: Contains code for data preprocessing, exploratory data analysis (EDA), and model training for brain tumor classification.

- **Models**:
  - `Model 1`: A CNN model built from scratch for brain tumor classification.
  - `Model 2`: A transfer learning model using a pre-trained CNN architecture, fine-tuned for improved accuracy.

- **Data**:
  - MRI images required for training and testing the models. Place the dataset in the project directory (see dataset information below).

- **Scripts** (optional):
  - Additional scripts for training, evaluation, and prediction tasks can be added here as needed.

## Getting Started

1. **Clone the Repository**

    ```bash
    git clone https://github.com/priyank206/Brain-Tumor-Classification-Using-CNN-and-Transfer-Learning.git
    cd Brain-Tumor-Classification-Using-CNN-and-Transfer-Learning
    ```

2. **Install Dependencies**
   Make sure Python is installed. The project requires libraries such as `tensorflow`, `numpy`, `matplotlib`, and `scikit-learn`. Install all dependencies with:

    ```bash
    pip install -r requirements.txt
    ```

3. **Open the Notebook**
   Start the Jupyter Notebook to explore and run the project:

    ```bash
    jupyter notebook deep_learning_da.ipynb
    ```

## Dataset

The dataset includes MRI brain images labeled with tumor presence. Download a relevant dataset, such as the [Kaggle Brain Tumor Dataset](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection), and place it in the project directory.

## Workflow

The main workflow includes the following steps:

1. **Data Loading**: Load and preprocess MRI images for use in the models.
2. **Exploratory Data Analysis (EDA)**: Visualize the data and understand the class distribution.
3. **Model Training**: Train both the custom CNN and transfer learning model to classify brain tumors.
4. **Evaluation**: Assess model performance on test data using metrics like accuracy and loss.
5. **Results**: Compare the two models and analyze performance metrics and training curves.

## Results

The project compares the performance of two models:
- **Custom CNN Model**: A CNN architecture built from scratch.
- **Transfer Learning Model**: A model using a pre-trained network, adapted for brain tumor classification.

Each model is evaluated based on metrics such as accuracy, precision, recall, and F1-score.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Author

- **Priyank** - [GitHub Profile](https://github.com/priyank206)
