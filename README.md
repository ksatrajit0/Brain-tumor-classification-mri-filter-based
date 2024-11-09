
# Efficient Brain Tumor Classification using Filter-Based Deep Feature Selection Methodology

## Description:

> Methodology

- Implementation of a novel two-stage framework for classifying 3 kinds of brain tumors and healthy patients from structural MRI scans.
- In the first stage, a pre-trained Convolutional Neural Network has been used to extract relevant features from pre-processed images through transfer learning, considerably reducing training time and extensive hardware requirements.
- In the second stage, a filter-based deep feature selection methodology using Mutual Information has been applied to minimize the extracted, high-dimensional feature maps.
- Finally, a Support Vector Machine with a polynomial kerenl ruse has been used for multi-class classification.

> Datasets used:
- **[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)**
- **[Crystal Clean: Brain Tumors MRI Dataset](https://www.kaggle.com/datasets/mohammadhossein77/brain-tumors-dataset)**
- **[Figshare: Brain Tumors MRI Dataset](https://www.kaggle.com/datasets/denizkavi1/brain-tumor)**

> Preprocessing of MRI scans:

![Progressive stages of structural MRI scan enhancement in Data Pre-processing Phase](https://github.com/user-attachments/assets/fc664a29-a361-4762-93aa-ce603965e38e)

> Methodology Used:

![Workflow of the proposed filter-based deep feature selection framework for Brain tumor Classification.](https://github.com/user-attachments/assets/a6cbb1c2-f122-4a2a-848b-ff89f4f23b14)

### Importance of Project:
- The promising results achieved underscore the potential of our lightweight framework’s robust nature and generalization capabilities.
- Suitable for deployment in real-time environments with limited technological resources.
- Assist medical professionals in making precise diagnoses and, ultimately enhance patient outcomes.

### Basic Requirements:

To ensure consistent results, please use the following libraries:

| Package Name     | Version       | Version Badge                                                      |
| ---------------- | ------------- | ------------------------------------------------------------------ |
| pandas           | 1.5.3         | ![pandas version](https://img.shields.io/badge/pandas-1.5.3-blue?style=flat-square)   |
| scikit-learn     | 1.0.2         | ![scikit-learn version](https://img.shields.io/badge/scikit--learn-1.0.2-yellowgreen?style=flat-square) |
| seaborn          | 0.11.2        | ![seaborn version](https://img.shields.io/badge/seaborn-0.11.2-green?style=flat-square) |
| matplotlib       | 3.5.1         | ![matplotlib version](https://img.shields.io/badge/matplotlib-3.5.1-blueviolet?style=flat-square) |
| opencv-python    | 4.5.3.20210927| ![opencv-python version](https://img.shields.io/badge/opencv--python-4.5.3.20210927-brightgreen?style=flat-square) |
| numpy            | 1.21.2        | ![numpy version](https://img.shields.io/badge/numpy-1.21.2-lightblue?style=flat-square)   |
| tensorflow       | 2.8.0         | ![tensorflow version](https://img.shields.io/badge/tensorflow-2.8.0-orange?style=flat-square) |
| pillow           | 9.1.0         | ![pillow version](https://img.shields.io/badge/pillow-9.1.0-lightyellow?style=flat-square)   |

### Program files:

The implementation files of this project can be found here:
- [Brain Tumor MRI Jupyter Notebook](https://github.com/ksatrajit0/Brain-tumor-classification-mri-filter-based/blob/main/brain_tumorMRI_GitHubv1.ipynb)
- [Crystal Clean Dataset Jupyter Notebook](https://github.com/ksatrajit0/Brain-tumor-classification-mri-filter-based/blob/main/crystal_dataset_github.ipynb)
- [Figshare Dataset Jupyter Notebook](https://github.com/ksatrajit0/Brain-tumor-classification-mri-filter-based/blob/main/figshare_dataset_github.ipynb)

## Credit(s) and Acknowledgement:

Collaborator: **[Utathya Aich](https://in.linkedin.com/in/utathyaaich)**
Supervisor: **[Dr. Pawan Kumar Singh](https://scholar.google.com/citations?user=LctgJHoAAAAJ&hl=en&oi=ao)**

### Paper:
> It'd be great if you could cite our manuscript if this code has been helpful to you:
  *Kar, S., Aich, U. & Singh, P.K. Efficient Brain Tumor Classification Using Filter-Based Deep Feature Selection Methodology. SN COMPUT. SCI. 5, 1033 (2024). https://doi.org/10.1007/s42979-024-03392-1*

> Thank you very much!

