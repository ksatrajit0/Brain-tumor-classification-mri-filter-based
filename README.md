
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

To ensure consistent results, please use the following commands:

|Package Name | Version|
| --- | ---|
| pandas | 1.5.3|
|scikit-learn | 1.0.2|
|seaborn | 0.11.2|
|matplotlib | 3.5.1|
|opencv-python | 4.5.3.20210927|
| numpy | 1.21.2 |
|tensorflow | 2.8.0|
|pillow | 9.1.0|

### Program files:

The implementation files of this project can be found here:
-
-
-

## Credit(s) and Acknowledgement:

Collaborator: **[Utathya Aich](https://in.linkedin.com/in/utathyaaich)**
Supervisor: **[Dr. Pawan Kumar Singh](https://jadavpuruniversity.in/faculty-profile/pawan-kumar-singh/)**

### Note:
This project's corresponding research paper is currently under review for publication, and the link will be uploaded here upon acceptance.
> Thank you very much!

