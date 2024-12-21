# Oral Disease Detection Using efficientnet_b0

This project focuses on building a machine learning solution for classifying oral diseases into two categories: **Class A** and **Class B**. The dataset is divided into training and testing directories, and the models are fine-tuned to achieve optimal performance.

## Tech Stack
- **Programming Language**: Python
- **Frameworks/Libraries**:
  - Pytorch
  - NumPy
  - Matplotlib
  - Scikit-learn
- **Environment**: Jupyter Notebook (`.ipynb` file)

## Dataset
The dataset is organized as follows:
- `TRAIN/`: Contains subdirectories for **Caries** and **Gingivitis** images.
- `TEST/`: Contains subdirectories for **Caries** and **Gingivitis** images.

## Pre-Trained Model
This project utilizes `efficientnet_b0` from torchvision library to leverage transfer learning. The top layers of the model have been modified and fine-tuned on the oral disease dataset.

### Key Steps in Fine-Tuning:
1. **Base Model Loading**:
   - The model is loaded with weights pre-trained on ImageNet.
   - The base model is frozen initially to retain its learned features.
   
2. **Custom Top Layers**:
   - Fully connected layers specific to the dataset are added to the model.
   - A softmax layer is included for binary classification.

3. **Fine-Tuning**:
   - The base model is unfrozen.
   - Selected layers are fine-tuned on the oral disease dataset using a reduced learning rate.

## Model Training
1. **Data Preprocessing**:
   - Images are resized and normalized.
   - Data augmentation techniques such as rotation, flipping, and scaling are applied to improve generalization.

2. **Training Strategy**:
   - The model is trained in two stages:
     - **Feature Extraction**: Train only the custom top layers with the base model frozen.
     - **Fine-Tuning**: Unfreeze the base model and train selected layers with a smaller learning rate.

3. **Evaluation**:
   - Accuracy and loss are monitored using the test dataset.
   - Confusion matrix and classification reports are generated to evaluate model performance.

## How to Run
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Oral_diseases.ipynb
   ```
3. Follow the instructions in the notebook to train and evaluate the model.

## Results
- The model achieves high accuracy on both the training and test datasets.
  
| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **Caries**    | 0.96      | 0.99   | 0.97     | 204     |
| **Gingivitis**| 0.98      | 0.96   | 0.97     | 204     |
| **Accuracy**  |           |        | 0.97     | 408     |
| **Macro Avg** | 0.97      | 0.97   | 0.97     | 408     |
| **Weighted Avg** | 0.97   | 0.97   | 0.97     | 408     |



- Visualization of loss and accuracy trends is provided in the notebook.

## Future Work
- Expand the dataset with more diverse samples to improve robustness.
- Experiment with other state-of-the-art pre-trained models.
- Deploy the trained model for real-time inference using a web application.

## Contributions
Contributions are welcome! Please fork the repository and create a pull request for any improvements or suggestions.




