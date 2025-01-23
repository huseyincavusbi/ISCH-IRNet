# ISCH-IRNet: ISCHemia detection using InceptionResNet

## Overview

ISCH-IRNet is a deep learning model I designed for the detection of ischemia using brain CT scan images from the **[STROKE DATA SET (TEKNOFEST 2021)](https://acikveri.saglik.gov.tr/Home/DataSetDetail/1)** dataset. This project leverages the power of the InceptionResNetV2 architecture, pre-trained on ImageNet, and I fine-tuned it for the specific task of ischemia detection. The model distinguishes between ischemic and non-ischemic cases, **illustrating the potential of AI to improve the efficiency and accuracy of ischemia diagnosis, ultimately leading to better patient outcomes.**

## Table of Contents

-   [Overview](#overview)
-   [Table of Contents](#table-of-contents)
-   [Installation](#installation)
-   [Usage](#usage)
-   [Challenges and Solutions](#challenges-and-solutions)
    -   [Challenge 1: Class Imbalance](#challenge-1-class-imbalance)
    -   [Challenge 2: Overfitting](#challenge-2-overfitting)
    -   [Challenge 3: Hyperparameter Tuning](#challenge-3-hyperparameter-tuning)
    -   [Challenge 4: Model Optimization](#challenge-4-model-optimization)
-   [Results](#results)
-   [Contributing](#contributing)
-   [License](#license)

## Installation

To set up the environment for ISCH-IRNet, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/huseyincavusbi/ISCH-IRNet.git
    cd ISCH-IRNet
    ```
2. **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate 
    ```
3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare your dataset:** Organize your dataset into two folders: `Ischaemic` and `Non Ischaemic`. Place these folders inside a `Train2` directory.
   *   In the specified database link, the images under the title ‘Training Data Set No Stroke_chronic process_PNG’ were grouped in Train2/Non Ischaemic and the images under the title ‘Training Data Set_Ischemia’ were grouped in Train2/Ichaemic folders.
2. **Run the Jupyter Notebook:** Open the `isch-irnet.ipynb` notebook and run the cells sequentially. The notebook includes code for data preprocessing, model training, evaluation, and prediction.
3. **Predict on a single image:** Use the `predict_image` function in the last cell of the notebook to predict on a single image. You will be prompted to enter the path to the image.
   *   For input image prediction, the model was given image input from the ‘Competition Data Set (Session 1)’ file but any CT scan in jpg or png format can be used.

## Challenges and Solutions
Here are the challenges I faced during the testing phase of this latest model and how I addressed them.

### Challenge 1: Class Imbalance

**Problem:** The dataset exhibited a significant class imbalance, with fewer ischemic samples compared to non-ischemic samples. This imbalance can lead to a biased model that performs poorly on the minority class.

**Solution:**

1. **Oversampling:** I oversampled the minority class (ischemic) during training to balance the class distribution. This was achieved by randomly duplicating samples from the ischemic class until it matched the number of non-ischemic samples.
2. **Focal Loss:** I employed Focal Loss as the loss function. Focal Loss down-weights the loss assigned to well-classified examples, thus focusing more on the hard, misclassified examples and addressing the class imbalance.
3. **Class Weights:** I calculated class weights based on the inverse of class frequencies and incorporated them into the Focal Loss function. This further emphasized the importance of the minority class during training.

### Challenge 2: Overfitting

**Problem:** The model showed signs of overfitting, where it performed well on the training data but poorly on the validation and test data.

**Solution:**

1. **Data Augmentation:** I applied various data augmentation techniques to the training set, including random horizontal flips, rotations, and color jittering. This increased the diversity of the training data and reduced overfitting.
2. **Dropout:** I added dropout layers to the fully connected part of the network. Dropout randomly deactivates a fraction of neurons during training, preventing the model from relying too heavily on any specific features.
3. **Reduced Model Complexity:** I reduced the dropout rate to 0.3 in the fully connected layers to find a balance between regularization and model capacity.
4. **Early Stopping:** I implemented early stopping during training. The training process was stopped if the validation loss did not improve for a certain number of epochs, preventing the model from overfitting to the training data.

### Challenge 3: Hyperparameter Tuning

**Problem:** Finding the optimal hyperparameters for the model was crucial for achieving good performance.

**Solution:**

1. **Learning Rate Scheduling:** I used the `ReduceLROnPlateau` scheduler to dynamically adjust the learning rate during training. The learning rate was reduced when the validation loss plateaued, allowing for finer adjustments in the later stages of training.
2. **Experimentation:** I experimented with different values for learning rates, batch size, and the number of epochs to find the best combination.

### Challenge 4: Model Optimization

**Problem:** Optimizing the model architecture and training process was essential for achieving good performance

**Solution:**

1. **Transfer Learning:** I utilized transfer learning by using a pre-trained InceptionResNetV2 model. This allowed me to leverage the knowledge learned from a large dataset (ImageNet) and fine-tune it for my specific task.
2. **Freezing and Unfreezing Layers:** I initially froze the weights of the pre-trained layers and trained only the newly added fully connected layers. Later, I unfroze some of the later layers of the InceptionResNetV2 model to allow for fine-tuning on my dataset.
3. **Differential Learning Rates:** I used different learning rates for the pre-trained layers and the new fully connected layers. The pre-trained layers were trained with a smaller learning rate to preserve the learned features, while the new layers were trained with a larger learning rate to adapt to the new task.

## Results

After implementing the above solutions, the ISCH-IRNet model achieved the following results on the test set:

-   **Test Accuracy:** 0.8910
-   **Test Precision:** 0.8854
-   **Test Recall:** 0.8910
-   **Test F1-Score:** 0.8829
-   **Test AUC:** 0.7630

These results demonstrate the effectiveness of my approach in addressing the challenges of class imbalance, overfitting, and hyperparameter tuning.

## Contributing

Contributions to ISCH-IRNet are welcome! If you have any suggestions, bug fixes, or improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.