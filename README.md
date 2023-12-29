# Tomato Ripeness Detection Project
## Overview

This GitHub repository contains the code for the Tomato Ripeness Detection project, which aims to revolutionize tomato harvesting using advanced robotics and computer vision techniques. The project includes two main components: a machine learning model for detecting ripe tomatoes and a real-time demo for predicting ripe tomatoes using computer vision.

## Files

1. **`tensor.py`**: This file contains the code for building a machine learning model using TensorFlow, Keras, and Scikit-learn. The model is trained to detect ripe tomatoes and is saved as `tomato_model.h5`.

2. **`demo1.py`**: The real-time prediction demo file. It utilizes computer vision techniques to detect and predict ripe tomatoes using the pre-trained machine learning model (`tomato_model.h5`).

## Technologies Used

- TensorFlow
- Keras
- Scikit-learn
- Computer Vision
- Python

## Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/tomato-harvesting-robot.git
   cd tomato-harvesting-robot
   ```

2. Run the `tensor.py` script to train the machine learning model:

   ```bash
   python tensor.py
   ```

   This will generate the `tomato_model.h5` file.

3. Run the real-time prediction demo using `demo1.py`:

   ```bash
   python demo1.py
   ```

   Ensure that the required libraries are installed by using:

   ```bash
   pip install -r requirements.txt
   ```

   The demo will utilize computer vision techniques to predict ripe tomatoes in real-time.

## Additional Notes

- Make sure to customize the paths and parameters in the code according to your setup.
- Feel free to explore and modify the code for further enhancements.

## Contributions

Contributions to improve and expand the functionality of the Tomato Ripeness Detection project are welcome. Fork the repository, make your changes, and submit a pull request.

Happy coding! 🍅🤖
