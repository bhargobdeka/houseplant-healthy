# Houseplant-Healthy

<!-- ![Houseplant Healthy Logo](houseplant.jpg) -->
<p align="center">
  <img src=houseplant.jpg width="700px" height="500px" >
</p>

Houseplant Health Check is a project that helps you keep your houseplants healthy and thriving. It employs deep learning techniques to analyze plant images, serves the model using TensorFlow Serving, provides a FastAPI backend for predictions, and has a Node.js frontend for a user-friendly interface.

## Table of Contents

- [Data Collection](#data-collection)
- [Deep Learning Model](#deep-learning-model)
- [FastAPI](#fastapi)
- [TensorFlow Serving](#tensorflow-serving)
- [Frontend using Node.js](#frontend-using-nodejs)
- [Future Work](#future-work)

## Data Collection

The dataset used in this project was taken from [Kaggle](https://www.kaggle.com/datasets/russellchan/healthy-and-wilted-houseplant-images/data). Also, other images from the net were downloaded and dodgy images were removed. The dataset includes two classes: **Healthy** and **Wilted**.

## Deep Learning Model

In this project, we employed a Convolutional Neural Network (CNN) to tackle the task of [insert the task or problem the model is solving, e.g., image classification, object detection, etc.]. The following is an overview of the CNN architecture used:

- **Input Shape**: The input images were preprocessed to have a shape of (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), where BATCH_SIZE is set to 32, IMAGE_SIZE is 256 pixels, and CHANNELS is 3 for RGB color images.

- **Layers**:

  - The model begins with data preprocessing, including resizing and rescaling.
  - Data augmentation techniques were applied to increase the diversity of the training data.
  - Several convolutional layers were used, each with a 3x3 kernel size and ReLU activation.
  - After each convolutional layer, there is a max-pooling layer with a 2x2 pool size to downsample the feature maps.
  - The process of convolution and max-pooling was repeated several times to capture hierarchical features.
  - The final layers include flattening the feature maps, followed by fully connected layers with ReLU activation.
  - The output layer has a softmax activation with 2 units, corresponding to the 2 classes in our problem.

- **Model Building**: The model architecture was built using TensorFlow's Sequential API, and the input shape was specified as input_shape.

This CNN architecture was designed to effectively learn and represent features from the input images, making it suitable for the specific problem at hand. Feel free to explore the model's details and fine-tune it as needed for your own use cases.

## Fast API

The FastAPI backend serves as the core of our houseplant health assessment project. It provides RESTful endpoints for image uploads and health predictions. To run the FastAPI server, follow the code in `main.py` in the `api` directory. You can run the `main.py` and then use the local host link to upload images or use **postman**

## TensorFlow Serving

I used TensorFlow Serving to serve our trained deep learning model with different versions based on our preference. Automatically, the current code would serve the latest model which is '3' provided in `model_saved` folder for this code. But, we can control this within the `main_TFserving.py`.

## Frontend using Node.js

My project includes an user-friendly frontend built with **Node.js**. It allows me to use a local host where I can upload plant images and receive health assessments, such as the class and the confidence level. The frontend code and setup instructions can be found in the `frontend` directory.

<p align="center">
  <img src=houseplantWeb.png width="700px" height="500px" >
</p>

## Future Work

In the future, I plan to expand the project by:

- Integrating with Google Cloud Platform (GCP) for scalability and additional services.
- Developing a mobile app using **React Native** for on-the-go plant health monitoring.
- Improving the predictive accuracy for the CNN model.
- Collecting more real data from community for training and testing.

Stay tuned for more updates!

---

**Disclaimer**: This project is for educational purposes and should not replace professional plant care advice. Always consult a plant expert for accurate plant health assessments and care recommendations.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

I would like to extend my heartfelt thanks to [codebasics](https://www.youtube.com/watch?v=dGtDTjYs3xc&list=PLeo1K3hjS3ut2o1ay5Dqh-r1kq6ZU8W0M&index=17&t=24s) on YouTube for their invaluable tutorials and inspiration. Their end-to-end deep learning project on potato-disease provided the initial spark and guidance for my own project on houseplants. I am grateful for their educational content and the impact it had on this project.

<!-- ## Installation

1. Clone the repository:

   	```bash
   	git clone https://github.com/bhargobdeka/houseplant-healthy.git

   	```

2. Change to the directory:

	```bash
	cd houseplant-healthy

	```
## Model

![Model 1](model-1.png)


## Results
![Model Predict 1](model-predict-1.png) -->
