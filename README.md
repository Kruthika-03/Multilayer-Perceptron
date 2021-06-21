# Multilayer-Perceptron
Train MLP on MNIST dataset

A perceptron includes an input, multiplied with weights and a bias is added for activation. A layer of these perceptrons, forming a dense connection can be referred to as a Neural network. A multi-layer perceptron is also called feed forward neural network (FFNN).

A basic MLP consists of three layers, namely, the input layer, the hidden layer and the output layer. Input is fed at the input layer and output is fetched at the output layer. We can have as many no. of hidden layers as we want. In an MLP every node/neuron of current layer is connected to every other neuron in the next layer. Each of the neurons have weights and biases. These parameters are all trainable i.e. in MLP all the parameters are trainable.
We pass the input to model and multiply with weights and add bias at every layer and find the calculated output of the model. In MLP we use the Back-Propagation Algorithm. The loss is calculated and we back propagate the loss. According to which the weights are updated/altered.

## Algorithm
1. Import the module future from python library
2. Import NumPy, TensorFlow, Keras and other required libraries
3. Import matplotlib and sklearn
4. Initialize the batch size, number of classes and epochs
5. Split the data into training and testing dataset
6. Check the shape and datatype
7. Using to_categorical in Keras to get a binary class matrix from vector class
8. Define the model architecture. Here we use sequential type of model
9. Compile the model using optimizer RMSprop
10. Fit the model
11. Evaluate the model
12. Plot the loss and accuracy
13. Print the Confusion Matrix
14. Print the model summary

## Dataset Description
I have considered the MNIST Data Set (Modified Institute of Standards and Technology database) of handwritten digits.
* It contains 60,000 training images set and 10,000 testing image set.
* It consists of 10 classes i.e. 0-9 handwritten digits.
* The resolution of each of these images is 28x28 =784 pixels.
* The images are in Grey-scale meaning the value of each pixel ranges from 0-255.

![image](https://user-images.githubusercontent.com/58825386/122784620-42c26a00-d2d0-11eb-99c4-bd707d95d90d.png)

The data files named train.csv and test.csv contain gray-scale images of hand-written digits, from zero all the way through nine. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.
The training data set, has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.
