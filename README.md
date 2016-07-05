# Car Logo Recognition using Neural Network
This project implemented a 3-layer neural network to recongnize different car logos. We use scikit-image is used in image processing and feature extraction, tkinter for user interface, and numpy / scipy for implementing neural network and other mathmatical operations.

- ./Logos - Original Images of Car Logo
- ./TrainingSet - Processed Images of Car Logo
- ./Test - Testing Images of Car Logo
- imageProcess.py - Logo extraction, cropping, and converting to grayscale
- generateFeature.py - Extract HOG feature from images
- ThreeLayerNN.py - A three-layer neural network for regonitizing logos
- NN_predict.py - Use trained parameters to predict new image
- UserInterface.py - An user interface to select input image and display result