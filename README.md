# Human-Activity-Recognition-using-RNNs-in-Smartphones

# I.  Introduction:

#### Project Overview:
The goal of this project is to build a `machine learning model` and a `signal processing pipeline` in offline mode capable of processing signals collected using smart phone inertial sensors and producing useful datasets will be used as inputs of a machine learning model capable of recognizing some of human daily activities (sitting, walking...). Three different types of recurrent neural networks have been used. The first type is a typical long-short-term Memory neural network (LSTM), the second type is a standard recurrent neural network (RNN) and the third type is a Grounded Recurrent Unit with gates (GRU). All networks are used for performance comparisons between them. 

#### Dataset:
A set of experiments were carried out to obtain the HAR dataset. A group of 30 volunteers with ages ranging from 19 to 48 years were selected for this task. Each person was instructed to follow a protocol of activities while wearing a waist-mounted Samsung Galaxy S II smartphone. The six selected ADL were standing, sitting, laying down, walking, walking downstairs and upstairs. Each subject performed the protocol twice: on the first trial the smartphone was fixed on the left side of the belt and on the second it was placed by the user himself as preferred. There is also a separation of 5 seconds between each task where individuals are told to rest, this facilitated repeatability (every activity is at least tried twice) and ground trough generation through the visual interface. The tasks were performed in laboratory conditions but volunteers were asked to perform freely the sequence of activities for a more naturalistic dataset. You can download the dataset from [here](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones).

# II. Software Requirements:
This project requires **Python 3.7** and the following Python libraries installed:
- [Python 3.7](https://www.python.org/downloads/) 
- [NumPy](http://www.numpy.org/)  , [SciPy](https://www.scipy.org/) , [Pandas](https://pandas.pydata.org/) , [matplotlib](http://matplotlib.org/)
- [Tensorflow](https://www.tensorflow.org), [scikit-learn](http://scikit-learn.org/stable/)
