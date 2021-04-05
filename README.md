# Human-Activity-Recognition-using-RNNs-in-Smartphones

# I.  Introduction:

#### Project Overview:
The goal of this project is to build a `machine learning model` and a `signal processing pipeline` in offline mode capable of processing signals collected using smart phone inertial sensors and producing useful datasets will be used as inputs of a machine learning model capable of recognizing some of human daily activities (sitting, walking...). Three different types of recurrent neural networks have been used. The first type is a typical long-short-term Memory neural network (LSTM), the second type is a standard recurrent neural network (RNN) and the third type is a Grounded Recurrent Unit with gates (GRU). All networks are used for performance comparisons between them. 

#### Dataset - UCL Machine Learning Repository:
A set of experiments were carried out to obtain the HAR dataset. A group of 30 volunteers with ages ranging from 19 to 48 years were selected for this task. Each person was instructed to follow a protocol of activities while wearing a waist-mounted Samsung Galaxy S II smartphone. The six selected ADL were standing, sitting, laying down, walking, walking downstairs and upstairs. Each subject performed the protocol twice: on the first trial the smartphone was fixed on the left side of the belt and on the second it was placed by the user himself as preferred. There is also a separation of 5 seconds between each task where individuals are told to rest, this facilitated repeatability (every activity is at least tried twice) and ground trough generation through the visual interface. The tasks were performed in laboratory conditions but volunteers were asked to perform freely the sequence of activities for a more naturalistic dataset. You can download the dataset from [here](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones).

**Dataset Architecture:** 

Under The `UCI-HAR-Dataset` Directory we have:
 
 1. ` ./Inertial Signals/`: This directory includes the **Semi-processed features** of this version.


		- ` ./Inertial-Signals/train/`:  The train folder includes 11 files.

			- `total_acc_x_train.txt`: The acceleration signal from the smartphone accelerometer X axis 
			                           in standard gravity unit 'g'. Every row shows a 128-element vector.
						   The same description applies for the `total_acc_y_train.txt` and                    
						   `total_acc_z_train.txt` files for the Y and Z axis. 

			- `body_acc_x_train.txt`: The body acceleration signal obtained by subtracting the gravity 
			                           from the total acceleration. The same description applies for the
						   `body_acc_y_train.txt` and `body_acc_z_train.txt` files for the Y 
						   and Z axis.

			- `body_gyro_acc_x_train.txt`: The angular velocity vector measured by the gyroscope for each
			                               window sample. The units are radians/second. The same description 
						       applies for the `body_gyro_y_train.txt` and `body_gyro_z_train.txt`
						       files for the Y and Z axis. 


		- ` ./Inertial-Signals/test/*`: This folder includes necessary testing files of inertial signals 
		                                following the same analogy as in `./Inertial Signals/train/`.



2. `./Processed-Data/` : This directory includes the **fully processed features** which concerns the same six activities. 


		- `X_train.txt`: Train features, each line is composed 561-feature vector with time and 
		                 frequency domain variables.

		- `X_test.txt`: Test features, each line is composed 561-feature vector with time and 
		                frequency domain variables.
				
		- `features_info.txt`: Shows information about the variables used on the feature vector.

		- `features.txt`: includes list of all 561 features


- `y_train.txt`: train activity labels, Its range is from 1 to 6

- `y_test.txt`: test activity labels, Its range is from 1 to 6

- `subject_train.txt`: training subject identifiers, Its range is from 1 to 30

- `subject_test.txt`: testing subject identifiers, Its range is from 1 to 30

- `activity_labels.txt`:

- `README.md`:

# II. Software Requirements:
This project requires **Python 3.7** and the following Python libraries installed:
- [Python 3.7](https://www.python.org/downloads/) 
- [NumPy](http://www.numpy.org/)  , [SciPy](https://www.scipy.org/) , [Pandas](https://pandas.pydata.org/) , [matplotlib](http://matplotlib.org/)
- [Tensorflow](https://www.tensorflow.org), [scikit-learn](http://scikit-learn.org/stable/)
### Installation
1. Clone this repository.
2. Install ```Python 3.7``` or older versions
3. Ensure all libraries are installed using ```pip install <library> --user```
# III. Project Architecture:

This repository [Human-Activity-Recognition-Using-Smartphones](https://github.com/anas337/Human-Activity-Recognition-Using-Smartphones) includes 1 main directory and 3 general files:

### III-1. Directories:

	- `\Results\Testing`: This folder includes evaluation results of predicted models in test dataset
	
	- `\Results\Training`: Contains images of training the models
	
	- `\Results\Visualization`: Contains images of the visualization of the dataset
		
		
### III-2. Files:

	- `TrainTheModel.py`: This file contains the signal processing pipeline and the train of the models	
	
	- `TestTheModel.py`:  This file contains the evaluation of the models in the dataset           
	
	- `Visualiazation.py`:  This file contains visualizations and detailed analysis of each step performed.  
	       
# IV. Results:



