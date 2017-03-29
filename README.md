# CarND_Traffic_Sign_Classifier_P2

## Description

***This my project result of Udacity self-driving car nanodegree (CarND) 2nd project. It required implementing a deep neural network (DNN) traffic sign classifier based on TensorFlow. I used LeNet-5 as base architecture and traffic sign image dataset is obtained from German Traffic Sign Dataset***

* Udacity self-driving car nanodegree (CarND) :

  https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013

* LeNet-5 is convolutional network designed for handwritten and machine-printed character recognition :

  http://yann.lecun.com/exdb/lenet/

* German Traffic Sign Dataset

  http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

[image1]: ./result_image/1.png "4 by 4 image matrix from training images"
![alt text][image1]


## Dataset of this project

* The size of training set is 34799 (data is stored in train.p)
* The size of validation set is 4410 (data is stored in valid.p)
* The size of test set is 12630 (data is stored in test.p)
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

## DNN model of this project 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6 |
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 				|
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 				|
| Flatten	      	| outputs 400 				|
| Fully connected		| outputs 120        									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| outputs 84        									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| outputs 43        									|


## Performance 
* training set accuracy of 99.7%
* validation set accuracy of 96.7%
* test set accuracy of 94.7%

