# Traffic Sign Classifier

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[image9]: ./result_image/1.png "4 by 4 image matrix from training images"
[image10]: ./result_image/2.png "Normalized images"
[image11]: ./result_image/trail1.png "training result trial1"
[image12]: ./result_image/trail2.png "training result trial2"
[image13]: ./result_image/trail3.png "training result trial3"
[image14]: ./result_image/trail4.png "training result trial4"
[image15]: ./result_image/new_test_image.png "new test image"
[image16]: ./result_image/top5.png "top 5"

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. 
It's a 4 by 4 image matrix with image randomly selected 16 images from the training image data set.

![alt text][image9]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 4th code cell of the IPython notebook.

I normalized the image data because this will balance weights space and not to easily diverse in training process.

Here's one image example before and after normalization :

![alt text][image10]

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I set up training, validation and testing data in the end of 4th code cell of the IPython notebook.  

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 6th cell of the ipython notebook. 

My final model consisted of the following layers:

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


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 7th and 9th cell of the ipython notebook. 

To train the model, I used an AdamOptimizer with learning rate 0.001 to minimize loss. The batch size is 64 and number of epochs is 22.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 8th and 10th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 96.7%
* test set accuracy of 94.7%

I chose LeNet-5 as the base architecture because several traffic signs contained numbers and it's LeNet-5 originally designed for. 
In the beginning, I modified the last layer from 10 nodes to 43 nodes to accommondata all the sign class id from the data set.

For the 1st trail, I ran 10 epochs and chose batch size as 128, everytime I shuffled the training data in case neural network not training in a similar pattern as previous epoch.

The performance was not good (validation accuracy of the final epoch is 89.5%).

Here's training result of 1st trail :

![alt text][image11]

Then I tried to adjust initial weights with variance from 0.1 to 0.08. The 2nd trial works better (validation accuracy of the final epoch is 91.9%), but it was not good enough.

Here's training result of 2nd trail :

![alt text][image12]

Then I tried to adjust epochs and batch size. The batch size is chose as 64 . For smaller batch size, I increased epochs to 22.
The 3rd trial made a good progress for validation accuracy greater than 93%. (validation accuracy of the final epoch is 93.9%). 

Here's training result of 3rd trail :

![alt text][image13]

Afterwards, I tried to add one dropout layer before the last 2 fully connected layer, respectively. 
The 4th trial got a great improvement. (validation accuracy of the final epoch is 96.7%)

Here's training result of 4th trail :

![alt text][image14]

The dropout layer plays an important role to help train neural network not relying on certain subset of neurons by dropping out 50% neurons randomly in training process. 

The accuracy of training, validation and test set is above 94%, the final architecture worked pretty well.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I downloaded German Traffic Sign Recognition dataset on the web. Here are 16 German traffic signs randomly chose from the dataset :

![alt text][image15]

I think the image at 3rd column and 3rd row might be difficult to classify. It's speed limit (50km/h) but the number pattern is blurred and it might easily be classfied to other type of speed limit such as 30, 70 or 100km/h.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For the first 5 image from previous 16 images, here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h) | Speed limit (30km/h) |
| Keep right | Keep right |
| Speed limit (60km/h) | Speed limit (60km/h) |
| Traffic signals | Traffic signals |
| Speed limit (100km/h) | Speed limit (100km/h) |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. The accuracy is higher than test set accuracy and it might result from too few test samples chose and ocassionally chose those easier cases.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Here's the first five image with top 5 probability of calssification :

![alt text][image16]
