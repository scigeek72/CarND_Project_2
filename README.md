# CarND_Project_2
Contains the files from Deep Learning Project

### Data Set Summary & Exploration

The data is loaded using the `pickle` package. The train and test data was provided seperately. The code on how to read those is provided in the first cell of the notebook.

* The size of the training set:                     39209
* The size of the testing set:                      12630
* The shape of a traffic sign image:                (32,32,3)
* The number of unique classes/labels in the data set: 43

#### Visualization

Visualizing some of the traffic signs. The code is in cell 6 of the jupyter notebook. 

![alt text](https://github.com/scigeek72/CarND_Project_2/blob/master/Visualization%20Files/Visualization1.png)



Below is the histogram showing the frequeancy of each of the classes present in the dataset.

![alt text](https://github.com/scigeek72/CarND_Project_2/blob/master/Visualization%20Files/hist1.png)


### Design and Test a Model Architechture

First the training set has been split into a training subset and a validation subset using `sklearn`'s `train_test_split()` function.
The code for that is given below. 

```python
#shuffle and then split
from sklearn.utils import shuffle
X_train_gray, y_train = shuffle(X_train_gray, y_train)

from sklearn.model_selection import train_test_split
xTrain,xVal, yTrain,yVal = train_test_split(X_train_gray, y_train, test_size = 0.2, random_state = 144)
```

* The size of the training set: 31367
* The size of the validation set: 7842

For data preprocessing, I used the `keras library`'s **`ImageGenerator`**.

Below is the code segment which achieves this. 

We first center the data according to the suggestion in CS231n course offered at Stanford University. See [here](http://cs231n.github.io/neural-networks-2/).

```python
from keras.preprocessing.image import ImageDataGenerator

image_datagen = ImageDataGenerator(rotation_range=15.,
                                  zoom_range = 0.2,
                                  width_shift_range=0.1,
                                  height_shift_range =0.1,
                                  horizontal_flip = True)

```

Below is the visualization of a particular traffic sign (randomly selected) which goes through the `ImageGenerator`.


![alt text](https://github.com/scigeek72/CarND_Project_2/blob/master/Visualization%20Files/Visualization2.png)


#### Neural Net architechture

* Number of epochs: 60
* Batch Size : 128
* mu : 0
* sigma : `np.sqrt(2.0/n_train)` [Following a suggestion at the CS231n course referred earlier]

```diff
1.  Convolutional Layer 1: 5 x 5 Convolution (in: 32 x 32 x 1, out: 28 x 28 x 1)
2.  ReLU  
3 . max_pooling (2 x 2)
4.  Convolutional Layer 2: 5 x 5 Convolution (in: 28 x 28 x 1, out: 14 x 14 x 1)
5.  ReLU  
6.  max_pooling (2 x 2)
8.  Flatten layers (flatter layer 1 and flatten layer 2) concatenated 1176 + 400 = 1576
9.  Fully Connected layer 1: (in: 1576, out: 120)
10.  Dropout layer
11.  Fully Connected layer 2: (in: 120, out: 84)
12.  Dropout layer
13. Outerlayer (in: 84, out: 43(no. of classes))
14. Softmax 
```

For training the model, I used the above architecture for the neural net. 
For optimzation I have used Adam optimizer. 
I have used regularization in the form of `Dropout` with keep-probability = 0.5


The final architecture is based on LeNet model that was described in the class videos. However, i have modified the architechture a bit and played around with it a bit to optimize it.
In the architecture, I have also used a notion of attaching some of the early layers during the flattening layer (as borrowed from the inception model shown in the class videos on deep learning).

I had tinkered wit the model a lot. But unfortunately, couldn't capture the level of accuracy that the others have obtained or the LeNet architecture has attained.

* Training Accuracy;   88.5%
* Validation Accuracy: 87.6% 

#### Test 

I have used a set of 5 (five) traffic signs that I have donwloaded from the internet and applied my mdoel. 

The accuracy on the new test set: 60%



The five pictures are visualized below.

![alt text](https://github.com/scigeek72/CarND_Project_2/blob/master/Visualization%20Files/Test_5.png)

I

Below is a visualization of the probabilities associated with each prediction.

![alt text](https://github.com/scigeek72/CarND_Project_2/blob/master/Visualization%20Files/Visualization3.png)




### Comments:

As can be seen, the model isn't performed well. I was expecting a higher accuracy than the validation accuracy (see above) I have obtained. So there is lot of scope to play arround with the model and imrove it. 
I hope to work on it in future. 


