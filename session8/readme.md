# Assignment 7 : Cifar-10,Depth wise Separable Convolution and dilation Convolution.

About the Code :

1. Data_utils.py : This files contain All the data Transforms , and data loading functions.
2. Model_cifar.py: Specific to model building and Designing . Every time we work in a new data set can build a new file with respective architecture  (only file we need to create in every data set change)
3. model_utils.py: this function is based with Training function , testing function , building_model function , getting model summary function , get_test_accuracy, Class based accuracy function etc. 
4. plot_utils.py: This is the function responsible for plotting. it is having the sample plotting function , miss classification function and the model performance plot function (accuracy,loss)
5. regularization.py: This file contents all the regularization functions.
6.Models.py : Restnet18 Model is here.

About Cifar - 10:

1. Total parameter used: 11,173,962

2. Number of epochs : 40

3. Best test accuracy : 87.02%

4. Best train accuracy : 100.00

5. Individual class accuracy:

Accuracy of plane : 85 %
Accuracy of   car : 93 %
Accuracy of  bird : 78 %
Accuracy of   cat : 67 %
Accuracy of  deer : 65 %
Accuracy of   dog : 86 %
Accuracy of  frog : 83 %
Accuracy of horse : 92 %
Accuracy of  ship : 83 %
Accuracy of truck : 96 %
   ```

Analysis: the model is over fit as the gap between training and testing is high after cert-en epochs . can use image augmentation and batch norm to overcome this. 

plots:
https://github.com/aigroupeva/EVA5/blob/master/session8/model_history.png
Miss classification images:
https://github.com/aigroupeva/EVA5/blob/master/session8/Test_missclassified_images.jpg
