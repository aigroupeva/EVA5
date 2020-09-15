1> Abhinav Rana (rabhinavcs@gmail.com)

2> Prashant Shinagare (techemerging1@gmail.com)

3> Pruthiraj Jayasingh (data.pruthiraj@gmail.com)



# Assignment 7 : Cifar-10,Depth wise Separable Convolution and dilation Convolution.

About the Code :

​			1. Data_utils.py : This files contain All the data Transforms , and data loading functions.

2. Model_cifar.py: Specific to model building and Designing . Every time we work in a new data set can build a new file with respective architecture  (only file we need to create in every data set change)
3. model_utils.py: this function is based with Training function , testing function , building_model function , getting model summary function , get_test_accuracy, Class based accuracy function etc. 
4. plot_utils.py: This is the function responsible for plotting. it is having the sample plotting function , miss classification function and the model performance plot function (accuracy,loss)
5. regularization.py: This file contents all the regularization functions.

About Cifar - 10:

1. Total parameter used: 447,050

2. Number of epochs : 40

3. Best test accuracy : 86.18  (86.26 is max )

4. Best train accuracy : 97.07 

5. Individual class accuracy:

    

   ```
   Accuracy of plane : 100 %
   Accuracy of   car : 93 %
   Accuracy of  bird : 79 %
   Accuracy of   cat : 80 %
   Accuracy of  deer : 83 %
   Accuracy of   dog : 88 %
   Accuracy of  frog : 89 %
   Accuracy of horse : 100 %
   Accuracy of  ship : 96 %
   Accuracy of truck : 94 %
   ```

Analysis: the model is over fit as the gap between training and testing is high after cert-en epochs . can use image augmentation and batch norm to overcome this. 

plots:
https://github.com/aigroupeva/EVA5/blob/master/session7/model_history.png

Miss classification images:

https://github.com/aigroupeva/EVA5/blob/master/session7/model_misclassified.png
