# Digit recognition of 2digits at once - a modified MNIST hand written digit dataset

### This is the 3rd project I made during MITx Micromasters

## Sample pics :
![](https://raw.githubusercontent.com/stabgan/Modified-MNIST-2-digit-classification-at-once-using-pyTorch/master/part2-twodigit/sample_images/img20000.jpg) ![](https://raw.githubusercontent.com/stabgan/Modified-MNIST-2-digit-classification-at-once-using-pyTorch/master/part2-twodigit/sample_images/img20002.jpg) ![](https://raw.githubusercontent.com/stabgan/Modified-MNIST-2-digit-classification-at-once-using-pyTorch/master/part2-twodigit/sample_images/img20004.jpg) ![](https://raw.githubusercontent.com/stabgan/Modified-MNIST-2-digit-classification-at-once-using-pyTorch/master/part2-twodigit/sample_images/img20004.jpg) ![](https://raw.githubusercontent.com/stabgan/Modified-MNIST-2-digit-classification-at-once-using-pyTorch/master/part2-twodigit/sample_images/img20010.jpg)

1) I implemented a normal *Multilayer Perceptron* to train a basic model first which paralelly trains two models during forward propagation .
2) I implemented a *Convolutional Neural Network* using pyTorch and parallely trained two neural nets for recognizing two digits in a sample pic and then used cross_entropy loss individually for both of them and used Adam optimiser to train the model .

There are 30 epochs and each epoch takes around 1 min to train in macbook air 2015 model without CUDA .
The model converges close to 97.42% as I used Dropout in many layers to avoid overfitting . Without Droupout the model converges above 98% .

![Screenshot 2019-07-30 at 12 13 19 PM](https://user-images.githubusercontent.com/20128859/62106679-756dae80-b2c3-11e9-94cc-2cd63c99b7ae.png)

