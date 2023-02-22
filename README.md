# Deep-Neural-Networks
Multi-layer neural network to classify images of fashion items (10 different classes) from the Fashion MNIST dataset from scratch

This DNN implemented using Stochastic gradient descent. This algorithm works for any number of hidden layers (Even with zero!). ReLU activation is assumed for all the hidden layers, and soft-max activaton for the output layer. Just edit the hyper parameters as you wish from the main function. If you want to tune the hyper parameters, add each parameter to the corresponding list in the main function, and change the key word argument of dnn class to ```True```. If the tune parameter is set to ```False```, the network chooses a random hyperparameter set.

# Dataset 


  [Train Images](https://s3.amazonaws.com/jrwprojects/fashion_mnist_train_images.npy)
  
  [Train Labels](https://s3.amazonaws.com/jrwprojects/fashion_mnist_train_labels.npy)
  
  [Test Images](https://s3.amazonaws.com/jrwprojects/fashion_mnist_test_images.npy)
  
  [Test Labels](https://s3.amazonaws.com/jrwprojects/fashion_mnist_test_labels.npy)
  
Example data:

  <img src = "https://github.com/shivakumar-tekumatla/Deep-Neural-Networks/blob/main/Best%20Performing/out.jpeg">

Run the following command .

``` python3 DNN_MNIST.py```
## Hyper parameters 

```
  Batch Size = 254, 
  Epsilon = 0.09, 
  Epochs = 300, 
  Alpha = 0.0025, 
  Hidden layers= 7,
  Hidden units at each hidden layer = [512, 512, 512, 512, 512, 512, 512]
```
# Test Accuracy  - ```90.02%```
### Training Error is  ```0.04018205912911439```
### Test Error is  ```0.8894396867771397```


# Weight Plots at different layers 
### After 1st layer , the plots may not make sense !
<img src="https://github.com/shivakumar-tekumatla/Deep-Neural-Networks/blob/main/Best%20Performing/Screenshot%202023-02-18%20at%2011.05.57%20PM.png" width="1000">

<img src="https://github.com/shivakumar-tekumatla/Deep-Neural-Networks/blob/main/Best%20Performing/Screenshot%202023-02-18%20at%2011.06.18%20PM.png">

<img src="https://github.com/shivakumar-tekumatla/Deep-Neural-Networks/blob/main/Best%20Performing/Screenshot%202023-02-18%20at%2011.06.34%20PM.png">

<img src="https://github.com/shivakumar-tekumatla/Deep-Neural-Networks/blob/main/Best%20Performing/Screenshot%202023-02-18%20at%2011.06.47%20PM.png">

<img src="https://github.com/shivakumar-tekumatla/Deep-Neural-Networks/blob/main/Best%20Performing/Screenshot%202023-02-18%20at%2011.07.02%20PM.png">

<img src="https://github.com/shivakumar-tekumatla/Deep-Neural-Networks/blob/main/Best%20Performing/Screenshot%202023-02-18%20at%2011.07.16%20PM.png">

<img src = "https://github.com/shivakumar-tekumatla/Deep-Neural-Networks/blob/main/Best%20Performing/Screenshot%202023-02-18%20at%2011.07.30%20PM.png">

<img src ="https://github.com/shivakumar-tekumatla/Deep-Neural-Networks/blob/main/Best%20Performing/Screenshot%202023-02-18%20at%2011.07.46%20PM.png">
