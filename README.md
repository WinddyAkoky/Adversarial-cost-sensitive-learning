
# The code of paper "Particularly protected class in adversarial learning"

**Note: 1.Before you run above code, you should change the path to the dataset!**
**2. When runing the code, plase use CPU rather than CUDA. If you want to use CUDA to accelerate runing, it is easy for reader to modify the original codes by modifying a few of lines codes.**

# Fast test 

File ```example.ipynb``` is a simple version to implement this paper. Through it, reader can know the details of implemetation of this paper. 

Moreover, the complete codes are placed in folders ```MNIST```, ```FASHION_MNIST``` and ```cifar```.

------------------------------

```example.ipybn``` 是这篇论文的一个简单实现。通过这份文件，读者可以了解到文章的实现细节。

更详细的代码被放在文件夹```MNIST```, ```FASHION_MNIST``` 和 ```cifar```中.

# MNIST, FASHION MNIST, CIFAR

Since the code structure for each folder is the same, let's just use CIFAR as an example:

1. Run ```Lenet_CIFAR_train.py``` to train a model with normal training.
2. Run ```Lenet_CIFAR_adv_train.py``` to train a model with adversarial training.
3. Run ```CSA_cifar.py``` to train some CSA models.
4. Run ```CSE_cifar.py``` to train some CSE models.
5. Run ```evaluation.py``` to test the performance of the model.

