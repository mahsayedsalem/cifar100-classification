# cifar100-classification

A convolutional Neural Network trained on cifar-100 dataset from scratch obtaining validation accuracy of 50%

![Validation Accuracy](https://raw.githubusercontent.com/mahsayedsalem/cifar100-classification/master/images/val_acc.PNG)

Loss graph

![Validation Loss](https://raw.githubusercontent.com/mahsayedsalem/cifar100-classification/master/images/val_loss.PNG)

The initial experiments was done on Colab, follow the link in case you want to up and running in no time:
https://colab.research.google.com/drive/1PF5q1AtU0CMOEdz8oWEb9SIviBvNx6QW

The project structure is adopted and modified from https://github.com/MrGemy95/Tensorflow-Project-Template

Steps to run the project:
`git clone https://github.com/mahsayedsalem/cifar100-classification.git`

The cifar100.cofig.json file contains all the hyperparameters for the experiment. 

Open it and add the cifar-100 train and test pickles path

It's highly recommended to create a conda environment. After creating the environment activate it and go to the project folder from the anaconda prompt and type
`pip install -r requirements.txt`

The experiments folder currently has 2 experiments (checkpoints and summaries). The one with the highest accuracy is experiment two. 

To continue training on this model, just put the experiment name in the cifar100_config.json file, then run 
`python cifar100.py -c "cifar100_config.json"`

In case you wanted to start a new training, change the experiment name. You can modify all the hyper parameters from the cifar100_config file. 

cifar100_serve.py was used to create a .pb file to be used in the cifar100_predit.py script, but an error arised while running the cifar100_predict and as the deadline appraoches, I had no time to debug as I needed to deliver. 



