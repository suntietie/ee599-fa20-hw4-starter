README (HW4 compatibility problem)

For the compatibility classification problem, I use a pretrained model (mobile net v2) to train the deep 
learning parameters.

To analysis the input, I overlay one picture on the other one, so the input channel will going to be 6,
instead of 3. The other modification of the model is that I modify the output class number to 1, and add 
a sigmoid function to turn the output into a number between 0 and 1. And set threshold as 0.5 to see their 
class. I trained with learning rate of 0.002, 10 epochs.

Finally training and validation accuracy for the model is around 60%. The result is not perfect, but it 
shows some efforts, because for a large dataset, the accuracy for a dummy 2 class classifier (True or 
False) is likely to be 50%. 