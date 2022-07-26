# traffic
 cs50ai Project 5a
 
 download and extract [data set](https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip) inside this directory

```
pip3 install -r requirements.txt

py traffic.py gtsrb
```

---
## Experimentation Process
The process used to find the best outcome for the task was to run the training and inference with
different parameters as per the tables bellow. The x-axis represents the number of dense (hidden)
layers in the network, and how many units they have. The y-axis is the number of convolutional layers,
how many filters they have and whether max-pooling was used after each layer. The values in table 1 are
training accuracy averaged over 3 runs of 10 epochs. The values in table 2 are testing accuracy averaged
over 3 runs of 10 epochs.

Table 1 - Training accuracy Phase 1

![Table 1](/results_01_training.png)

Table 2 - Testing Accuracy Phase 1

![Table 1](/results_01_testing.png)

The most promising runs used an x value of:
- 2 dense layers with 256 units
- 1 dense layer with 512 units

and a y value of:
- 2 convolutional layers with 32 filters and no max-pooling
- 3 convolutional layers with 16 filters and no max-pooling
- 3 convolutional layers with 32 filters and no max-pooling

Training accuracy was fairly accurate for quite a few runs achieving 99.0% or higher accuracy. The testing
accuracy however never achieved greater than 98.3%. The next phase of experimentation was to use dropout in
an attempt to improve testing accuracy on the most promising runs from the first phase.

In the tables bellow, the y-axis is the same as phase 1, but the x-axis now includes a value for dropout rate. Again
values were averaged over 3 runs of 10 epochs.

Table 3 - Training accuracy Phase 2

![Table 3](/results_02_training.png)

Table 4 - Testing Accuracy Phase 2

![Table 4](/results_02_testing.png)

Training accuracy for phase 2 was still quite high, although higher dropout rates tended to reduce the accuracy.
However, higher dropout rates would improve the accuracy during testing, e.g. the model was able to generalise better.

The best performing network when testing was:
- 3 convolutional layers with 32 filters, 1 dense layer with 512 units and a dropout rate of 0.5
