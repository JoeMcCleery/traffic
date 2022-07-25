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
different parameters as per the table bellow. The x-axis represents the number of dense (hidden)
layers in the network, and how many units they have. The y-axis is the number of convolutional layers,
how many filters they have and whether max-pooling was used after each layer. The values in the table
are (training accuracy / test accuracy) averaged over 3 runs of 10 epochs.

| Filters-Pooling-Layers \ Units-Layers | 0-0 | 128-1 | 128-2 | 128-3 | 256-1 | 256-2 | 256-3 | 512-1 | 512-2 | 512-3 |
|--------------------------------------:|----:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
|                             0-false-0 |     |       |       |       |       |       |       |       |       |       |
|                              0-true-0 |     |       |       |       |       |       |       |       |       |       |
|                             8-false-1 |     |       |       |       |       |       |       |       |       |       |
|                              8-true-1 |     |       |       |       |       |       |       |       |       |       |
|                            16-false-1 |     |       |       |       |       |       |       |       |       |       |
|                             16-true-1 |     |       |       |       |       |       |       |       |       |       |
|                            32-false-1 |     |       |       |       |       |       |       |       |       |       |
|                             32-true-1 |     |       |       |       |       |       |       |       |       |       |
|                             8-false-2 |     |       |       |       |       |       |       |       |       |       |
|                              8-true-2 |     |       |       |       |       |       |       |       |       |       |
|                            16-false-2 |     |       |       |       |       |       |       |       |       |       |
|                             16-true-2 |     |       |       |       |       |       |       |       |       |       |
|                            32-false-2 |     |       |       |       |       |       |       |       |       |       |
|                             32-true-2 |     |       |       |       |       |       |       |       |       |       |
|                             8-false-3 |     |       |       |       |       |       |       |       |       |       |
|                              8-true-3 |     |       |       |       |       |       |       |       |       |       |
|                            16-false-3 |     |       |       |       |       |       |       |       |       |       |
|                             16-true-3 |     |       |       |       |       |       |       |       |       |       |
|                            32-false-3 |     |       |       |       |       |       |       |       |       |       |
|                             32-true-3 |     |       |       |       |       |       |       |       |       |       |

