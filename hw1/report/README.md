## Question 2.2

Task           | Expert Mean | Expert STD | Mean    | STD
---            | ---         | ---        | ---     | ---
HalfCheetah-v2 | 4107.7      | 69.693     | 3969.9  | 122.75
Walker2d-v2    | 5533.6      | 36.253     | 346.12  | 376.01
Hopper-v2      | 3780.8      | 5.2116     | 1193.9  | 113.42
Ant-v2         | 4798.3      | 128.48     | 4338.9  | 610.99
Humanoid-v2    | 10374       | 43.331     | 373.84  | 105.71
Reacher-v2     | -3.6407     | 1.4249     | -8.4093 | 4.6135

The behavior cloning (BC) agent trained on HalfCheetah-v2 is comparable to the
expert agent while the one trained on Walker2d-v2 does not.
The expert data is produced by running for 10 roll-outs.
Both BC agents are simple feed-forward networks with single hidden layer of 256
units and ReLU activation function.
They are trained using Adam optimizer with batch size 64 and initial learning
rate 0.01 for 30 epochs.
Finally they are tested over 30 roll-outs to produce the mean and standard
deviation of the return.



## Question 3.2

![Dagger](dagger-vs-bc.pdf)

The agents are trained on Hopper-v2 with Behavior cloning and Dagger
algorithms, respectively.
The Dagger agent achieves higher mean return than the BC agent afer trained
on 10000 steps of expert data for 200 steps.
Both agent are feed-forward networks with single hidden layer of 1000
units and ReLU activation function.
