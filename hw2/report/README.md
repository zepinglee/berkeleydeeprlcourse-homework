# Policy Gradients

## Problem 4

Graph the results of your experiments using the plot.py file we provide. Create
two graphs.

In the first graph, compare the learning curves (average return at each iteration) for the experiments prefixed with `sb_`.

![Small batch](4.1-small-batch.pdf)

In the second graph, compare the learning curves for the experiments prefixed
with `lb_`. (The large batch experiments.)

![Large batch](4.2-large-batch.pdf)

- Which gradient estimator has better performance without advantage-centering— the trajectory-centric one, or the one using reward-to-go?

  The reward-to-go has better performance.

- Did advantage centering help?

  The advantage centering helps faster convergence on small batches but it
  doesn't have significant effects on large batches.

- Did the batch size make an impact?

  Large batch size reduces the variance of returns.


- Provide the exact command line configurations you used to run your
  experiments. (To verify batch size, learning rate, architecture, and so on.)

```
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -dna --exp_name sb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna --exp_name sb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg --exp_name sb_rtg_na
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -dna --exp_name lb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -dna --exp_name lb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg --exp_name lb_rtg_na
```



## Problem 5

- Given the `b*` and `r*` you found, provide a learning curve where the policy
  gets to optimum (maximum score of 1000) in less than 100 iterations.
  (This may be for a single random seed, or averaged over multiple.)

![InvertedPendulum](5.InvertedPendulum.pdf)

- Provide the exact command line configurations you used to run your
  experiments.

```
python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 5 -l 2 -s 64 -b 1000 -lr 0.01 -rtg --exp_name hc_b1000_r0.01
```



## Problem 7

Plot a learning curve for the above command. You should expect to achieve an average return of around 180.

![LunarLander](7-LunarLander.pdf)

The learning curve for LunarLander.



## Problem 8

- How did the batch size and learning rate affect the performance?

  The best parameters are b = 50000 and r = 0.02 .
  Large batch size and large learning rate help the policy converge faster.

- Once you’ve found suitable values of b and r among those choices, use b and r
  and run the following commands:

```
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 --exp_name hc_b50000_r0.02
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --exp_name hc_rtg_b50000_r0.02
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 --nn_baseline --exp_name hc_bl_b50000_r0.02
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline --exp_name hc_rtg_bl_b50000_r0.02
```

The run with reward-to-go and the baseline should achieve an average score close to 200. Provide a single plot plotting the learning curves for all four runs.

![HalfCheetah](8-HalfCheetah.pdf)


Note that the tanh activation function is important! I first chose ReLU and the
final performance could hardly achieve average return of 100.
