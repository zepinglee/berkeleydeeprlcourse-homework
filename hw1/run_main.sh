#!/bin/bash

for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2;
do
    python3 main.py $e --behavior_cloning;
    python3 main.py $e --dagger;
    python3 plot_returns.py $e;
done
