#!/usr/bin/env python3

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle


def plot(ax, path, label):
    max_steps = 5e6
    with open(path, 'rb') as f:
        data = pickle.load(f)
    timestep = np.array(data['timestep'])
    mean_episode_reward = np.array(data['mean_episode_reward'])
    best_mean_episode_reward = np.array(data['best_mean_episode_reward'])

    indeces = np.where(timestep < max_steps)

    timestep = timestep[indeces]
    mean_episode_reward = mean_episode_reward[indeces]
    best_mean_episode_reward = best_mean_episode_reward[indeces]

    ax.plot(timestep, mean_episode_reward, label=label+'_mean')
    ax.plot(timestep, best_mean_episode_reward, label=label+'_best')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--dqn', type=str, default='')
    parser.add_argument('--ddqn', type=str, default='')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s: %(filename)s, %(lineno)d: %(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    fig, ax = plt.subplots()

    if args.dqn:
        plot(ax, args.dqn, 'DQN')
    if args.ddqn:
        plot(ax, args.ddqn, 'DDQN')

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Reward')
    if args.dqn:
        if args.ddqn:
            ax.set_title('DQN vs Double DQN')
        else:
            ax.set_title('DQN')
    else:
        ax.set_title('Double DQN')
    ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='x')
    ax.legend()
    fig.savefig('data/dqn.pdf')


if __name__ == '__main__':
    main()
