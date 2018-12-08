#!/usr/bin/env python3

import argparse
import logging
import os

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import tensorflow as tf

import tf_util
import load_policy

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('envname', default='Ant-v2', type=str)
parser.add_argument('--save', default='data', metavar='DIR')
parser.add_argument(
    '-b',
    '--batch-size',
    default=100,
    type=int,
    metavar='N',
    dest='batch_size')
parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.01,
    type=float,
    metavar='LR',
    dest='lr')
parser.add_argument('-s', '--steps', default=20000, type=int)
parser.add_argument('-e', '--epochs', default=20, type=int)
parser.add_argument('--log_freq', default=0, type=int)
parser.add_argument('-bc', '--behavior_cloning', action='store_true')
parser.add_argument('-da', '--dagger', action='store_true')
args = parser.parse_args()


class Agent(nn.Module):
    def __init__(self, env, num_hidden_units):
        super().__init__()
        num_input = np.prod(env.observation_space.shape)
        num_output = np.prod(env.action_space.shape)
        self.net = nn.Sequential(
            nn.Linear(num_input, num_hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden_units, num_output),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.net(x)
        return x


def get_actions(agent, observations):
    if isinstance(agent, nn.Module):
        with torch.no_grad():
            observations = torch.from_numpy(observations).float()
            if torch.cuda.is_available():
                observations = observations.cuda()
            actions = agent(observations)
            if torch.cuda.is_available():
                actions = actions.cpu()
            actions = actions.numpy()
    else:
        actions = agent(observations)

    return actions


def run_rollout(env, agent):
    observations = []
    actions = []
    totalr = 0.
    step = 0
    obs = env.reset()
    done = False
    while not done:
        action = get_actions(agent, obs[np.newaxis, :])[0]
        observations.append(obs)
        actions.append(action)
        obs, r, done, _ = env.step(action)
        totalr += r
        step += 1
        if step >= env.spec.timestep_limit:
            break

    return observations, actions, totalr


def run_agent(env, agent, rollouts=0,
              steps=0) -> (np.ndarray, np.ndarray, np.ndarray):
    if isinstance(agent, nn.Module):
        agent.eval()

    returns = []
    observations = []
    actions = []
    if rollouts > 0:
        for i in range(rollouts):
            obs, act, r = run_rollout(env, agent)
            observations += obs
            actions += act
            returns.append(r)
    else:  # steps > 0:
        while len(observations) < steps:
            obs, act, r = run_rollout(env, agent)
            observations += obs
            actions += act
            returns.append(r)
        observations = observations[:steps]
        actions = actions[:steps]

    observations = np.array(observations)
    actions = np.array(actions)
    returns = np.array(returns)

    return observations, actions, returns


def eval_agent(env, agent) -> np.ndarray:
    obs, act, r = run_agent(env, agent, rollouts=20)
    return r


def train_agent(observations, actions, agent, optimizer, criterion, epochs=1):
    if isinstance(observations, np.ndarray):
        observations = torch.from_numpy(observations).float()
        actions = torch.from_numpy(actions).float()

    dataset = TensorDataset(observations, actions)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    agent.train()
    for epoch in range(epochs):
        for x, y in dataloader:
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            y_pred = agent(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def behavior_clone(env, expert, steps, epochs):
    agent = Agent(env, num_hidden_units=1024)

    if torch.cuda.is_available():
        agent = agent.cuda()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            agent.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(agent.parameters(), args.lr)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    train_iteration = 0
    history = []

    observations, actions, returns = run_agent(env, expert, steps=steps)
    logging.info('Generated {} steps for {} roll-outs'.format(
        observations.shape[0], returns.shape[0]))
    observations = torch.from_numpy(observations).float()
    actions = torch.from_numpy(actions).float()
    dataset = TensorDataset(observations, actions)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(epochs):
        logging.info('behavior cloning epoch {}'.format(epoch))
        # train_agent(obs, act, agent, optimizer, criterion)
        agent.train()
        for x, y in dataloader:
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            y_pred = agent(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_iteration += 1
            if train_iteration % args.log_freq == 0:
                r = eval_agent(env, agent)
                logging.info('iterations:{}\tmean: {:.2f}\tstd:{:.2f}'.format(
                    train_iteration, np.mean(r), np.std(r)))
                history.append((train_iteration, r))
                agent.train()

    iterations = np.array([data[0] for data in history])
    returns = np.stack([data[1] for data in history])
    logging.debug('{}, {}'.format(iterations.shape, returns.shape))
    np.savez(
        os.path.join(args.save, args.envname + '-behavior_clone.npz'),
        iterations=iterations,
        returns=returns,
    )


def dagger(env, expert, steps, epochs):
    agent = Agent(env, num_hidden_units=1024)

    if torch.cuda.is_available():
        agent = agent.cuda()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            agent.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(agent.parameters(), args.lr)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    train_iteration = 0
    history = []

    observations, actions, returns = run_agent(
        env, expert, steps=steps // epochs)
    logging.info('Generated {} steps for {} roll-outs'.format(
        observations.shape[0], returns.shape[0]))

    for epoch in range(epochs):
        logging.info('dagger epoch {}'.format(epoch))
        # train_agent(observations, actions, agent, optimizer, criterion)
        observations = torch.from_numpy(observations).float()
        actions = torch.from_numpy(actions).float()
        dataset = TensorDataset(observations, actions)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True)

        agent.train()
        for x, y in dataloader:
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            y_pred = agent(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_iteration += 1
            if train_iteration % args.log_freq == 0:
                r = eval_agent(env, agent)
                logging.info('iterations:{}\tmean: {:.2f}\tstd:{:.2f}'.format(
                    train_iteration, np.mean(r), np.std(r)))
                history.append((train_iteration, r))
                agent.train()

        policy_observations, _, policy_returns = run_agent(
            env, agent, steps=steps // epochs)
        expert_actions = get_actions(expert, policy_observations)
        logging.info('Generated {} steps for {} roll-outs'.format(
            policy_observations.shape[0], policy_returns.shape[0]))

        observations = np.concatenate((observations, policy_observations))
        actions = np.concatenate((actions, expert_actions))

    iterations = np.array([data[0] for data in history])
    returns = np.stack([data[1] for data in history])
    logging.debug('{}, {}'.format(iterations.shape, returns.shape))
    np.savez(
        os.path.join(args.save, args.envname + '-dagger.npz'),
        iterations=iterations,
        returns=returns,
    )


def main():
    global args
    if args.log_freq == 0:
        args.log_freq = args.steps // args.batch_size // 2

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    logging.basicConfig(
        format='%(levelname)s: %(filename)s, %(lineno)d: %(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    env = gym.make(args.envname)
    logging.info('environment: {}'.format(args.envname))
    logging.info('observation space: {}'.format(env.observation_space.shape))
    logging.info('action space: {}'.format(env.action_space.shape))

    expert_policy_file = os.path.join('experts', args.envname + '.pkl')
    expert = load_policy.load_policy(expert_policy_file)
    with tf.Session():
        tf_util.initialize()

        if args.behavior_cloning:
            behavior_clone(
                env, expert, steps=args.steps, epochs=args.epochs // 2)

        if args.dagger:
            dagger(env, expert, steps=args.steps, epochs=args.epochs)


if __name__ == '__main__':
    main()
