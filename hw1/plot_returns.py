#!/usr/bin/env python3

import argparse
import logging
import os

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('envname', default='Ant-v2', type=str)
    parser.add_argument('--save', default='data', metavar='DIR')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s: %(filename)s, %(lineno)d: %(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    logging.info('plotting {}'.format(args.envname))
    fig, ax = plt.subplots()

    for method in ('behavior_clone', 'dagger'):
        file_path = os.path.join(args.save, args.envname+'-'+method+'.npz')
        data = np.load(file_path)
        iterations = data['iterations']
        returns = data['returns']
        # if len(returns.shape) == 1:
        #     returns = returns.reshape(iterations.shape[0], -1)
        logging.debug('{}, {}'.format(iterations.shape, returns.shape))

        mean = np.mean(returns, axis=1)
        std = np.std(returns, axis=1)

        ax.errorbar(iterations, mean, yerr=std, fmt='-o', label=method)
        # ax.fill_between(iterations, down, up, alpha=0.2)

    ax.set_title(args.envname)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Mean return')
    ax.legend(loc='upper left')
    fig.savefig(os.path.join(args.save, args.envname+'.pdf'))


if __name__ == '__main__':
    main()
