# **************************************************************************** #
#                                   Copyright                                  #
# **************************************************************************** #
#                                                                              #
# Copyright © 2021-2021 Thales SIX GTS FRANCE                                  #
#                                                                              #
# **************************************************************************** #
#                                   Copyright                                  #
# **************************************************************************** #


import torch
import sys
import datetime
import numpy as np
from therutils.parser.experiment import TrainExperiment as TrainExperimentParser
from therutils.parser.env.intrusion import Intruder as ParserIntruder
from therutils.parser.env.intrusion import Guard as ParserGuard
from therutils.parser.env.intrusion import FixedGuard as ParserFixedGuard
from therutils.parser.env.intrusion import Intrusion as ParserIntrusion
from therutils.parser.algorithm.qd_rl import ParserQDRLIntrusion
from therutils.parser.noise import EpsilonGreedy as ParserEpsilonGreedy
from therlib.tools.tools import seed_experiment
from therenv.se_star.intrusion.utils import make_env as make_intrusion_env
from plotintru import plot_rol, plot_ac


if __name__ == '__main__':
    """Simple testing script to experiment with the diversity metric."""
    sys.argv[2] += str(datetime.datetime.now())

    # Parse the options
    parser = TrainExperimentParser(ParserIntrusion(ParserIntruder(), ParserGuard(), ParserFixedGuard()),
                                   ParserQDRLIntrusion(ParserEpsilonGreedy()))
    args = parser.parse()
    seed_experiment(args['--train_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # We create the environment
    env = make_intrusion_env(args)

    args['--port'] = env.link.connector.simu_port
    parser._write_options(args['--exp'], 'exp_options.json', args)

    done = False


    def actor(state):
        print('state', state)
        if state[1] > .5136:
            return np.array([0.4, 0.03])
        else:
            return np.array([0., 0.03])


    def actor2(state):
        print('state2', state)

        return np.array([0.0, 0.0])


    def actor3(state):
        print('state', state)
        if state[0] > .136 and state[1] > .9:
            return np.array([-1, 1])
        else:
            return np.array([0., 0.03])


    plot_ac(args, [actor3, actor, actor2], device)
    # rol1 = one_ep(device, actor, env)
    # rol2 = one_ep(device, actor3, env)
    # print('distance', distance_test(rol1, rol2))
