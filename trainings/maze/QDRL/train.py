# **************************************************************************** #
#                                   Copyright                                  #
# **************************************************************************** #
#                                                                              #
# Copyright © 2021-2021 Thales SIX GTS FRANCE                                  #
#                                                                              #
# **************************************************************************** #
#                                   Copyright                                  #
# **************************************************************************** #


from therenv.se_star.intrusion.utils import make_env as make_intrusion_env
from therlib.criterion.qd_rl import QDRL as QDRLCriterion
from therlib.criterion.util import soft_update
from therlib.logger.experiment import PopAgent as ExpLoggerAgent, create_experiment, stop_experiment
from therutils.model.pendulum.deterministic_policy_gradient import LowDPolicyNetwork as PolicyNetwork
from therutils.model.pendulum.deterministic_policy_gradient import LowDActionValueNetworkTD3 as ActionValueNetwork
from therutils.parser.experiment import TrainExperiment as TrainExperimentParser
from therutils.parser.env.intrusion import Intruder as ParserIntruder
from therutils.parser.env.intrusion import Guard as ParserGuard
from therutils.parser.env.intrusion import FixedGuard as ParserFixedGuard
from therutils.parser.env.intrusion import Intrusion as ParserIntrusion
from therutils.parser.algorithm.qd_rl import ParserQDRLIntrusion
from therutils.parser.noise import EpsilonGreedy as ParserEpsilonGreedy
from therlib.tools.tools import seed_experiment
from utils import Archive, create_element, KNN, QDRLReplayBuffer, NoveltyScore, transs
from plotintru import plot_rol, plot_ac
from utils import keep_efficient, distance_test
import os
import sys
import torch
import datetime
import copy
import logging
import numpy as np

my_logger = logging.getLogger(__name__)
my_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s  %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
my_logger.addHandler(stream_handler)
my_logger.propagate = False


# TODO be careful with the distinction between computing the diversity between state and between actors
# TODO update the diversity of actors only with respect to the new actors (way faster)


class QDRLAlgo:
    """
    The QDRL Algorithm

    Attributes
        args: the arguments
        device: the device
        env: the environment

    """

    def __init__(self, args, device, env):
        """
        Building the QDRL Algorithm
        Args:
            args: the arguments
            device: the device
            env: the environment
        """
        self.device = device
        self.args = args
        self.env = env
        print('err',args)
        self.random_state = np.random.RandomState(args['--train_seed'])

        action_dim = self.env.action_space.shape[0]
        state_dim = self.env.observation_space.shape[0]

        # We create the value and policy networks as well as their target
        self.critic_net_diversity1, self.critic_net_diversity2, self.target_critic_net_diversity1, self.target_critic_net_diversity2, \
        self.critic_net_quality1, self.critic_net_quality2, self.target_critic_net_quality1, self.target_critic_net_quality2 = [
            ActionValueNetwork(state_dim, action_dim, args['--hidden']).to(device)
            for _ in range(8)
        ]

        self.selected_actors = []
        # TODO check the copy system of selected actors
        for _ in range(args['--population_size']):
            actor = {'net': PolicyNetwork(state_dim, action_dim, args['--hidden'], init_w=0.1).to(device)}
            # We create the optimizers
            actor['optimizer'] = torch.optim.Adam(actor['net'].parameters(), lr=args['--policy_lr'])
            self.selected_actors.append(actor)

        self.critic_quality1_optimizer = torch.optim.Adam(self.critic_net_quality1.parameters(), lr=args['--value_lr'])
        self.critic_quality2_optimizer = torch.optim.Adam(self.critic_net_quality2.parameters(), lr=args['--value_lr'])
        self.critic_diversity1_optimizer = torch.optim.Adam(self.critic_net_diversity1.parameters(),
                                                            lr=args['--value_lr'])
        self.critic_diversity2_optimizer = torch.optim.Adam(self.critic_net_diversity2.parameters(),
                                                            lr=args['--value_lr'])

        # We initialize the target models to be identical to the other models
        soft_update(self.critic_net_quality1, self.target_critic_net_quality1, soft_tau=1.)
        soft_update(self.critic_net_quality2, self.target_critic_net_quality2, soft_tau=1.)
        soft_update(self.critic_net_diversity1, self.target_critic_net_diversity1, soft_tau=1.)
        soft_update(self.critic_net_diversity2, self.target_critic_net_diversity2, soft_tau=1.)

        # We create the replay buffer
        self.replay_buffer = QDRLReplayBuffer(args['--buffer'])

        # We create the criterion
        # TODO: figure out why not two TD3
        self.qdrl_criterion = QDRLCriterion(
            self.critic_net_diversity1, self.critic_net_diversity2, self.target_critic_net_diversity1,
            self.target_critic_net_diversity2,
            self.critic_net_quality1, self.critic_net_quality2, self.target_critic_net_quality1,
            self.target_critic_net_quality2,
            gamma=args['--gamma'], soft_tau=args['--soft_tau'],
            noise_std=args['--g_smooth_sigma'], noise_clip=args['--g_smooth_clip'],
            device=device
        )

        # We prepare the experiment
        exp_options = {
            'critic_loss_diversity_1': {'plot': 'line', 'yscale': 'log'},
            'critic_loss_diversity_2': {'plot': 'line', 'yscale': 'log'},
            'critic_loss_quality_1': {'plot': 'line', 'yscale': 'log'},
            'critic_loss_quality_2': {'plot': 'line', 'yscale': 'log'},
            'score': {'plot': 'line', 'yscale': 'linear'},
            'return_test': {'plot': 'line', 'yscale': 'linear'},
            'return_test_sparse': {'plot': 'line', 'yscale': 'linear'},
            'actor_loss': {'plot': 'line', 'yscale': 'linear'},
        }
        agent_id = 0
        description = 'TD3: {} with {} frames for training'.format(
            args['--env_name'], args['--budget']
        )
        self.exp_id = create_experiment(args['--exp'], description, './', exp_options)
        my_logger.info('exp_id: {}'.format(self.exp_id))

        # QDRL init
        self.archive = Archive()
        knn = KNN(k=args['--k_knn'], distance_func=distance_test)
        self.novelty_score_evaluator = NoveltyScore(knn=knn)

        self.storage_critic = ExpLoggerAgent(
            self.exp_id, agent_id, os.path.join(args['--exp'], 'agent_' + str(agent_id)),
            {'critic_model_quality1': self.critic_net_quality1, 'critic_model_quality2': self.critic_net_quality2,
             'critic_model_diversity1': self.critic_net_diversity1,
             'critic_model_diversity2': self.critic_net_diversity2},
            {'critic1_optimizer_quality': self.critic_quality1_optimizer,
             'critic2_optimizer_quality': self.critic_quality2_optimizer,
             'critic1_optimizer_diversity': self.critic_diversity1_optimizer,
             'critic2_optimizer_diversity': self.critic_diversity2_optimizer}
            , {'hyperp': agent_id})  # no real use
        self.storage_actor_ids = {}

        self.step_idx = 0
        self.algo_iter = 0

    def compute_pareto_front(self):
        """
        Computes the Pareto front for QDRL

        Returns:
            (list) a list of ids of the actors in the pareto front
        """
        pareto_front = []
        final_size = min(self.args['--population_size'], len(self.archive.elements))

        # initial points
        remaining_pts = []
        for element in self.archive.elements:
            remaining_pts.append({'values': [element.quality, element.diversity], 'actor_id': element.id})

        iteration = 0
        while len(pareto_front) < final_size:
            efficient_pts = keep_efficient(remaining_pts)
            efficient_pts_truncated = efficient_pts[0:final_size - len(pareto_front)]
            pareto_front += efficient_pts_truncated
            new_remaining_pts = []
            pareto_front_ids = [p['actor_id'] for p in pareto_front]
            for p in remaining_pts:
                if p['actor_id'] not in pareto_front_ids:
                    new_remaining_pts.append(p)
            remaining_pts = new_remaining_pts
            iteration += 1
            if iteration == 1:
                my_logger.info('true pareto_front {}'.format(pareto_front))

        my_logger.info('final pareto_front {}'.format(pareto_front))
        assert (len(set(pareto_front_ids)) == len(pareto_front_ids))
        return pareto_front_ids

    def qd_split_pareto_front(self, pareto_front_ids):
        """
        Divides the pareto front into quality and diversity
        Args:
            pareto_front_ids: the pareto_front

        Returns:
            (list): the selected actors
        """
        half = int(len(pareto_front_ids) / 2)
        selected_actors = []
        for pareto_idx, ac_id in enumerate(pareto_front_ids):
            element = self.archive.elements[ac_id]
            actor_ = {'net': element.actor['net'], 'optimizer': element.actor['optimizer'], 'pareto_id': pareto_idx,
                      'id_copy': element.id}
            selected_actors.append(actor_)
        selected_actors_copy = [copy.deepcopy(ac) for ac in selected_actors]
        self.random_state.shuffle(selected_actors_copy)
        self.qdrl_criterion.actors_diversity = selected_actors_copy[:half]
        self.qdrl_criterion.actors_quality = selected_actors_copy[half:]
        my_logger.info('quality actors {}'.format([ac['id_copy'] for ac in self.qdrl_criterion.actors_diversity]))
        my_logger.info('diversity actors {}'.format([ac['id_copy'] for ac in self.qdrl_criterion.actors_quality]))
        return selected_actors_copy

    def update_actors_diversity(self):
        """
        Updates the diversity of all the actors
        """
        for element in self.archive.elements:
            element.diversity = self.novelty_score_evaluator.compute_ns_actor(self.archive, element)
            my_logger.info('diversity {} of element {}'.format(element.diversity, element.id))

    def one_episode_test_and_store(self, actor):
        """
        Run an episode from the actor on the env and decide whether or not to add to the archive and replay buffer if diverse enough
        Args:
            actor: the actor for which to execute an episode

        Returns:
            (int) the number of steps run in the environment
        """

        replay_buffer_temp = []
        actor['net'].eval()
        state = self.env.reset()
        done = False
        episode_reward = 0
        rollout = [state]
        step_idx_in_episode = 0
        detected = 0
        idx = 0

        while not done:
            idx += 1
            model_state = torch.FloatTensor(state).to(self.device)
            action = actor['net'](model_state).detach().cpu().numpy()
            next_state, reward, done, info_ = self.env.step(action)
            detected += info_['detections']
            episode_reward += reward
            rollout.append(next_state)
            if not done or step_idx_in_episode != 0:
                replay_buffer_temp.append(
                    {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}
                )
            # do not forget to update time and state
            step_idx_in_episode += 1
            state = next_state

        success = (detected == 0) and (info_['arrived'] == 1)
        element = create_element(episode_reward, success, rollout, actor)
        distance_to_archive, nn = self.archive.distance(element, distance_test)
        my_logger.info('new element distance to archive is {}'.format(distance_to_archive))
        if (
                distance_to_archive > self.args["--min_archive_inter_distance"]
                or nn.quality < episode_reward or nn.success < success
        ):
            if nn is not None and nn.quality < episode_reward and distance_to_archive < .001:
                print('CASE when close but better, replacing id {}, my parent {}', nn.id,
                      element.actor['id_copy'] if 'id_copy' in element.actor else None)
                self.archive.replace_element(nn.id, element)
            else:
                actor_id = self.archive.add_element(element)
                my_logger.info(
                    '+ADD+ to archive actor with id {} and episode reward {} success {} with parent {}'.format(
                        actor_id,
                        episode_reward,
                        success,
                        element.actor['id_copy'] if 'id_copy' in element.actor else None)
                )
                for trans in replay_buffer_temp:
                    self.replay_buffer.push(
                        trans['state'],
                        trans['action'],
                        trans['reward'],
                        trans['next_state'],
                        trans['done'],
                        actor_id
                    )
        else:
            my_logger.info(
                '-NOT- adding to archive actor with episode reward {} success {} because close to id {}, my parents {}'.format(
                    episode_reward, success, nn.id, element.actor['id_copy'] if 'id_copy' in element.actor else None))

        return len(rollout)

    def sample_replay_buffer_and_gradient_descent(self):
        """
        Routine that does the sampling from the replay buffer and does the gradient descent steps
        """
        for actor in self.selected_actors:
            actor['net'].train()
        self.critic_net_diversity1.train()
        self.critic_net_diversity2.train()
        self.critic_net_quality1.train()
        self.critic_net_quality2.train()
        if len(self.replay_buffer) > self.args['--batch_size']:
            # Sample from the relay buffer
            state_replay, action_replay, reward_replay, next_state_replay, done_replay, actor_id_replay = self.replay_buffer.sample(
                self.args['--batch_size']
            )
            reward_diversity_replay = np.array([self.archive.elements[ac_id].diversity for ac_id in actor_id_replay])

            # Compute, store and optimize the losses
            self.critic_loss_diversity1, self.critic_loss_diversity2, self.critic_loss_quality1, self.critic_loss_quality2, self.actor_losses = self.qdrl_criterion.loss(
                state_replay, action_replay, reward_replay, reward_diversity_replay,
                next_state_replay, done_replay
            )

            self.critic_quality1_optimizer.zero_grad()
            self.critic_diversity1_optimizer.zero_grad()
            self.critic_loss_diversity1.backward(retain_graph=True,
                                                 inputs=list(self.critic_net_diversity1.parameters()))
            self.critic_loss_quality1.backward(retain_graph=True, inputs=list(self.critic_net_quality1.parameters()))
            self.critic_quality2_optimizer.zero_grad()
            self.critic_diversity2_optimizer.zero_grad()
            self.critic_loss_diversity2.backward(retain_graph=True,
                                                 inputs=list(self.critic_net_diversity2.parameters()))
            self.critic_loss_quality2.backward(retain_graph=True, inputs=list(self.critic_net_quality2.parameters()))
            if self.algo_iter % self.args['--delay_policy_update'] == 0:
                for actor in self.selected_actors:
                    actor['optimizer'].zero_grad()
                    self.actor_losses[actor['pareto_id']].backward(inputs=list(actor['net'].parameters()))
            self.critic_quality1_optimizer.step()
            self.critic_quality2_optimizer.step()
            self.critic_diversity1_optimizer.step()
            self.critic_diversity2_optimizer.step()
            if self.algo_iter % self.args['--delay_policy_update'] == 0:
                for actor in self.selected_actors:
                    actor['optimizer'].step()
                soft_update(self.critic_net_quality1, self.target_critic_net_quality1, self.args['--soft_tau'])
                soft_update(self.critic_net_quality2, self.target_critic_net_quality2, self.args['--soft_tau'])
                soft_update(self.critic_net_diversity1, self.target_critic_net_diversity1, self.args['--soft_tau'])
                soft_update(self.critic_net_diversity2, self.target_critic_net_diversity2, self.args['--soft_tau'])

    def train_model(self):
        """
            Train the QDRL
        """
        my_logger.info('-------first evaluation of the initial actors')
        for actor in self.selected_actors:
            self.step_idx += self.one_episode_test_and_store(actor)
        self.update_actors_diversity()
        if self.args["--display_video_archive"]:
            actors = [transs(el.actor, self.device) for el in self.archive.elements]
            rollouts = [el.outcome for el in self.archive.elements]
            plot_rol(self.env, rollouts, self.device)

        my_logger.info('--------Starting the loop')
        last_save_idx = 0

        # We train the networks
        while self.step_idx < self.args['--budget']:
            if self.algo_iter % self.args['--delay_policy_update'] == 0:
                self.archive.print_info(my_logger)
                pareto_front_ids = self.compute_pareto_front()
                my_logger.info('pareto front ids {}'.format(pareto_front_ids))
                self.selected_actors = self.qd_split_pareto_front(pareto_front_ids)

            # Train/Update the actor and critic based on resampling transitions from the replay buffer
            gradient_steps = int(self.step_idx * self.args['--gradient_steps_ratio'] + 1)
            my_logger.info('Algorithm Iteration {} -- steps done {}'.format(self.algo_iter, self.step_idx))
            my_logger.info(
                'gradient_steps = {} {}'.format(gradient_steps, self.args['--gradient_steps_ratio']))
            for _ in range(gradient_steps):
                self.sample_replay_buffer_and_gradient_descent()

            if self.algo_iter % self.args['--delay_policy_update'] == 0:
                my_logger.info('-------new evaluation of the actors')
                for selected_actor in self.selected_actors:
                    self.step_idx += self.one_episode_test_and_store(selected_actor)
                self.update_actors_diversity()

                if self.args["--display_video_archive"] and self.algo_iter % self.args["--display_video_archive_frequency"] == 0:
                    rollouts = [el.outcome for el in self.archive.elements]
                    plot_rol(self.env, rollouts, self.device)
                    #with open('archive' + str(self.algo_iter), 'wb') as f:
                    #    import pickle
                    #    pickle.dump(self.archive, f)
                    rollouts = [el.outcome for el in self.archive.elements if el.success]
                    plot_rol(self.env, rollouts, self.device)

            # Save and print performance information
            if self.step_idx - last_save_idx > self.args["--save_interval_performance"] and self.step_idx > 1:
                last_save_idx = self.step_idx
                stor = {
                    'critic_loss_quality_1': self.critic_loss_quality1.item(),
                    'critic_loss_quality_2': self.critic_loss_quality2.item(),
                    'critic_loss_diversity_1': self.critic_loss_diversity1.item(),
                    'critic_loss_diversity_2': self.critic_loss_diversity2.item()
                }

                self.storage_critic.performance(self.step_idx, stor)
                self.storage_critic.write()
                my_logger.info('Loss at {}/{}: value1={:.4}, value2={:.4}'.format(
                    self.step_idx, self.args['--budget'], self.critic_loss_quality1.item(),
                    self.critic_loss_quality2.item(),
                ))

            # save the weights of the model
            if True:  # step_idx % args['--save_interval'] == 0:
                for idx, ac in enumerate(self.selected_actors):
                    actor_archive_idx = pareto_front_ids[ac['pareto_id']]
                    if actor_archive_idx in self.storage_actor_ids:
                        storage_ac = self.storage_actor_ids[actor_archive_idx]
                    else:
                        storage_ac = ExpLoggerAgent(
                            self.exp_id, actor_archive_idx + 1,
                            os.path.join(self.args['--exp'], 'agent_' + str(actor_archive_idx + 1)),
                            {'actor_model': ac['net']},
                            {'actor_optimizer': ac['optimizer']},
                            {'hyperp': 1}
                        )
                        self.storage_actor_ids[actor_archive_idx] = storage_ac

                    stor = {}
                    stor['actor_loss'] = self.actor_losses[ac['pareto_id']].item()
                    stor['return_test'] = self.archive.elements[actor_archive_idx].quality
                    stor['return_test_sparse'] = self.archive.elements[actor_archive_idx].success
                    storage_ac.performance(self.step_idx, stor)
                    storage_ac.write()
                    storage_ac.state(self.step_idx)

            self.algo_iter += 1

        self.env.close()
        self.storage.close()
        stop_experiment(self.exp_id)


def one_episode_test(init_state, device, actor_net, env):
    done = False
    episode_reward = 0
    state = init_state
    while not done:
        model_state = torch.FloatTensor(state).to(device)
        action = actor_net(model_state).detach().cpu().numpy()
        next_state, reward, done, info_ = env.step(action)
        episode_reward += reward
        state = next_state
    return episode_reward


def train_model(args):
    seed_experiment(args['--train_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # We create the environment
    env = make_intrusion_env(args)

    args['--port'] = env.link.connector.simu_port
    parser._write_options(args['--exp'], 'exp_options.json', args)

    args["--save_interval_performance"] = 1000
    qdrl_algo = QDRLAlgo(args, device, env)
    qdrl_algo.train_model()


if __name__ == '__main__':
    sys.argv[2] += str(datetime.datetime.now())

    # Parse the options
    parser = TrainExperimentParser(ParserIntrusion(ParserIntruder(), ParserGuard(), ParserFixedGuard()),
                                   ParserQDRLIntrusion(ParserEpsilonGreedy()))
    _args = parser.parse()
    train_model(_args)
