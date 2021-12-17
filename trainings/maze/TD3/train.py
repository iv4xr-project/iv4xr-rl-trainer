from therenv.se_star.intrusion.utils import make_env as make_intrusion_env
from therlib.criterion.td3 import TD3 as TD3Criterion
from therlib.criterion.util import soft_update
from therlib.logger.experiment import Agent as ExpLoggerAgent, create_experiment, stop_experiment
from therlib.memory.replay import ReplayBuffer
from therlib.noise.action_noise import Gaussian, EpsilonGreedy
from therlib.tools.tools import seed_experiment
from therutils.model.intrusion.deterministic_policy_gradient import LowDPolicyNetwork as PolicyNetwork
from therutils.model.intrusion.deterministic_policy_gradient import LowDActionValueNetworkTD3 as ActionValueNetwork
from therutils.parser.experiment import TrainExperiment as TrainExperimentParser
from therutils.parser.env.intrusion import Intruder as ParserIntruder
from therutils.parser.env.intrusion import Guard as ParserGuard
from therutils.parser.env.intrusion import FixedGuard as ParserFixedGuard
from therutils.parser.env.intrusion import Intrusion as ParserIntrusion
from therutils.parser.algorithm.td3 import ParserTD3Intrusion
from therutils.parser.noise import EpsilonGreedy as ParserEpsilonGreedy
import os
import sys
import torch
import datetime
from collections import deque


def train_model(args):
    seed_experiment(args['--train_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # We create the environment
    env = make_intrusion_env(args)

    args['--port'] = env.link.connector.simu_port
    parser._write_options(args['--exp'], 'exp_options.json', args)

    args["--save_interval_performance"] = 1000

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    # We create the action noise process for exploration around the behavior policy
    # TODO: handle conditional parsers
    # if args["--noise"] == 'Gaussian':
    #    noise_process_exploration = Gaussian(env.action_space, args['--g_min_sigma'], args['--g_max_sigma'],
    #                                         decay_period=args['--g_decay'])
    if args["--noise"] == 'epsilon_greedy':
        noise_process_exploration = EpsilonGreedy(args['--epsilon_start'], args['--epsilon_end'], args['--decay_period']
                                                  , action_space=env.action_space)

    # We create the value and policy networks as well as their target
    critic_net1, critic_net2, target_critic_net1, target_critic_net2 = [
        ActionValueNetwork(state_dim, action_dim, args['--hidden']).to(device)
        for _ in range(4)
    ]
    actor_net, target_actor_net = (
        PolicyNetwork(state_dim, action_dim, args['--hidden']).to(device),
        PolicyNetwork(state_dim, action_dim, args['--hidden']).to(device)
    )

    # We create the optimizers
    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=args['--policy_lr'])
    critic1_optimizer = torch.optim.Adam(critic_net1.parameters(), lr=args['--value_lr'])
    critic2_optimizer = torch.optim.Adam(critic_net2.parameters(), lr=args['--value_lr'])

    # We initialize the target models to be identical to the other models
    soft_update(critic_net1, target_critic_net1, soft_tau=1.)
    soft_update(critic_net2, target_critic_net2, soft_tau=1.)
    soft_update(actor_net, target_actor_net, soft_tau=1.)

    # We create the replay buffer
    if args["--replay_buffer_kickstart_file"] is not None:
        replay_buffer = ReplayBuffer.load_from_file(args["--replay_buffer_kickstart_file"])
    else:
        replay_buffer = ReplayBuffer(args['--buffer'])

    # We create the criterion
    td3_criterion = TD3Criterion(
        actor_net, target_actor_net,
        critic_net1, critic_net2, target_critic_net1, target_critic_net2,
        gamma=args['--gamma'], soft_tau=args['--soft_tau'],
        noise_std=args['--g_smooth_sigma'], noise_clip=args['--g_smooth_clip'],
        device=device
    )

    # We prepare the experiment
    exp_options = {
        'episode_reward_train': {'plot': 'line', 'yscale': 'linear'},
        'episode_reward_test': {'plot': 'line', 'yscale': 'linear'},
        'episode_reward_test_sparse': {'plot': 'line', 'yscale': 'linear'},
        'episode_reward_test_sparse_ring': {'plot': 'line', 'yscale': 'linear'},
        'score_train': {'plot': 'line', 'yscale': 'linear'},
        'score_test': {'plot': 'line', 'yscale': 'linear'},
        'score_test_sparse': {'plot': 'line', 'yscale': 'linear'},
        'score': {'plot': 'line', 'yscale': 'linear'},
        'actor_loss': {'plot': 'line', 'yscale': 'log'},
        'critic_loss_1': {'plot': 'line', 'yscale': 'log'},
        'critic_loss_2': {'plot': 'line', 'yscale': 'log'},
    }
    agent_id = 0
    description = 'TD3: {} with {} frames for training'.format(
        args['--env_name'], args['--budget']
    )
    exp_id = create_experiment(args['--exp'], description, './', exp_options)
    print('exp_id', exp_id)

    storage = ExpLoggerAgent(
        exp_id, agent_id, os.path.join(args['--exp'], 'agent_0'),
        {'critic_model1': critic_net1, 'critic_model2': critic_net2, 'actor_model': actor_net},
        {'critic1_optimizer': critic1_optimizer, 'critic2_optimizer': critic2_optimizer,
         'actor_optimizer ': actor_optimizer}
    )
    reward_buffer_test = deque(maxlen=100)
    reward_buffer_test_sparse = deque(maxlen=100)
    reward_buffer_test_sparse_ring = deque(maxlen=100)
    reward_buffer_train = deque(maxlen=100)

    # We train the networks
    step_idx = 0
    episode_idx = 0
    episode_reward_train = 0
    state = env.reset()
    step_idx_in_episode = 0

    while step_idx < args['--budget']:

        actor_net.eval()
        # Do one step in the environment and save information
        model_state = torch.FloatTensor(state).to(device)
        action = actor_net(model_state).detach().cpu().numpy()
        action = noise_process_exploration.get_action(action, t=step_idx)
        next_state, reward, done, _ = env.step(action)
        if not done or step_idx_in_episode != 0:
            replay_buffer.push(state, action, reward, next_state, done)
        episode_reward_train += reward

        # Train/Update the actor and critic based on resampling transitions from the replay buffer
        if step_idx % args['--delay_policy_update'] == 0:
            actor_net.train()
        critic_net1.train()
        critic_net2.train()
        if len(replay_buffer) > args['--batch_size']:
            # Sample from the relay buffer
            state_replay, action_replay, reward_replay, next_state_replay, done_replay = replay_buffer.sample(
                args['--batch_size']
            )
            # Compute, store and optimize the losses
            critic_loss1, critic_loss2, actor_loss = td3_criterion.loss(
                state_replay, action_replay, reward_replay,
                next_state_replay, done_replay
            )

            critic1_optimizer.zero_grad()
            critic_loss1.backward(retain_graph=True, inputs=list(critic_net1.parameters()))
            critic2_optimizer.zero_grad()
            critic_loss2.backward(retain_graph=True, inputs=list(critic_net2.parameters()))
            if step_idx % args['--delay_policy_update'] == 0:
                actor_optimizer.zero_grad()
                actor_loss.backward(inputs=list(actor_net.parameters()))
            critic1_optimizer.step()
            critic2_optimizer.step()
            if step_idx % args['--delay_policy_update'] == 0:
                actor_optimizer.step()
                soft_update(critic_net1, target_critic_net1, args['--soft_tau'])
                soft_update(critic_net2, target_critic_net2, args['--soft_tau'])
                soft_update(actor_net, target_actor_net, args['--soft_tau'])

        # Save and print performance information
        if step_idx % args["--save_interval_performance"] == 0 and step_idx > 1 and len(reward_buffer_test) > 1:
            storage.performance(step_idx, {
                'critic_loss_1': critic_loss1.item(),
                'critic_loss_2': critic_loss2.item(),
                'actor_loss': actor_loss.item()
            })
            mean_reward_train = sum(reward_buffer_train) / len(reward_buffer_train)
            mean_reward_test = sum(reward_buffer_test) / len(reward_buffer_test)
            mean_reward_test_sparse = sum(reward_buffer_test_sparse) / len(reward_buffer_test_sparse)

            storage.performance(step_idx, {'score_train': mean_reward_train})
            storage.performance(step_idx, {'score_test': mean_reward_test})
            storage.performance(step_idx, {'score_test_sparse': mean_reward_test_sparse})
            storage.write()
            print('Loss at {}/{}: value1={:.4}, value2={:.4},  policy={:.4}.'.format(
                step_idx, args['--budget'], critic_loss1.item(), critic_loss2.item(), actor_loss.item()
            ))
            print('Result train at {}/{}: {}.'.format(
                step_idx, args['--budget'], mean_reward_train
            ))
            print('Result test at {}/{}: {}.'.format(
                step_idx, args['--budget'], mean_reward_test
            ))

        # save the weights of the model
        if step_idx % args['--save_interval'] == 0:
            storage.state(step_idx)

        # do not forget to update time and state
        step_idx += 1
        step_idx_in_episode += 1
        state = next_state

        if done:  # the episode came to an end.
            episode_idx += 1
            step_idx_in_episode = 0
            storage.performance(step_idx, {'episode_reward_train': episode_reward_train})
            reward_buffer_train.append(episode_reward_train)
            episode_reward_train = 0

            if episode_idx % args["--test_frequency"] == 0:
                # Testing the learned policy on one episode
                actor_net.eval()
                total_number_test = 1
                episode_reward_test = 0
                episode_reward_test_sparse = 0
                episode_reward_test_sparse_ring = 0

                for test_number in range(total_number_test):
                    state_test = env.reset(intruder_position=args['--intruder_position_test'],
                                           reward=args['--reward'])
                    episode_reward_test += one_episode(state_test, device, actor_net, env)
                for test_number in range(total_number_test):
                    state_test = env.reset(intruder_position=args['--intruder_position_test'],
                                           reward='sparse')
                    episode_reward_test_sparse += one_episode(state_test, device, actor_net, env)
                for test_number in range(total_number_test):
                    state_test = env.reset(intruder_position='ring',
                                           reward='sparse')
                    episode_reward_test_sparse_ring += one_episode(state_test, device, actor_net, env)

                normalized_episode_reward_test = episode_reward_test / total_number_test
                normalized_episode_reward_test_sparse = episode_reward_test_sparse / total_number_test
                normalized_episode_reward_test_sparse_ring = episode_reward_test_sparse_ring / total_number_test

                reward_buffer_test.append(normalized_episode_reward_test)
                reward_buffer_test_sparse.append(normalized_episode_reward_test_sparse)
                reward_buffer_test_sparse_ring.append(normalized_episode_reward_test_sparse_ring)
                storage.performance(step_idx, {'episode_reward_test': episode_reward_test})
                storage.performance(step_idx, {'episode_reward_test_sparse': episode_reward_test_sparse})
                storage.performance(step_idx, {'episode_reward_test_sparse_ring': episode_reward_test_sparse_ring})

            state = env.reset(intruder_position=args['--intruder_position_train'], reward=args['--reward'])

    env.close()

    storage.state(step_idx)
    print('Loss at {}/{}: value1={:.4}, value2={:.4}, policy={:.4}.'.format(
        step_idx, args['--budget'], critic_loss1.item(), critic_loss2.item(), actor_loss.item()
    ))
    storage.close()
    stop_experiment(exp_id)


def one_episode(init_state, device, actor_net, env):
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


if __name__ == '__main__':
    sys.argv[2] += str(datetime.datetime.now())

    # Parse the options
    parser = TrainExperimentParser(ParserIntrusion(ParserIntruder(), ParserGuard(), ParserFixedGuard()),
                                   ParserTD3Intrusion(ParserEpsilonGreedy()))
    _args = parser.parse()
    train_model(_args)
