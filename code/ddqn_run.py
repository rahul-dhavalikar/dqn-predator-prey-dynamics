import gym
import numpy as np
import tensorflow as tf
import argparse
import itertools
import time
import os
import pickle
import code
import random

from make_env import make_env

from ddqn import DDQN
from experience import experience_buffer
import utility

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag_guided', type=str)
    parser.add_argument('--episodes', default=20000, type=int)
    parser.add_argument('--render', default=False, action="store_true")
    parser.add_argument('--testing', default=False, action="store_true")
    parser.add_argument('--random_seed', default=2, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--load_model', default=False, action="store_true")
    parser.add_argument('--model_path', default='./save/ddqn', help="where to save/load models")
    parser.add_argument('--exit_reward', default=-100, type=int)
    parser.add_argument('--max_episode_length', default=3000, type=int)

    args = parser.parse_args()

    # init env
    env = make_env(args.env)

    # set random seed
    env.seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    # Parameters
    # batch size for training
    batch_size = args.batch_size
    update_freq = 4
    # discount factor
    y = .99
    startE = 1
    endE = 0.1
    annealing_steps = 10000.
    num_episodes = args.episodes
    pre_train_steps = 10000
    max_epLength = args.max_episode_length
    # load previous saved model
    load_model = args.load_model
    # location of model
    path = args.model_path
    # rate to update target network
    tau = 0.001
    reward_exit_arena = args.exit_reward

    tf.reset_default_graph()

    # init DDQNs
    n_actions = [env.action_space[i].n for i in range(env.n)]
    state_sizes = [env.observation_space[i].shape[0] for i in range(env.n)]
    mainQN = [DDQN(n_actions[i], state_sizes[i]) for i in range(env.n)]
    targetQN = [DDQN(n_actions[i], state_sizes[i]) for i in range(env.n)]

    # init tensorflow
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    trainables = tf.trainable_variables()
    targetOps = updateTargetGraph(trainables, tau)

    # assign experience buffer for each agent
    experiences = [experience_buffer() for i in range(env.n)]

    # chance of random actions
    if args.testing:
        e = 0.1
        pre_train_steps = 0
    else:
        e = startE
        stepDrop = (startE - endE) / annealing_steps

    jList = []
    rList = []
    cList = []
    total_steps = 0

    if not os.path.exists(path):
        os.makedirs(path)

    # init log header
    log_header = ["episode"]
    log_header.append("steps")
    log_header.extend(["reward_{}".format(i) for i in range(env.n)])
    log_header.extend(["collisions_{}".format(i) for i in range(env.n)])
    print("Collecting statistics {}:".format(" ".join(log_header)))
    performance_logs = utility.Logger(log_header)

    with tf.Session() as sess:
        sess.run(init)
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        for episode in range(num_episodes):
            if episode % 10 == 0:
                print("Running episode ", episode)

            episodeBuffer = [experience_buffer() for i in range(env.n)]

            states = env.reset()
            steps = 0

            rAll = 0
            j = 0
            collision_count = np.zeros(env.n)
            episode_rewards = np.zeros(env.n)

            while True:
                steps += 1
                j += 1

                # render
                if args.render:
                    env.render()

                # act
                actions = []
                actions_onehot = []
                for i in range(env.n):
                    if np.random.rand(1) < e or total_steps < pre_train_steps:
                        action = random.randrange(n_actions[i])
                    else:
                        action = sess.run(mainQN[i].predict, feed_dict={mainQN[i].scalarInput: [states[i]]})[0]
                        # print(action)

                    speed = 0.9 if env.agents[i].adversary else 1

                    onehot_action = np.zeros(n_actions[i])
                    onehot_action[action] = speed
                    actions_onehot.append(onehot_action)
                    actions.append(action)

                # step
                states_next, rewards, done, info = env.step(actions_onehot)

                total_steps += 1
                for i in range(env.n):
                    if done[i]:
                        rewards[i] += reward_exit_arena
                    episodeBuffer[i].add(np.reshape(np.array([states[i], actions[i], rewards[i], states_next[i], done[i]]), [1, 5]))

                # learn
                if not args.testing:

                    if total_steps > pre_train_steps:
                        if e > endE:
                            e -= stepDrop

                        if total_steps % (update_freq) == 0:

                            for i in range(env.n):
                                trainBatch = experiences[i].sample(batch_size)
                                Q1 = sess.run(mainQN[i].predict,
                                              feed_dict={mainQN[i].scalarInput: np.vstack(trainBatch[:, 3])})
                                Q2 = sess.run(targetQN[i].Qout,
                                              feed_dict={targetQN[i].scalarInput: np.vstack(trainBatch[:, 3])})

                                end_multiplier = -(trainBatch[:, 4] - 1)
                                doubleQ = Q2[range(batch_size), Q1]
                                targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                                _ = sess.run(mainQN[i].updateModel,
                                             feed_dict={mainQN[i].scalarInput: np.vstack(trainBatch[:, 0]),
                                                        mainQN[i].targetQ: targetQ, mainQN[i].actions: trainBatch[:, 1]})

                                updateTarget(targetOps, sess)

                states = states_next

                # The following tasks needs to be common for both training and testing
                # Count total rewards from all agents, count collisions, check if any agent exited environment
                rAll += sum(rewards)
                episode_rewards += rewards
                collision_count += np.array(utility.count_agent_collisions(env))

                if any(done) or j > max_epLength:
                    episode_rewards = episode_rewards / steps

                    plog = [episode]
                    plog.append(steps)
                    plog.extend([episode_rewards[i] for i in range(env.n)])
                    plog.extend(collision_count.tolist())
                    performance_logs.add_log(plog)

                    break

            # Periodically save the model.
            if not args.testing and episode % 500 == 0:
                saver.save(sess, path + '/model-' + str(episode) + '.ckpt')
                print("Saved Model")
                log_file_name = path + '/logs-' + str(episode) + '.csv'
                performance_logs.dump(log_file_name)
                if episode >= 500:
                    old_log_file_name = path + '/logs-' + str(episode-500) + '.csv'
                    os.remove(old_log_file_name)

            for i in range(env.n):
                experiences[i].add(episodeBuffer[i].buffer)

            jList.append(j)
            rList.append(rAll)
            cList.append(np.sum(collision_count))

            # print(total_steps, mean of [steps, rewards, collisions] from last 10 episodes, epsilon greedy)
            if len(rList) % 10 == 0:
                print(total_steps, np.mean(jList[-10:]), np.mean(rList[-10:]), np.mean(cList[-10:]), e)


        #save the last training model
        if not args.testing:
            saver.save(sess, path + '/model-' + str(episode) + '.ckpt')
            log_file_name = path + '/logs-' + str(episode) + '.csv'
            performance_logs.dump(log_file_name)
