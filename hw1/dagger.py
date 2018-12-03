#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from keras import models
import matplotlib.pyplot as plt

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--dagger_iteration', type=int, default=20,
                        help='Number of dagger iteration')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    print('loading and building behavioral policy')
    my_model_name = args.envname +  '-my-model.h5'
    model = models.load_model(my_model_name)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        return_mean_array = []
        return_std_array = []
        obs_shape = 0
        action_shape = 0


        for dagger_iteration in range(args.dagger_iteration):
            print('dagger_iteration',dagger_iteration)
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    # trained model

                    obs_reshaped = obs.reshape((1,obs.shape[0]))
                    action_reshaped = model.predict(obs_reshaped)
                    action = action_reshaped.reshape((action_reshaped.shape[1],))
                    obs_shape = obs.shape[0]
                    action_shape = action_reshaped.shape[1]

                    observations.append(obs)
                    expert_action = policy_fn(obs[None,:])
                    actions.append(expert_action)

                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))
            return_mean_array.append(np.mean(returns))
            return_std_array.append(np.std(returns))


            train_data = np.array(observations)
            train_labels = np.array(actions)
            num_training_samples = train_labels.shape[0]
            train_labels = train_labels.reshape((num_training_samples,action_shape))
            EPOCHS = 10
            model.fit(train_data, train_labels, epochs=EPOCHS, validation_split=0.2, verbose=1)

    plt.errorbar(range(1,args.dagger_iteration+1),return_mean_array,yerr = return_std_array)
    plt.show()

if __name__ == '__main__':
    main()
