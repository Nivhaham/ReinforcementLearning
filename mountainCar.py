import math
import random
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
state = env.reset()
number_of_actions = env.action_space.n
alpha = 0.1  # learning rate
gamma = 1.0  # discount factor
num_of_episodes = 2000
pie_p = 0.04
pie_v = 0.0004



# print(state)
def generating_starting_positions():
    # starting_pos = np.random.uniform(-1.2, 0.6, 4)
    starting_pos = np.array([-1.2, -0.3, 0.1, 0.5])
    starting_pos = starting_pos.round(3)
    # print(starting_pos)

    # starting_velo = np.random.uniform(-0.07, 0.07, 8)
    starting_velo = np.array([-0.07, -0.05, -0.025, -0.01, 0.07, 0.05, 0.025, 0.01])
    starting_velo = starting_velo.round(4)
    # print(starting_velo)

    starting_poistion = np.transpose(
        [np.tile(starting_pos, len(starting_velo)), np.repeat(starting_velo, len(starting_pos))])
    # print(starting_poistion)
    return starting_poistion


def distances_from_center(state):
    starting_pos = generating_starting_positions()
    # current_state = np.array([1, 1])  # (p,v)
    # print(current_state)
    distances = state.T - starting_pos
    distances = distances.round(4)
    # print(xsis)

    return distances


def common_gaussion(state):
    diag = np.diag((pie_p, pie_v))
    distances = distances_from_center(state)
    # print(distances)
    theta_p_v = np.zeros((distances.shape[0]))
    for i in range(len(theta_p_v)):
        theta_p_v[i] = ((distances[i].T @ np.linalg.inv(diag) @ distances[
            i]) / -2)  ## high\low number because the inverse matrix of 0.0004 is 2500
        theta_p_v[i] = np.exp(theta_p_v[i])
    return theta_p_v


def Q(state, action, w):
    # print(w[action].shape)
    #print(state.shape)
    value = state.T @ (w[action])
    return value


# Epsilon greedy policy
def greedy_policy(state, w, epsilon=0.1):  # when evalute epsilon is zero
    A = np.ones(number_of_actions) * epsilon / number_of_actions
    best_action = np.argmax([Q(state, a, w) for a in range(number_of_actions)])
    A[best_action] += (1.0 - epsilon)
    sample = np.random.choice(number_of_actions, p=A)
    return sample


def run_simulation(w,with_render=False, with_print=False):
    #print("inside run simu")
    #print(w)
    state = env.reset()
    f_state = common_gaussion(state)
    step = 0
    total_reward = 0
    actions_to_str = ["al", "na", "ra"]
    step_to_print = []
    while True:
        step += 1

        if with_render:
            env.render()

        action = greedy_policy(f_state, w, epsilon=0.0)

        next_state, reward, done, _ = env.step(action)

        if with_print:
            step_to_print.append(
                str(step) + '. ' + str(state[0]) + ',' + str(state[1]) + ' ' + "0.5, 0, " + actions_to_str[
                    action] + " " + str(reward))
        f_next_state = common_gaussion(next_state)

        total_reward += reward

        if done:
            if with_print:
                print("total steps: " + str(step))
                print("total_reward: " + str(total_reward))
                for line in step_to_print:
                    print(line)
            break
        # update our state
        state = next_state
        f_state = f_next_state
    return total_reward


def eval_policy(w,num=10):
    total = 0
    for _ in range(num):
        val = run_simulation(w)
        total += val
    return total / num


def sarsa_learning():
    w = np.zeros((3, 32))
    evaluate = []
    best_w = None
    best_eval = -math.inf
    for episode in range(1, 50):  # range(num_of_episodes):
        total = 0
        print(episode)
        state = env.reset()
        state = common_gaussion(state)
        while True:

            # env.render()
            # Sample from our policy
            # print(state.shape)

            action = greedy_policy(state, w)
            # Staistic for graphing
            # plt_actions[action] += 1
            # Step environment and get next state and make it a feature
            next_state, reward, done, _ = env.step(action)
            next_state_2 = next_state
            next_state = common_gaussion(next_state)

            total += reward
            # Figure out what our policy tells us to do for the next state
            next_action = greedy_policy(next_state, w)

            # Statistic for graphing
            # episode_rewards[e] += reward

            # Figure out target and td error
            target = reward + gamma * Q(next_state, next_action, w)
            td_error = np.array(Q(state, action, w) - target)

            # Find gradient with code to check it commented below (check passes)
            dw = td_error.dot(state)

            # Update weight
            w[action] -= alpha * dw
            if (episode % 10 == 0):
                temp_eval = eval_policy(w)
                if temp_eval > best_eval:
                    best_eval = temp_eval
                    print(best_eval)
                    print(w)
                    best_w = w
                run_simulation(w,with_render=True, with_print=True)
            if done:
                print(next_state_2)
                break
            # update our state
            state = next_state_2
            state = common_gaussion(state)

    print(evaluate)
    return best_w


def main():
    best_w = sarsa_learning()
    #print("inside main")
    print(best_w)
    #run_simulation(best_w,with_render=True,with_print=True)

if __name__ == '__main__':
    main()
