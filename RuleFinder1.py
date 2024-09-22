# A Sequence Memory Agent for a Delayed Match-to-Sample Task

import gymnasium as gym
import numpy as np

import sys
import argparse
import json
from SequenceMemory import OneHotDial
from SequenceMemory import SequenceMemory


class HypotheticalSequences:
    def __init__(self, input_dim, seq_mem_size):
        self.input_dim = input_dim
        self.seq_mem_size = seq_mem_size
        self.seq_mem = None  # SequenceMemory()
        self.ohd = OneHotDial(seq_mem_size)
        self.ohd2features = SequenceMemory(self.ohd, input_dim)
        self.current_state = self.ohd.make_afar_hot()  # init
        self.traced_states = []
        self.states_to_be_evaluated = []
        self.values = {}
        self.predictions = {}

    def proceed(self, observation):
        next_traced = []
        for next_state, prediction in self.predictions.items():
            if tuple(prediction) == tuple(observation):
                next_traced.append(next_state)
        if len(self.traced_states) == 0:
            value = 0
        elif len(next_traced) == 0:
            value = -1
            self.states_to_be_evaluated = self.traced_states
        else:
            value = 1
            self.states_to_be_evaluated = next_traced
        self.traced_states = next_traced
        self.set_predictions()
        self.current_state = self.ohd2features.tic()
        self.ohd2features.memorize_features(observation, self.current_state)
        return value

    def set_predictions(self):
        self.predictions = {}
        for state in self.traced_states:
            next_state = np.argmax(self.ohd2features.ohd.transition[state])
            prediction = self.ohd2features.retrieve_features(next_state)
            self.predictions[next_state] = prediction

    def init(self, observation):
        cells = observation @ self.ohd2features.dejavu
        self.traced_states = []
        for cell in range(len(cells)):
            if cells[cell] > 0 and self.values[cell] >= 0.0 and \
                    tuple(self.ohd2features.retrieve_features(cell)) == tuple(observation):
                self.traced_states.append(cell)
        self.set_predictions()
        self.current_state = self.ohd2features.tic()
        self.ohd2features.ohd.clear_reverse(self.current_state)
        self.values[self.current_state] = 0.0
        self.ohd2features.memorize_features(observation, self.current_state)

    def evaluate(self, value):
        for s in self.states_to_be_evaluated:
            state = s
            self.values[state] = value
            state = self.ohd2features.ohd.get_previous(state)
            while state >= 0:
                self.values[state] = value
                state = self.ohd2features.ohd.get_previous(state)


class Chooser:
    def __init__(self, dim):
        self.dim = dim
        self.rng = np.random.default_rng()  # random generator

    def step(self, distribution):
        distribution = np.array(distribution)
        if sum(distribution) <= 0:
            return np.array([0] * self.dim)
        else:
            if sum(distribution) > 0:
                distribution = distribution / sum(distribution)
            return np.array(self.rng.multinomial(1, distribution))


class Attention:
    def __init__(self, config, hs):
        self.attr_num = config['env']['attribute_number']   # + 1
        self.attribute_dim = config['env']['attribute_dim']
        self.discount = config['agent']['discount']
        self.hs = hs
        self.chooser = Chooser(self.attr_num)
        self.episode_init_state = -1

    def predictions2distribution(self):
        distribution = np.zeros(self.attr_num)
        for state, prediction in self.hs.predictions.items():
            if state in self.hs.values and self.hs.values[state] > 0:
                discount = 1.0
            else:
                discount = self.discount
            distribution = [prediction[x*self.attribute_dim:x*self.attribute_dim+self.attribute_dim].max() * discount
                            for x in range(self.attr_num)] + distribution
        if distribution.sum() == 0:
            distribution = np.ones(distribution.size) / distribution.size
        else:
            distribution = distribution / distribution.sum()
        return distribution

    def distribution(self, saliency):
        if np.count_nonzero(saliency) > 1:
            distribution = self.predictions2distribution()
            salient_distribution = distribution * saliency
            if salient_distribution.sum() > 0:
                salient_distribution = salient_distribution / salient_distribution.sum()
        else:
            salient_distribution = saliency
        return salient_distribution

    def attention(self, inputs):
        saliency = [inputs[x*self.attribute_dim:x*self.attribute_dim+self.attribute_dim].max()
                    for x in range(self.attr_num)]
        distribution = self.distribution(saliency)
        attention = self.chooser.step(distribution)
        return attention


class RuleFinder:
    def __init__(self, config):
        self.attribute_dim = config['env']['attribute_dim']
        self.attribute_number = config['env']['attribute_number']
        self.input_dim = self.attribute_number * self.attribute_dim
        self.hs = HypotheticalSequences(self.input_dim, config['agent']['seq_mem_size'])
        self.dump = config['dump']
        self.dump_level = config['dump_level']
        self.attention = Attention(config, self.hs)
        self.prev_observation = np.zeros(self.input_dim)
        self.reward = 0
        self.init = True

    def step(self, observation, done):
        attention = self.attention.attention(observation)
        gated_observation = np.zeros(observation.size)
        self.reward = 0
        for i in range(observation.size):
            gated_observation[i] = observation[i] * attention[i//self.attribute_dim]    # mask
        if gated_observation.max() > 0:
            if self.init:
                self.hs.init(gated_observation)
                self.init = False
                if self.dump_level >= 2:
                    print(observation, gated_observation, self.hs.current_state, self.hs.traced_states)
            elif tuple(observation) != tuple(self.prev_observation):
                self.reward = self.hs.proceed(gated_observation)
                if self.dump_level >= 2:
                    print(observation, gated_observation, self.hs.current_state, self.hs.traced_states)
        if done:
            self.hs.evaluate(self.reward)
            if self.reward != 0:
                self.hs.ohd2features.erase_traces(self.hs.current_state)
        self.prev_observation = observation

    def reset(self):
        self.prev_observation = np.zeros(self.input_dim)
        self.init = True


def main():
    parser = argparse.ArgumentParser(description='Rule Finder #1')
    parser.add_argument('--dump', help='dump file path')
    parser.add_argument('--episode_count', type=int, default=5, metavar='N',
                        help='Number of training episodes (default: 5)')
    parser.add_argument('--episode_sets', type=int, default=1, metavar='N',
                        help='Number of episode sets (default: 1)')
    parser.add_argument('--eval_period', type=int, default=5, metavar='N',
                        help='Evaluation span (default: 5)')
    parser.add_argument('--config', type=str, default='RuleFinder1.json', metavar='N',
                        help='Model configuration (default: RuleFinder1.json')
    parser.add_argument('--dump_level', type=int, default=0, help='>=0')
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    if args.dump is not None:
        try:
            dump = open(args.dump, mode='w')
        except IOError:
            print('Dump path error', file=sys.stderr)
            sys.exit(1)
    else:
        dump = None

    counts = {"episode_count": args.episode_count, "episode_sets": args.episode_sets}
    config['dump'] = dump
    config['dump_level'] = args.dump_level

    env = gym.make(config['env']['name'], config=config['env'])

    reward_zeros = np.zeros((args.episode_sets, args.episode_count // args.eval_period))
    reward_negs = np.zeros((args.episode_sets, args.episode_count // args.eval_period))
    reward_ones = np.zeros((args.episode_sets, args.episode_count // args.eval_period))
    for j in range(counts["episode_sets"]):
        rf = RuleFinder(config)
        reward_sum = 0
        reward_zero = 0
        reward_neg = 0
        reward_one = 0
        for i in range(counts["episode_count"]):
            if args.dump_level >= 2:
                print('--')
            env.reset()
            action = np.random.randint(1, 3)
            while True:
                obs, reward, terminated, truncated, info = env.step(action)
                rf.step(obs, terminated)
                # print(obs)
                if terminated:
                    rf.reset()
                    reward_sum += rf.reward
                    if rf.reward == 0:
                        reward_zero += 1
                    elif rf.reward < 0:
                        reward_neg += 1
                    else:
                        reward_one += 1
                    if args.dump_level >= 2:
                        print(j, i, ":", rf.reward)
                    break
            if i % args.eval_period == args.eval_period - 1:
                reward_zeros[j, i // args.eval_period] = reward_zero / args.eval_period
                reward_negs[j, i // args.eval_period] = reward_neg / args.eval_period
                reward_ones[j, i // args.eval_period] = reward_one / args.eval_period
                reward_zero = 0
                reward_neg = 0
                reward_one = 0
    env.close()
    for i in range(args.episode_count // args.eval_period):
        for j in range(counts["episode_sets"]):
            dump.write('{0};{1};{2}\t'.format(reward_zeros[j, i], reward_negs[j, i], reward_ones[j, i]))
        dump.write('\n')


if __name__ == '__main__':
    main()
