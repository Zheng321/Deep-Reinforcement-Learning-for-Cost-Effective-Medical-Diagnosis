import os
import numpy as np
import pandas as pd
import torch as th

from data_loader import Data_Loader
from imputation import Imputer
from classifier import Classifier

from sklearn.metrics import roc_auc_score

import gym
from gym import Env
from gym.spaces import Discrete, Box

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback

#from util import maskable_evaluate_policy
#from util import MaskableEvalCallback

# from stable_baselines3 import PPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from tqdm import tqdm
from my_result_writer import MyResultWriter


class RL():
        def __init__(self, data_loader, imputer, classifier, cost, rl_para):
                '''
                rl_para:        'lr': 0.0003
                                        'n_steps':  2048
                                        'batch_size': 256
                                        'net_size': (64, 64)
                                        'penalty_ratio': 11.23
                                        'wrong_prediction_penalty': 100
                '''
                self.data_loader = data_loader

                self.train = self.data_loader.train_rl
                self.test = self.data_loader.test
                self.val = self.data_loader.val_rl

                self.train_size, self.total_dim = self.train.shape
                self.num_chosen_block = len(self.data_loader.block) - 1  # the first block is always observed
                # self.chosen_dim = sum([len(self.data_loader.block[i]) for i in self.data_loader.block])

                self.cost = cost
                # self.penalty_matrix = penalty_matrix
                self.para = rl_para

                ## train for #iterations steps on data
                self.lr = self.para['lr']
                self.n_steps = self.para['n_steps']
                self.batch_size = self.para['batch_size']
                self.net_size = self.para['net_size']
                self.penalty_ratio = self.para['penalty_ratio']
                self.wrong_prediction_penalty = self.para['wrong_prediction_penalty']
                # learning_rate, n_steps, batch_size, net_size, penalty_ratio, wrong_prediction_penalty = self.para  ## how to add penalty matrix?
                self.file_string = f'{self.lr}_{self.n_steps}_{self.batch_size}_{self.net_size}_{self.penalty_ratio}_{self.wrong_prediction_penalty}'
                # self.file_string = f'{self.para['lr']}_{self.para['n_steps']}_{self.para['batch_size']}_{self.para['net_size']}_{self.para['penalty_ratio']}_{self.para['wrong_prediction_penalty']}'

                self.env = self.env_val = self.env_test = None
                self.update(imputer, classifier)

                #self.val_callback = MaskableEvalCallback(self.env_val, best_model_save_path='./rl_model/best/logs_'+self.file_string + '/',
                                                                         #log_path='./rl_model/log/logs_'+self.file_string + '/', n_eval_episodes = 5, eval_freq = 5000,
                                                                         #deterministic = True, render = False, verbose = 0)

                policy_kwargs = dict(activation_fn=th.nn.SiLU, net_arch=[dict(pi=list(self.net_size), vf=list(self.net_size))])

                self.model = MaskablePPO(MaskableActorCriticPolicy, self.env, learning_rate = self.lr, n_steps = self.n_steps, batch_size = self.batch_size, policy_kwargs=policy_kwargs, verbose=0)
                self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

                # self.model_save()
                return

        def update(self, imputer, classifier):
                self.imputer = imputer
                self.classifier = classifier

                if self.env is not None:
                    self.env.close()
                if self.env_val is not None:
                    self.env_val.close()
                if self.env_test is not None:
                    self.env_test.close()

                log_dir_train = './rl_model/tmp/train_' + self.file_string + '/'
                os.makedirs(log_dir_train, exist_ok=True)
                log_dir_val = './rl_model/tmp/val_' + self.file_string + '/'
                os.makedirs(log_dir_val, exist_ok=True)

                self.env = dynamic_testing('train', self.data_loader, self.imputer, self.classifier, self.cost, self.penalty_ratio, self.wrong_prediction_penalty)
                self.env_val = dynamic_testing('val', self.data_loader, self.imputer, self.classifier, self.cost, self.penalty_ratio, self.wrong_prediction_penalty)
                self.env_test = dynamic_testing('test', self.data_loader, self.imputer, self.classifier, self.cost, self.penalty_ratio, self.wrong_prediction_penalty)


                #self.val_callback = MaskableEvalCallback(self.env_val, best_model_save_path='./rl_model/best/logs_'+self.file_string + '/',
                                                                         #log_path='./rl_model/log/logs_'+self.file_string + '/', n_eval_episodes = 5, eval_freq = 5000,
                                                                         #deterministic = True, render = False, verbose = 0)

                # policy_kwargs = dict(activation_fn=th.nn.SiLU, net_arch=[dict(pi=list(self.net_size), vf=list(self.net_size))])

                # self.model = MaskablePPO(MaskableActorCriticPolicy, self.env, learning_rate = self.lr, n_steps = self.n_steps, batch_size = self.batch_size, policy_kwargs=policy_kwargs, verbose=0)

                return

        def train_model(self, imputer, classifier, total_timesteps):
                # update the imputer and classifier
                self.update(imputer, classifier)
                self.model.set_env(self.env)
                # print(self.model.env)
                # print(self.model.env.reset())

                #self.model.learn(total_timesteps, callback = self.val_callback)
                self.model.learn(total_timesteps, callback = None)

                return

        def test_model_zero_start(self, data, max_episodes = None, generate_new_data = False):
                ## data: 'test' or 'val' or 'train'
                ## output: accuracy, F1, AUC
                if data == 'test':
                        env = self.env_test
                elif data == 'val':
                        env = self.env_val
                elif data == 'train':
                        env = self.env

                if not max_episodes:
                        max_episodes = len(env.patient_data)
                else:
                        max_episodes = min(max_episodes, len(env.patient_data))

                if generate_new_data:  # need to store both original data and mask
                        self.new_data_patient = np.empty((0, self.total_dim))
                        self.new_data_mask = np.empty((0, self.total_dim - 1))

                cumulative_reward = 0
                n_tested_ave = 0
                cost_tested_ave = 0
                n_healthy = 0
                n_ill = 0
                n_healthy_acc_predict = 0
                n_ill_acc_predict = 0
                predict_prob = np.zeros(max_episodes)
                true_label = np.zeros(max_episodes)
                action_taken = {}

                for episode in (range(max_episodes)):
                        obs = env.reset()

                        if env.patient_label == 0:
                                n_healthy += 1
                        else:
                                n_ill += 1

                        # env.patient = env.patient_data[index]
                        # env.patient_label = env.patient_data_label[index]
                        # mask out all blocks except for the first one
                        mask_blocks = list(range(1, env.n_panel + 1))
                        env.raw_state = env.patient.copy()
                        env.raw_state = self.data_loader.mask_out(env.raw_state, mask_blocks)
                        env.get_complete_state()

                        env.invalid_actions = []
                        env.length = env.n_panel + 1
                        obs = env.complete_state
                        done = False

                        if generate_new_data:  #  do we need to include completely empty test? yes
                                new_patient = np.append(env.patient, env.patient_label)
                                self.new_data_patient = np.vstack([self.new_data_patient, new_patient])
                                self.new_data_mask = np.vstack([self.new_data_mask, np.isnan(env.raw_state)])


                        true_label[episode] = env.patient_label

                        while not done:
                                action_masks = env.action_masks()
                                action, _ = self.model.predict(obs, deterministic = True, action_masks = action_masks)
                                obs, reward, done, info = env.step(action)
                                cumulative_reward += reward

                                if action < env.n_panel:
                                        n_tested_ave += 1
                                        cost_tested_ave += env.cost[action]
                                        if action in action_taken:
                                                action_taken[action] += 1
                                        else:
                                                action_taken[action] = 1

                                        if generate_new_data:
                                                new_patient = np.append(env.patient, env.patient_label)
                                                self.new_data_patient = np.vstack([self.new_data_patient, new_patient])
                                                self.new_data_mask = np.vstack([self.new_data_mask, np.isnan(env.raw_state)])
                                                # new_data = np.append(env.raw_state, env.patient_label)
                                                # self.output_new_data = np.vstack([self.output_new_data, new_data])

                                else: # assume binary classification
                                        obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
                                        action_list = th.tensor(list(range(env.n_panel + env.n_label))).to(self.device)
                                        prob = th.exp(self.model.policy.evaluate_actions(obs_tensor, action_list, action_masks)[1])[-env.n_label:]
                                        predict_prob[episode] = (prob[-2] / th.sum(prob)).item() #predict prob of ill patient, class 1

                                        if env.patient_label == 0:
                                                n_healthy_acc_predict += (action - env.n_panel == 1).item()
                                        else:
                                                n_ill_acc_predict += (action - env.n_panel == 0).item()



                # report statistics: accuracy, length, test distributions,
                self.n_episodes = max_episodes
                print('episodes: ', self.n_episodes)
                self.n_tested_ave = n_tested_ave / max_episodes
                self.cost_tested_ave = cost_tested_ave / max_episodes

                self.n_healthy = n_healthy
                print('n_healthy: ', self.n_healthy)
                self.n_ill = n_ill
                print('n_ill: ', self.n_ill)
                self.n_healthy_acc_predict = n_healthy_acc_predict
                print('n_healthy_acc_predict: ', self.n_healthy_acc_predict)
                self.n_ill_acc_predict = n_ill_acc_predict
                print('n_ill_acc_predict: ', self.n_ill_acc_predict)
                self.tested_distribution = {i:action_taken[i] / max_episodes for i in action_taken}
                print('tested_distribution: ', self.tested_distribution)
                self.predict_prob = predict_prob
                self.true_label = true_label
                self.cumulative_reward = cumulative_reward / max_episodes

                return self.cumulative_reward, self.n_tested_ave, self.cost_tested_ave

        def baseline_fixed_action(self, data, action_taken, max_episodes = None):
            ## baseline rl: with fixed test panel chosen for each patient
            ## action_taken should be a dic: key is panel 1,2,..,n_panel+1, value is the chosen prob (independently).
            if data == 'test':
                cur_env = self.env_test
            elif data == 'val':
                cur_env = self.env_val
            elif data == 'train':
                cur_env = self.env

            if not max_episodes:
                max_episodes = len(cur_env.patient_data)
            else:
                max_episodes = min(max_episodes, len(cur_env.patient_data))

            cumulative_reward = 0
            n_tested_ave = 0
            cost_tested_ave = 0
            n_healthy = 0
            n_ill = 0
            n_healthy_acc_predict = 0
            n_ill_acc_predict = 0
            predict_prob = np.zeros(max_episodes)
            true_label = np.zeros(max_episodes)
            action_taken = {}

            for episode in (range(max_episodes)):
                obs = cur_env.reset()

                if cur_env.patient_label == 0:
                    n_healthy += 1
                else:
                    n_ill += 1

                mask_blocks = [i for i in range(1, cur_env.n_panel + 1) if np.random.rand() > action_taken[i]]
                cur_env.raw_state = cur_env.patient.copy()
                cur_env.raw_state = self.data_loader.mask_out(cur_env.raw_state, mask_blocks)
                cur_env.get_complete_state()

                cur_env.invalid_actions = list(range(cur_env.n_panel))
                cur_env.length = 1
                obs = cur_env.complete_state
                done = False

                true_label[episode] = cur_env.patient_label
                n_tested_ave += cur_env.n_panel - len(mask_blocks)
                cost_tested_ave += cur_env.n_panel + 1
                for i in mask_blocks:
                    cost_tested_ave -= cur_env.cost[i]

                while not done:
                    action_masks = cur_env.action_masks()
                    action, _ = self.model.predict(obs, deterministic = True, action_masks = action_masks)
                    obs, reward, done, info = cur_env.step(action)
                    cumulative_reward += reward

                    if action < env.n_panel:
                        print('Error: should not pick more tasks')

                    else: # assume binary classification
                        obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
                        action_list = th.tensor(list(range(cur_env.n_panel + cur_env.n_label))).to(self.device)
                        prob = th.exp(self.model.policy.evaluate_actions(obs_tensor, action_list, action_masks)[1])[-cur_env.n_label:]
                        predict_prob[episode] = (prob[-2] / th.sum(prob)).item() #predict prob of ill patient, class 1

                        if cur_env.patient_label == 0:
                            n_healthy_acc_predict += (action - cur_env.n_panel == 1).item()
                        else:
                            n_ill_acc_predict += (action - cur_env.n_panel == 0).item()



            # report statistics: accuracy, length, test distributions,
            self.n_episodes = max_episodes
            print('episodes: ', self.n_episodes)
            self.n_tested_ave = n_tested_ave / max_episodes
            self.cost_tested_ave = cost_tested_ave / max_episodes

            self.n_healthy = n_healthy
            print('n_healthy: ', self.n_healthy)
            self.n_ill = n_ill
            print('n_ill: ', self.n_ill)
            self.n_healthy_acc_predict = n_healthy_acc_predict
            print('n_healthy_acc_predict: ', self.n_healthy_acc_predict)
            self.n_ill_acc_predict = n_ill_acc_predict
            print('n_ill_acc_predict: ', self.n_ill_acc_predict)
            self.tested_distribution = {i:action_taken[i] / max_episodes for i in action_taken}
            print('tested_distribution: ', self.tested_distribution)
            self.predict_prob = predict_prob
            self.true_label = true_label
            self.cumulative_reward = cumulative_reward / max_episodes

            return self.cumulative_reward, self.n_tested_ave, self.cost_tested_ave

        # def transform(self, data):
        #       ## using self.model to transform data
        #       return

        # def model_validate(self, data):
        #       ## evaluate the model on test data
        #       return

        def cal_stats(self):
                # accuracy for healthy
                acc_heal = self.n_healthy_acc_predict / self.n_healthy
                # accuracy for ill
                acc_ill = self.n_ill_acc_predict / self.n_ill
                # overall accuracy (non-weighted)
                acc = (self.n_healthy_acc_predict + self.n_ill_acc_predict) / (self.n_healthy + self.n_ill)
                bacc =  (acc_heal + acc_ill ) / 2
                # f1 score
                tp = self.n_ill_acc_predict
                fn = self.n_ill - self.n_ill_acc_predict
                fp = self.n_healthy - self.n_healthy_acc_predict
                f1 = tp / (tp + (fp + fn) / 2)
                # auroc
                auroc = roc_auc_score(self.true_label, self.predict_prob) # again assume binary, class 1 is the ill patient class

                return acc_heal, acc_ill, bacc, f1, auroc

        def model_save(self, file_name, save_dir):
                ## save imputation model
                if save_dir is None:
                    save_dir = './rl_model/'
                self.model.save(save_dir + '/' + file_name + '.pth')
                return

        def model_load(self, file_name, load_dir):
                ## load imputation model
                if load_dir is None:
                    load_dir = './rl+model'
                self.model = MaskablePPO.load(load_dir + '/' + file_name + '.pth')
                self.model.set_env(self.env)
                return

class dynamic_testing(Env):
        def __init__(self, train_test_val, data_loader, imputer, classifier, cost, penalty_ratio, wrong_prediction_penalty = 100): # assume binary classification
                self.results_writer = MyResultWriter()
                self.data_loader = data_loader
                self.imputer = imputer
                self.classifier = classifier

                self.blocks = self.data_loader.block
                self.n_panel = len(self.blocks) - 1 # first block is always observed

                if train_test_val == 'train':
                        self.patient_data = self.data_loader.train_rl[:, :-1]
                        self.patient_data_label = self.data_loader.train_rl[:, -1]
                elif train_test_val == 'test':
                        self.patient_data = self.data_loader.test[:, :-1]
                        self.patient_data_label = self.data_loader.test[:, -1]
                else:
                        self.patient_data = self.data_loader.val_rl[:, :-1]
                        self.patient_data_label = self.data_loader.val_rl[:, -1]

                self.dim = len(self.patient_data[0])
                self.n_patient = len(self.patient_data)

                unique, counts = np.unique(self.patient_data_label, return_counts=True)
                self.n_label = len(unique)
                # self.penalty_ratio = penalty_ratio

                # Actions we can take, choose test or predict label
                self.action_space = Discrete(self.n_panel + self.n_label)
#         self.action_space = self.action_a

                # State array
                # high = 3 * np.ones(self.n_test_total + self.n_test * k + self.n_label)
                # low = - 3 * np.ones(self.n_test_total + self.n_test * k + self.n_label)
                high = 5 * np.ones(self.n_label + self.dim * 2)
                low = - 5 * np.ones(self.n_label + self.dim * 2)
                self.observation_space = Box(np.float32(low), np.float32(high), dtype = np.float32)

                self.reset()

                # # Pick random patient from batch dataset
                # index = np.random.randint(self.n_patient)
                # self.patient = self.patient_data[index]
                # self.patient_label = self.patient_data_label[index]

                # # Set start state to be randomly selected observed test
                # n_obs_test = np.random.randint(self.n_panel + 1)
                # self.unobserved_panels = [i + 1 for i in range(self.n_panel) if np.random.rand() > n_obs_test / self.n_panel]
                # # raw state: unobserved --> np.nan
                # self.raw_state = self.data_loader.mask_out(self.patient, self.unobserved_panels)
                # # pass through the imputer
                # self.state = self.imputer.transform(self.raw_state)
                # # pass through the classifier
                # self.complete_state = np.append(self.classifier.predict(self.state), self.state)

                # # Set episode length
                # self.length = len(self.unobserved_panels) + 1

                # # Set invalid action (i.e., observed panels - 1)
                # self.invalid_actions = [i for i in range(self.n_panel) if i + 1 not in self.unobserved_panels]

                # Set reward
                self.cost = cost
                self.penalty_ratio = penalty_ratio
                self.wrong_prediction_penalty = wrong_prediction_penalty

                # dummy features
                self.output_new_data = None
                self.n_episodes = None
                self.n_tested_ave = None
                self.cost_tested_ave = None
                self.n_healthy = None
                self.n_ill = None
                self.n_healthy_acc_predict = None
                self.n_ill_acc_predict = None
                self.tested_distribution = None
                self.predict_prob = None
                self.true_label = None
                self.cumulative_reward = None

                return

        def action_masks(self):
                return [action not in self.invalid_actions for action in range(self.n_panel + self.n_label)]


        def get_complete_state(self):
            """
                get self.complete_state from self.raw_state
            """
            # pass through the imputer
            self.state = self.imputer.transform(self.raw_state)
            # pass through the classifier
            cls_result = self.classifier.predict(self.state).cpu()
            self.complete_state = np.append(cls_result,self.state)
            # add indicator of observed panels
            self.complete_state = np.append(self.complete_state, 1 - np.isnan(self.raw_state)) # final state embedding

        def step(self, action):
                # If choose to continue testing
                if action < self.n_panel:
                        self.invalid_actions += [action]
                        # update raw state
                        self.data_loader.mask_in(self.raw_state, self.patient, [action + 1])
                        self.get_complete_state()

                        reward = - self.cost[action]
                        done = False

                # If choose to predict:
                else:
                        if self.patient_label == 0: # healthy patient
                                reward = - self.wrong_prediction_penalty * (action == self.n_panel)   # prediction penalty of predicting wrongly
                        else: # ill patient -> penalize more
                                reward = - self.wrong_prediction_penalty * self.penalty_ratio * (action == self.n_panel + 1)
                        # reward = - self.penalty_matrix[action - self.n_panel, self.patient_label] * self.wrong_prediction_penalty
                        done = True

                info = {'imputed': self.state.cpu().numpy().copy(), 'label': self.patient_label}
                self.results_writer.write_row(info)

                self.length -= 1
                if self.length == 0:
                    done = True

                if self.length < 0:
                        print()
                        print('Error: episode length < 0')
                        print()

                # Return step information
                return self.complete_state, reward, done, info

        def render(self):
                pass

        def reset(self):
                # Pick random patient from batch dataset
                index = np.random.randint(self.n_patient)
                self.patient = self.patient_data[index]
                self.patient_label = self.patient_data_label[index]

                # Set start state to be randomly selected observed test
                n_obs_test = np.random.randint(self.n_panel + 1)
                self.unobserved_panels = [i + 1 for i in range(self.n_panel) if np.random.rand() > n_obs_test / self.n_panel]
                # raw state: unobserved --> np.nan
                self.raw_state = self.data_loader.mask_out(self.patient, self.unobserved_panels)
                self.get_complete_state()

                # Set episode length
                self.length = len(self.unobserved_panels) + 1

                # Set invalid action (i.e., observed panels - 1)
                self.invalid_actions = [i for i in range(self.n_panel) if i + 1 not in self.unobserved_panels]

                return self.complete_state

## test
