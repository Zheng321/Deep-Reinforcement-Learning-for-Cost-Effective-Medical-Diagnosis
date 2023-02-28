import os
import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm
import csv

from data_loader import Data_Loader
from imputation import Imputer
from classifier import Classifier
from rl import RL
from blood_panel_data_preprocessing import blood_panel_data

from torch.utils.data import Dataset, DataLoader
from classifier import clf_data
from tensorboardX import SummaryWriter

DIM = 53
NUM_CLASS = 2

def main(args):
        '''
        impute_para = {'batch_size': 256,
                                        'lr': 1e-4,
                                        'alpha': 1e6}

        clf_para: {'hidden_size': 64,
                                'lr': 0.005}

        rl_para:  {'lr': 0.0003
                                'n_steps':  2048
                                'batch_size': 256
                                'net_size': (64, 64)
                                'penalty_ratio': 10
                                'wrong_prediction_penalty': 100}

        training_para: {'imputer_warmup_episodes': 10,
                                        'classifier_warmup_episodes': 10,
                                        'n_outer_loop': 10,
                                        'rl_timesteps_per_loop': 100000,
                                        'imputer_episodes_per_loop': 10,
                                        'classifier_episodes_per_loop': 10}
        '''
        imputer_para, clf_para, rl_para, training_para, save_dir = args
        clf_para['save_dir'] = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_dir + '/' + 'args.txt', 'w') as f:
            f.write(str(args))
        writer = SummaryWriter(save_dir + '/' + 'end-to-end')

        # import data
        data, block, cost = blood_panel_data()

        # build data_loader
        data_loader = Data_Loader(data, block, test_ratio = 0.2, val_ratio = 0.2)

        # build imputer & classifer
        imputer = Imputer(DIM, imputer_para)
        classifier = Classifier(DIM, NUM_CLASS, clf_para)

        if False: #TODO
            ## warm-up phase
            # train imputer on uniform at random augmented training data
            augment_train, augment_train_mask, augment_train_missing = data_loader.random_augment(data_loader.train, M = 1)
            augment_val, augment_val_mask, augment_val_missing = data_loader.random_augment(data_loader.val, M = 1)
            augment_test, augment_test_mask, augment_test_missing = data_loader.random_augment(data_loader.test, M = 1)

            imputer.set_dataset(augment_train[:, :-1], augment_train_mask) # imputer takes training data without label information
            imputer.train_model(data = None, max_iter = training_para['imputer_warmup_episodes'])
            imputer.model_save('warmup')

            print('imputation warmup training done!')

            # train classsifier on imputed, uniform at random augmented training data
            imputed_augment_train = np.concatenate(( imputer.transform(), augment_train[:, -1:]), axis = 1)
            imputed_augment_val = np.concatenate(( imputer.transform(augment_val[:, :-1]), augment_val[:, -1:]), axis = 1)
            imputed_augment_test = np.concatenate(( imputer.transform(augment_test[:, :-1]), augment_test[:, -1:]), axis = 1)
            classifier.set_dataset(imputed_augment_train, imputed_augment_val, imputed_augment_test)
            classifier.train_model(classifier.train_dl, training_para['classifier_warmup_episodes'], verbose = 0)
            classifier.model_save('warmup')

            print('classifier warmup training done!')
        else:
            print('loading exisiting warmup models.')
            imputer.model_load('best_split_warmup')
            classifier.model_load('best_warmup')

        rl = RL(data_loader, imputer, classifier, cost, rl_para)
        # rl.model_save('')

        print('rl initialization done!')

        ## end-to-end training phase
        for outer_iter in (range(training_para['n_outer_loop'])):
                # train rl
                if outer_iter == 0:
                        timesteps = training_para['rl_warmup_timesteps']
                else:
                        timesteps = training_para['rl_timesteps_per_loop']

                print('rl training ' + str(outer_iter) + ' start!')

                rl.train_model(imputer, classifier, timesteps)
                # rl.model_save('')

                print('rl training ' + str(outer_iter) + ' done!')

                not_final_round = (outer_iter < (training_para['n_outer_loop']-1 ))
                print("not final round: ", not_final_round)
                # output new training data for imputer and classifier
                cumulative_reward, n_tested_ave, cost_tested_ave = rl.test_model_zero_start('train', max_episodes = training_para['new_data_size'], generate_new_data = not_final_round)
                print('Outer loop ' + str(outer_iter) + '(reward, n_test, cost_test):', (cumulative_reward, n_tested_ave, cost_tested_ave))

                acc_heal, acc_ill, acc, f1, auroc = rl.cal_stats()
                print('Outer loop ' + str(outer_iter) + '(acc_heal, acc_ill, bacc, f1, auroc):', (acc_heal, acc_ill, acc, f1, auroc ))

                if not_final_round: # if this is not the final round, then train imputer & classifier with the new data generated by RL.
                    # continue training imputer on new data output by rl
                    imputer.train_model(rl.new_data_patient[:, :-1], rl.new_data_mask, max_iter = training_para['imputer_episodes_per_loop'])

                    # continue training classifier on new data output by rl
                    data_X = rl.new_data_patient[:, :-1]
                    data_X[rl.new_data_mask.astype(np.bool_)] = np.nan # mask
                    new_data = np.concatenate(( imputer.transform(data_X), rl.new_data_patient[:, -1:]), axis = 1)
                    # print(new_data[:10, :])
                    # data = DataLoader(clf_data(np.concatenate(( imputer.transform(rl.output_new_data[:, :-1]), rl.output_new_data[:, -1:]), axis = 1)))
                    new_data_dl= DataLoader(clf_data(new_data),batch_size = clf_para['batch_size'], shuffle=True, drop_last=True)
                    classifier.train_model(new_data_dl, training_para['classifier_episodes_per_loop'], verbose = 0, fresh=False)

        imputer.model_save('final_imp',save_dir = save_dir)
        classifier.model_save('final_cls', save_dir = save_dir)
        rl.model_save('final_rl', save_dir = save_dir)

        # rl.env.reset()
        # print(rl.env.complete_state)
        # for _ in range(3):
        #       mask_blocks = list(range(1, rl.env.n_panel + 1))
        #       rl.env.raw_state = rl.env.patient.copy()
        #       rl.env.raw_state = data_loader.mask_out(rl.env.raw_state, mask_blocks)
        #       # rl.env.raw_state = np.full(rl.env.dim, np.nan)
        #       # pass through the imputer
        #       rl.env.state = imputer.transform(rl.env.raw_state)
        #       # pass through the classifier
        #       rl.env.complete_state = np.append(rl.env.classifier.predict(rl.env.state), rl.env.state)

        #       rl.env.reset()
        #       print(rl.env.raw_state, rl.env.state, rl.env.complete_state)

        ## testing out the final model
        # rl = RL(data_loader, imputer, classifier, cost, rl_para)
        # rl.update(imputer, classifier)

        # evaluate on val set
        cumulative_reward, n_ave_tested, cost_tested_ave = rl.test_model_zero_start('val', max_episodes = None, generate_new_data = False)
        print('Final on val set (reward, n_test, cost_test): ', (cumulative_reward, n_ave_tested, cost_tested_ave))
        acc_heal, acc_ill, acc, f1, auroc = rl.cal_stats()
        print('Final on val set (acc_heal, acc_ill, bacc, f1, auroc): ', (acc_heal, acc_ill, acc, f1, auroc ))

        # evaluate on test set
        test_cumulative_reward, test_n_tested_ave, test_cost_tested_ave = rl.test_model_zero_start('test', max_episodes = None, generate_new_data = False)
        print('Final on test set (reward, n_test, cost_test):', (test_cumulative_reward, test_n_tested_ave, test_cost_tested_ave))
        test_acc_heal, test_acc_ill, test_acc, test_f1, test_auroc = rl.cal_stats()
        print('Final on test set (acc_heal, acc_ill, bacc, f1, auroc): ', (test_acc_heal, test_acc_ill, test_acc, test_f1, test_auroc))

        # save to tensorboard.
        hparam =  rl_para.copy()
        hparam2 = training_para.copy()
        del hparam2['imputer_warmup_episodes']
        del hparam2['classifier_warmup_episodes']
        hparam.update(hparam2)
        hparam['net_size'] = str(hparam['net_size'])
        metric = {
            'bacc': acc,
            'f1': f1,
            'auroc': auroc,
            'reward': cumulative_reward,
            'n_ave_tested': n_ave_tested,
            'cost_tested_ave': cost_tested_ave,
            'test_bacc': test_acc,
            'test_f1': test_f1,
            'test_auroc': test_auroc,
            'test_reward': test_cumulative_reward,
            'test_n_ave_tested': test_n_tested_ave,
            'test_cost_tested_ave': test_cost_tested_ave
        }
        writer.add_hparams(hparam, metric, name='hparam_all')
        writer.close()



        return









