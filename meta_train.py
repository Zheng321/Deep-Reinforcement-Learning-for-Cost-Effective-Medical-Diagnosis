from train import main
import sys
import time
from itertools import product

impute_para = {'batch_size': 256,
                                'lr': 1e-4, # use 0.1x lr for finetuning
                                'alpha': 1e6}

clf_para = {'hidden_size': 64,
                        'batch_size': 256,
                        'lr': 1e-5, # use 0.1x lr for finetuning
            'class_weights': [1.0, 3.0],
                        }

rl_para = {'lr': 0.0001,
                        'n_steps':  1024,
                        'batch_size': 128,
                        'net_size': (128, 128),
                        'penalty_ratio': 5,
                        'wrong_prediction_penalty': 100}

training_para = {'imputer_warmup_episodes': -1,
                                'classifier_warmup_episodes': -1,
                                'rl_warmup_timesteps': 500000, # train RL warmup first.
                                'new_data_size': 1000, # set to None when doing end-to-end training
                                'n_outer_loop': 1,
                                'rl_timesteps_per_loop': -1,
                                'imputer_episodes_per_loop': -1,
                                'classifier_episodes_per_loop': -1}
save_dir='exp_end_to_end_final'

### set hyperparameters

index = int(sys.argv[1])
p = list(product( list(range(1,11)), [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]))
print("total experiment number:", len(p))
hparam = p[index]
rl_para['penalty_ratio'], rl_para['wrong_prediction_penalty'] = hparam

save_dir = save_dir + '/' + str(index)

args = (impute_para, clf_para, rl_para, training_para, save_dir)
print(args)

### execute and get the running time.

start = time.time()
main(args)
end = time.time()
print('time', end-start)
