import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Data_Loader():
        def __init__(self, data, block, test_ratio = 0.2, val_ratio = 0.2):
                '''
                data: [X,y]
                block: dic: {0: array([1,2,3]), 1:array([0,4,5])}
                NOTE: block[0] always indicates non-missing tests, such as age, ethic group etc.

                !!! Important: split the dataset into four parts:
                part 1: (train: 25%, test: 5%) for train imputer
                part 2: (train: 25%, test: 5%) for train classifier
                part 3: (train 25%, test: 5%) for train rl (warmup)
                part 4: test 10%, final evaluation here (no training or hyperpara tuning dependent)
                '''
                self.data = data
                self.block = block

                # self.train_test_val_split(test_ratio, val_ratio)
                self.train_test_split_onto_4()

        def train_test_val_split(self, test_ratio, val_ratio):
                '''
                '''
                self.train, self.test = train_test_split(self.data, test_size=test_ratio, random_state = 21)

                self.train, self.val = train_test_split(self.train, test_size=val_ratio, random_state = 12)

        def train_test_split_onto_4(self):
                '''
                part 1: (train: 25%, test: 5%) for train imputer
                part 2: (train: 25%, test: 5%) for train classifier
                part 3: (train 25%, test: 5%) for train rl (warmup)
                part 4: test 10%, final evaluation here (no training or hyperpara tuning dependent)
                '''
                self.train, self.test = train_test_split(self.data, test_size=0.1, random_state = 7)  # self.test: 10%

                self.train, self.train_rl = train_test_split(self.train, test_size=1/3, random_state = 14) 
                self.train_rl, self.val_rl = train_test_split(self.train_rl, test_size=1/6, random_state = 21) # train for rl: 25%, val for rl: 5%

                self.train_imp, self.train_clf = train_test_split(self.train, test_size=0.5, random_state = 28)

                self.train_imp, self.val_imp =  train_test_split(self.train_imp, test_size=1/6, random_state = 35) # train for imp: 25%, val for imp: 5%
                self.train_clf, self.val_clf =  train_test_split(self.train_clf, test_size=1/6, random_state = 42) # train for imp: 25%, val for imp: 5%

                del self.train # to avoid messing up with previous version.

                self.train = np.vstack((self.train_imp, np.vstack((self.train_clf, self.train_rl))))
                self.val = np.vstack((self.val_imp, np.vstack((self.val_clf, self.val_rl))))

                # with open('data/complete_clf_blood_panel_data.npy', 'wb') as f:
                #     np.save(f, self.train)
                #     np.save(f, self.val)
                #     np.save(f, self.test)


        def mask_out(self, data, blocks):
                '''
                transform the blocks to np.nan
                '''
                output = data.copy()
                for i in blocks:
                        if i == 0:
                                print('error: do not mask out first block')
                        output[self.block[i]] = np.nan
                return output

        def mask_in(self, data, real_data, blocks):
                '''
                reveal the blocks
                '''
                for i in blocks:
                        if i == 0:
                                print('error: do not mask in first block')
                        data[self.block[i]] = real_data[self.block[i]]
                return

        def random_augment_with_rate(self, data, start, end, M = 1):
                '''
                '''
                np.random.seed(21)
                N = len(data)
                m = len(self.block) - 1 # first block is never missing

                size = (end - start + 1) * N * M
                complete_data = np.zeros((size, len(data[0])))
                mask = np.zeros((size, len(data[0])))
                missing_data = np.zeros((size, len(data[0])))

                count = 0

                for i in range(N):
                        for n_obs in range(start, end + 1):
                                for _ in range(M):
                                        # unobserved test panels
                                        complete_data[count] = data[i, :]
                                        unobserved_blocks = np.array([k + 1 for k in range(m) if np.random.rand() < n_obs / m])
                                        missing_data[count] = self.mask_out(data[i,:], unobserved_blocks)
                                        mask[count] = np.isnan(missing_data[count])
                                        count += 1

                print('missing rate is:', np.sum(mask) / mask.size)

                return complete_data, mask[:, :-1], missing_data  # mask does not contain label information


        def random_augment(self, data, M = 1):
                '''
                '''
                N = len(data)
                m = len(self.block) - 1 # first block is never missing

                size = (m + 1) * N * M
                complete_data = np.zeros((size, len(data[0])))
                mask = np.zeros((size, len(data[0])))
                missing_data = np.zeros((size, len(data[0])))

                count = 0

                for i in range(N):
                        for n_obs in range(m + 1):
                                for _ in range(M):
                                        # unobserved test panels
                                        complete_data[count] = data[i, :]
                                        unobserved_blocks = np.array([k + 1 for k in range(m) if np.random.rand() < n_obs / m])
                                        missing_data[count] = self.mask_out(data[i,:], unobserved_blocks)
                                        mask[count] = np.isnan(missing_data[count])
                                        count += 1

                return complete_data, mask[:, :-1], missing_data  # mask does not contain label information

        def biased_augment(self, data, M = 1):
                '''
                '''
                N = len(data)
                m = len(self.block) - 1 # first block is never missing

                size = (m + 1 - m // 2) * N * M
                complete_data = np.zeros((size, len(data[0])))
                mask = np.zeros((size, len(data[0])))
                missing_data = np.zeros((size, len(data[0])))

                count = 0

                for i in range(N):
                        for n_obs in range(m // 2, m + 1):
                                for _ in range(M):
                                        # unobserved test panels
                                        complete_data[count] = data[i, :]
                                        # unobserved_blocks = np.array([k + 1 for k in range(m) if np.random.rand() < n_obs / m])

                                        unobserved_blocks = np.random.choice(m, n_obs, p=[10/51, 10/51, 10/51, 10/51, 1/51, 10/51]) + 1

                                        missing_data[count] = self.mask_out(data[i,:], unobserved_blocks)
                                        mask[count] = np.isnan(missing_data[count])
                                        count += 1

                return complete_data, mask[:, :-1], missing_data  # mask does not contain label information

# # ## test

# data = np.random.rand(20, 4)
# block = {0: np.array([0]), 1: np.array([1,2]), 2: np.array([3])}

# mydata = Data_Loader(data, block)

# print(mydata.random_augment(mydata.train_imp, 1))

# print()
# d = mydata.train_imp[0]
# print(mydata.mask_out(d, [1, 2]))
# print(mydata.mask_out(d, []))
# f = mydata.mask_out(d, [1, 2])
# print(mydata.mask_in(f, d, [2]))
