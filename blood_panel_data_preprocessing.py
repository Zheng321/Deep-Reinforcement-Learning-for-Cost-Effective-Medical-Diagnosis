import numpy as np
import pandas as pd
import math
from sklearn import preprocessing

def blood_panel_data():
        '''
        data: [X,y]
        block: dic: {0: array([1,2,3]), 1:array([0,4,5])}
        NOTE: block[0] always indicates non-missing tests, such as age, ethic group etc.
        NOTE: blocks can be overlapping
        '''
        df = pd.read_csv("data/log_Imputed.csv", index_col = [0])

        y= (df['Ferritin'] < math.log(11)).to_numpy().astype(np.int64)

        df = pd.concat([pd.get_dummies(df['race_1'], prefix='race_1'),df], axis = 1)
        df.drop(['race_1'], axis = 1, inplace = True)
        df = pd.concat([pd.get_dummies(df['ethnic_group'], prefix='ethnic_group'), df], axis = 1)
        df.drop(['ethnic_group'], axis = 1, inplace = True)
        df.drop(['Ferritin'], axis = 1, inplace = True)
        X = df.to_numpy().astype(np.float32)

        # standardize the data

        scaler = preprocessing.StandardScaler().fit(X[:, 13:])  # the first 13 columns are 0,1 indicators always observed.
        X[:, 13:] = scaler.transform(X[:, 13:])

        X[X > 5] = 5    # get rid of the outlier, to prevent the imputation to be numerically instable (only 0.1% of data are outside [-5,5])
        X[X < -5] = -5

        data = np.concatenate((X,y.reshape(len(y), 1)), axis = 1) # final data = [X ,y]

        block = {}
        cost = []
        block[0] = np.array(list(range(14))) # ethnic group, race, age

        block[1] = np.array([23, 26, 27, 28, 29, 30, 48, 45, 32]) # BMPC panel
        cost.append(36)

        block[2] = np.array([20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 48, 49, 51, 45, 32, 31]) # CMPC panel
        cost.append(48)

        block[3] = np.array([14, 52, 46, 44, 38, 36, 37, 47, 33, 34]) # AXCBC panel
        cost.append(26)

        block[4] = np.array([14, 15, 16, 17, 18, 19, 39, 40, 41, 42, 43, 52, 46, 44, 38, 36, 37, 47, 33, 34]) # CBCD panel
        cost.append(44)

        block[5] = np.array([35, 50]) # TSAT panel: Iron, TIBC
        cost.append(40)

        block[6] = np.array([25]) # B-12 panel
        cost.append(66)

        cost = np.array(cost)

        cost = cost / np.sum(cost) * len(cost) # average cost is 1

        return data, block, cost

def aki_data():
        '''
        index: DataFrame, will be used for train_test_split
        data: [X,y]
        block: dic: {0: array([1,2,3]), 1:array([0,4,5])}
        NOTE: block[0] always indicates non-missing tests, such as age, ethic group etc.
        NOTE: blocks can not be overlapping for this case
        '''
        df = pd.read_csv("/home/ylo7832/AKI_features.csv", index_col=0)

        y = df.aki3.to_numpy().astype(np.int64)
        del df['aki3']

        index = df[['subject_id','hadm_id','icustay_id']] # will be used for train/test split
        for c in ['subject_id','hadm_id','icustay_id']: del df[c]

        binary_columns = df.columns[df.isin([0,1]).all()].tolist() #no scaler

        X_noscaler = df[binary_columns].to_numpy().astype(np.float32)
        X_scaler = df[[c for c in df.columns if c not in binary_columns]].to_numpy().astype(np.float32)

        # standardize and clip
        X_scaler = preprocessing.StandardScaler().fit_transform(X_scaler)
        X_scaler = np.clip(X_scaler, -5, 5)

        data = np.concatenate((X_noscaler,X_scaler, y.reshape(len(y), 1)), axis = 1)

        block = {}
        cost = [44,48,473,26]

        block[0] = np.array(list(range(15))) # will never be masked
        block[1] = np.array([15,16,17]) #CBC
        block[2] = np.array([18,19,20,21,22,23,24,25]) #CMP
        block[3] = np.array([26,27,28,29,30,31]) #APTT
        block[4] = np.array([32,33]) #ABG

        cost = np.array(cost)
        cost = cost / np.sum(cost) * len(cost) # average cost is 1

        return index, data, block, cost

def sepsis_data():
        '''
        data: [X,y]
        block: dic: {0: array([1,2,3]), 1:array([0,4,5])}
        NOTE: block[0] always indicates non-missing tests, such as age, ethic group etc.
        NOTE: blocks can not be overlapping for this case
        '''
        df = pd.read_csv("/share/fsmresfiles/Sepsis/sepsis_features.csv", index_col=0)

        y = df.hospital_expire_flag.to_numpy().astype(np.int64)
        del df['hospital_expire_flag']

        index = df[['hadm_id','icustay_id']] # will be used for train/test split
        for c in ['hadm_id','icustay_id']: del df[c]

        binary_columns = df.columns[df.isin([0,1]).all()].tolist() #no scaler

        X_noscaler = df[binary_columns].to_numpy().astype(np.float32)
        X_scaler = df[[c for c in df.columns if c not in binary_columns]].to_numpy().astype(np.float32)

        # standardize and clip
        X_scaler = preprocessing.StandardScaler().fit_transform(X_scaler)
        X_scaler = np.clip(X_scaler, -5, 5)

        data = np.concatenate((X_noscaler,X_scaler, y.reshape(len(y), 1)), axis = 1)

        block = {}
        cost = [44,48,473,26,0]

        block[0] = np.array([4,0,1,2,5,3,8,9,10,11,23,25,27,39,41,42,43,44]) # will never be masked, 18 in total
        block[1] = np.array([7,28,29,38,45]) #CBC, N=5
        block[2] = np.array([12,14,15,16,17,18,19,20,21,22,26,31,32,36,46]) #CMP, N=15
        block[3] = np.array([30,37]) #APTT, N=2
        block[4] = np.array([13,24,33,34,35,40]) #ABG, N=6
        block[5] = np.array([6]) #sofa

        cost = np.array(cost)
        cost = cost / np.sum(cost) * len(cost) # average cost is 1
        
        return data, block, cost

## test
# blood_panel_data()
