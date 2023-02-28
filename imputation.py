import numpy as np
import pandas as pd
import torch
import flow_models
import nflow as nf
from sklearn import model_selection
from sklearn import preprocessing

import time
from tqdm import tqdm
import matplotlib.pyplot as plt


class Imputer():  # mean_imputer
        def __init__(self, dim, impute_para):
                self.d = dim
                self.batch_size = impute_para['batch_size']
                self.lr = impute_para['lr']
                self.alpha = impute_para['alpha']

        def set_dataset(self, train, mask):
                '''
                train: complete data with no label information
                mask: 0-1 indicator of same shape, 1 means missing, 0 means observed

                impute_para = {'batch_size': 256,
                #                          'lr': 1e-4,
                #                          'alpha': 1e6}
                '''
                self.X = torch.tensor(train).float()
                self.mask = torch.tensor(mask).int()

                # Generate mask and X_hat
                assert self.d == self.X.shape[1], 'wrong dim'
                self.X_hat = ((1 - self.mask) * self.X).float()

                mu_hat = torch.mean(self.X, dim=0).float()  # 1st change: initialize it to true mean and cov
                Sigma_hat = torch.cov(self.X.T).float()

                L, V = torch.linalg.eig(Sigma_hat)  # ensure positive definite
                if torch.min(L.real) < 1e-6:
                        print('Not PD, minimum eig is: ', torch.min(L.real))
                        Sigma_hat = torch.add(Sigma_hat, torch.eye(self.d) * (1e-6 - torch.min(L.real)))

                self.init_model()
                # Set prior and q0
                q0 = torch.distributions.multivariate_normal.MultivariateNormal(mu_hat, covariance_matrix=Sigma_hat)
                self.nfm.q0 = q0
                # self.train_model(train)

                return

        def init_model(self):
                # Construct flow model
                num_flows = 32
                torch.manual_seed(0)

                flows = []
                for i in range(num_flows):
                        if self.d % 2 == 1:
                                param_map = flow_models.MLP([self.d//2+1, 32, 32, self.d], init_zeros=True)
                        else:
                                param_map = flow_models.MLP([self.d//2, 32, 32, self.d], init_zeros=True)
                        flows.append( nf.AffineCouplingBlock(param_map) )
                        flows.append( nf.Permute(self.d, mode='swap') )
                        # flows.append( nf.flows.BatchNorm() )

                self.nfm = nf.NormalizingFlow(q0=None, flows=flows)

        def generate_hat_mask(self, data, mask):  # todo
                if data is None:
                        return self.X, self.X_hat, self.mask

                if data.ndim == 1:
                        data = data.reshape((1, len(data)))
                        mask = mask.reshape((1, len(data)))

                X = torch.tensor(data).float()
                mask = torch.tensor(mask).int()
                X_hat = ((1 - mask) * X).float()

                return X, X_hat, mask

        def train_model(self, data = None, mask = None, max_iter = 10):
                ## train for #iterations steps on data
                batch_size = self.batch_size
                lr = self.lr
                alpha = self.alpha
                rho = 0.99 * np.arange(1, max_iter+1)**(-0.8) # can be one of the tuning, but assume robust
                beta = 1e-3 / np.arange(1, max_iter+1)

                d = self.d
                nfm = self.nfm
                mu_hat = nfm.q0.loc
                Sigma_hat = nfm.q0.covariance_matrix

                X_true_train, X_hat_train, mask_train = self.generate_hat_mask(data, mask)

                optimizer = torch.optim.Adam(nfm.parameters(), lr=lr, weight_decay=1e-6)

                for J in tqdm(range(max_iter)):
                        nfm.zero_grad()

                        batch_ind = np.random.choice( X_hat_train.shape[0], size=batch_size ,replace=False )
                        x_hat = X_hat_train[batch_ind, :]
                        x_true = X_true_train[batch_ind, :]
                        mask_batch = mask_train[batch_ind, :]
                        # x_hat, mask_batch = x_hat.cuda(), mask_batch.cuda()

                        # Update flow model
                        log_prob = nfm.log_prob(x_hat)
                        L1 = -torch.mean(log_prob)
                        L1.backward()

                        optimizer.step()
                        optimizer.zero_grad() # Not sure if I need this here

                        # Update base distribution
                        z = nfm.inverse(x_hat)

                        Sigma_mask_mo = mask_batch.view(batch_size, d, 1) @ (1-mask_batch).view(batch_size, 1, d)
                        Sigma_mask_oo = (1-mask_batch).view(batch_size, d, 1) @ (1-mask_batch).view(batch_size, 1, d)
                        inv_Sigma_oo = []
                        for i in range(batch_size):
                                inv_Sigma_oo.append( self.inverse_masked( Sigma_hat, Sigma_mask_oo[i,:,:] ).float().unsqueeze(0) )
                        inv_Sigma_oo = torch.cat(inv_Sigma_oo, dim=0)
                        z_m = mu_hat*mask_batch + ((Sigma_hat.unsqueeze(0) * Sigma_mask_mo) @ inv_Sigma_oo @ ((z - mu_hat)*(1-mask_batch)).view(batch_size, d, 1)).squeeze()
                        z_hat = z*(1-mask_batch) + z_m
                        Sigma_m = Sigma_hat*(mask_batch.view(batch_size, d, 1) @ mask_batch.view(batch_size, 1, d)) \
                                          - (Sigma_hat * Sigma_mask_mo) @ inv_Sigma_oo @ (Sigma_hat*((1-mask_batch).view(batch_size, d, 1) @ mask_batch.view(batch_size, 1, d)))

                        # Compute local -> global mu and Sigma
                        mu_hat_local = torch.mean(z_hat, dim=0)
                        Sigma_hat_local = torch.mean((z_hat - mu_hat).view(batch_size, d, 1) @ (z_hat - mu_hat).view(batch_size, 1, d) + Sigma_m, dim=0)
                        mu_hat = rho[J]*mu_hat_local + (1 - rho[J])*mu_hat
                        Sigma_hat = rho[J]*Sigma_hat_local + (1 - rho[J])*Sigma_hat
                        # Sigma_hat = Sigma_hat + beta[J] * torch.diag(torch.diagonal(Sigma_hat))

                        # Initialize new base distribution
                        mu_hat = mu_hat.detach()
                        Sigma_hat = Sigma_hat.detach()
                        Sigma_hat = (Sigma_hat + torch.t(Sigma_hat)) / 2
                        new_base = torch.distributions.multivariate_normal.MultivariateNormal(mu_hat, covariance_matrix=Sigma_hat)
                        nfm.init_base(new_base)

                        # Update flow model again
                        x_tilde = nfm(z_hat)
                        log_prob_tilde = nfm.log_prob(x_tilde)
                        # L_rec = torch.sum((1 - mask_batch)*torch.pow(x_tilde - x_hat, 2), dim=1)
                        L_rec = torch.sum(torch.pow(x_tilde - x_true, 2), dim=1) #  2nd change: match real data instead of x_hat
                        L2 = -torch.mean(log_prob_tilde - alpha*L_rec)
                        L2.backward()
                        optimizer.step()

                return

        def inverse_masked(self, A, mask):
                num_observed = torch.count_nonzero(mask[0]).item()
                sub_matrix = torch.masked_select(A, mask.bool())
                sub_matrix = sub_matrix.view(np.sqrt(sub_matrix.shape[0]).astype(int), np.sqrt(sub_matrix.shape[0]).astype(int))
                inv_mat = torch.inverse(sub_matrix)
                B = torch.zeros(A.shape)
                B[mask.bool()] = inv_mat.flatten().float()
                return B

        def transform(self, data = None):
                with torch.no_grad():
                        d = self.d
                        nfm = self.nfm
                        if data is None:
                                X_hat, mask = self.X_hat, self.mask
                        else:
                                if data.ndim == 1:
                                        data = data.reshape((1, len(data)))
                                X_hat = torch.tensor(data).float()
                                mask = torch.isnan(X_hat).int()
                                X_hat = torch.nan_to_num(X_hat).float()

                        # X_hat, mask = self.generate_hat_mask(data)
                        Z = nfm.inverse(X_hat)

                        mu_hat = nfm.q0.loc
                        Sigma_hat = nfm.q0.covariance_matrix

                        Sigma_mask_mo = mask.view(X_hat.shape[0], d, 1) @ (1-mask).view(X_hat.shape[0], 1, d)
                        Sigma_mask_oo = (1-mask).view(X_hat.shape[0], d, 1) @ (1-mask).view(X_hat.shape[0], 1, d)
                        inv_Sigma_oo = torch.zeros(Sigma_mask_oo.shape)
                        for i in range(X_hat.shape[0]):
                                inv_Sigma_oo[i,:,:] = self.inverse_masked( Sigma_hat, Sigma_mask_oo[i,:,:] ).float()
                        Z_m = mu_hat*mask + ((Sigma_hat.unsqueeze(0) * Sigma_mask_mo) @ inv_Sigma_oo @ ((Z - mu_hat)*(1-mask)).view(X_hat.shape[0], d, 1)).squeeze()
                        Z_hat = Z*(1-mask) + Z_m

                        X_tilde = nfm(Z_hat)
                        X_hat = X_hat*(1-mask) + X_tilde*mask

                        return X_hat

        def model_save(self, file_name, save_dir = None):
                if save_dir is None:
                    save_dir = './imp_model'
                self.nfm.save(save_dir + '/' + file_name + '_merged.pth')
                return

        def model_load(self, file_name):
                self.init_model()
                self.nfm.load('./imp_model/' + file_name + '_merged.pth')
                return


# # ## test
# # data = np.random.rand(5000, 10)
# df = pd.read_csv('data/letter.csv')
# data=df.to_numpy()
# scaler = preprocessing.StandardScaler().fit(data)
# data = scaler.transform(data)

# true_data = data.copy()
# mask = (np.random.rand(data.shape[0], data.shape[1]) < 0.2).astype(np.int64)

# for i in range(len(mask)):
#       for j in range(len(mask[0])):
#               if mask[i,j] == 1:
#                       data[i,j] = np.nan

# perm = np.random.permutation(np.arange(data.shape[0]))
# train_ind = perm[:int(len(perm)*.8)]
# test_ind = perm[int(len(perm)*.8):]
# train = data[train_ind, :]
# test = data[test_ind, :]
# # X_hat_train, X_hat_test = X_hat[train_ind,:], X_hat[test_ind,:]
# mask_train, mask_test = mask[train_ind,:], mask[test_ind,:]

# impute_para = {'batch_size': 256,
#                          'lr': 1e-4,
#                          'alpha': 1e6}

# imputer = Imputer(train, impute_para)

# criterion = torch.nn.MSELoss()
# print('before_train_train_rmse:', criterion(imputer.transform(), torch.tensor(true_data[train_ind,:])))
# print('before_train_test_rmse:', criterion(imputer.transform(test), torch.tensor(true_data[test_ind,:])))

# imputer.train_model(max_iter = 100)

# print('before_train_train_rmse:', criterion(imputer.transform(), torch.tensor(true_data[train_ind,:])))
# print('before_train_test_rmse:', criterion(imputer.transform(test), torch.tensor(true_data[test_ind,:])))

# imputer.model_save('test')

# imputer = Imputer(train, impute_para)
# imputer.model_load('test')

# print('before_train_train_rmse:', criterion(imputer.transform(), torch.tensor(true_data[train_ind,:])))
# print('before_train_test_rmse:', criterion(imputer.transform(test), torch.tensor(true_data[test_ind,:])))
