import numpy as np
from scipy.stats import chi2
from scipy import stats
save_str = "Oct27V3"
save_dir = save_str+"_noskip/"
date = "0729"
r_state = 7
q=20
FX_center= True
n_G = 17008+37154
Z = np.load(save_dir+"ref"+date+save_str+"v1.npy")
Z_mean = np.expand_dims(Z.mean(axis=0),axis=0)
Z = Z - np.repeat(Z_mean, repeats = Z.shape[0], axis=0)
W = np.load(save_dir+"beta_save0317v1_"+str(r_state)+".npy")
if FX_center== False:
    #Z = Z1[:,1:]
    W = W[1:,:]

n_R = len(Z)
ZTZ_G = n_G/n_R*(Z.T@Z)
summary_data = np.load(save_dir+"sum_"+date+save_str+"v1.npy")
alpha_G = summary_data[:,1].astype(float)
alpha_var_G = summary_data[:,2].astype(float)**2
ZTY_G = np.diag(ZTZ_G)*alpha_G
YTY_G = np.median((n_G-1)*np.diag(ZTZ_G)*alpha_var_G+alpha_G*ZTY_G)

WZZW = W.T@ZTZ_G@W
WZZW_inv = np.linalg.inv(WZZW)
beta_hat = WZZW_inv@W.T@ZTY_G
beta_hat = np.expand_dims(beta_hat, axis=-1)
sigma2_hat = 1/(n_G-q)*(YTY_G-beta_hat.T@W.T@ZTY_G)
var_hat_beta_hat = WZZW_inv*sigma2_hat

dis = chi2(q)
1-dis.cdf(beta_hat.T@np.linalg.inv(var_hat_beta_hat)@beta_hat)
beta_hat.T@np.linalg.inv(var_hat_beta_hat)@beta_hat

t_dis = stats.t(n_G-q)
t_record = np.zeros(q)
for j in range(q):
    t_stat = beta_hat[j]/np.sqrt(var_hat_beta_hat[j,j])
    t_record[j] = 2*(1-t_dis.cdf(t_stat)) if t_dis.cdf(t_stat)>0.5 else 2*t_dis.cdf(t_stat)
