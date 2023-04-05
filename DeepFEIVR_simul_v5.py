import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import scipy
import math
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, ZeroPadding1D, GlobalAveragePooling1D, Dense, Flatten, Activation
from tensorflow.keras.layers import Add, Dropout, Concatenate, Dot, Lambda
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
#from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization as BN
from scipy.stats import chi2
from scipy import stats
from itertools import combinations



save_dir_ori = "simul_0216v5/"
epochs = 15
learning_rate = 2e-4
RepeatTimes = 100
FEAT_SIZE_GLOBAL = 4
FEAT_SIZE_REAL = 2
Z_dim = 50
FX_center = True
N_val = 200
N_test1 = 4000
#N_test2 = 4000
N_test3 = 20000 # reference
l1_p = 1e-5
l2_p = 1e-4
D_list = np.array([[0.0, 0.02, 0.03, 0.05],[0.0, 0.02, 0.03, 0.05]])
beta = np.array([-0.1, 0.2, -0.2, 0.1])


def np_mse(y_true, y_pred):
    return ((y_true-y_pred)**2).mean()


def summary_data_ob(Z_test_sum, y_test_sum):
    summary_data = np.zeros((Z_test_sum.shape[1],2))
    for j in range(Z_test_sum.shape[1]):
        Z_test_tmp = Z_test_sum[:,j]
        ols = stats.linregress(Z_test_tmp, y_test_sum)
        summary_data[j,0] = ols[0]
        summary_data[j,1] = ols[4]
    return summary_data


def MVIWAS(summary_data, Z_ref1, W, n_G, feat_size = FEAT_SIZE_GLOBAL, feat_size_eff = FEAT_SIZE_GLOBAL, SLS_flag = False):
    q = W.shape[1]
    Z = Z_ref1
    if FX_center == False:
        Z = Z[:,1:]
        W = W[1:,:]
    n_R = len(Z)
    ZTZ_G = n_G/n_R*(Z.T@Z)
    alpha_G = summary_data[:,0]
    alpha_var_G = summary_data[:,1]**2
    ZTY_G = np.diag(ZTZ_G)*alpha_G
    YTY_G = np.median((n_G-1)*np.diag(ZTZ_G)*alpha_var_G+alpha_G*ZTY_G)
    WZZW = W.T@ZTZ_G@W
    if SLS_flag:
        WZZW_inv, q = scipy.linalg.pinv(WZZW,return_rank=True)
    else:
        WZZW_inv = np.linalg.inv(WZZW)
    beta_hat = WZZW_inv@W.T@ZTY_G
    beta_hat = np.expand_dims(beta_hat, axis=-1)
    sigma2_hat = 1/(n_G-q)*(YTY_G-beta_hat.T@W.T@ZTY_G)
    var_hat_beta_hat = WZZW_inv*sigma2_hat
    dis = chi2(feat_size_eff)
    #print(q, feat_size_eff, feat_size, beta_hat.shape)
    if SLS_flag:
        p_value = 1-dis.cdf(beta_hat[-feat_size:].T@np.linalg.pinv(var_hat_beta_hat[-feat_size:,-feat_size:])@beta_hat[-feat_size:])
    else:
        p_value = 1-dis.cdf(beta_hat[-feat_size:].T@np.linalg.inv(var_hat_beta_hat[-feat_size:,-feat_size:])@beta_hat[-feat_size:])
    if SLS_flag==False:
        t_dis = stats.t(n_G-q)
        t_record = np.zeros(q)
        for j in range(q):
            t_stat = beta_hat[j]/np.sqrt(var_hat_beta_hat[j,j])
            t_record[j] = 2*(1-t_dis.cdf(t_stat)) if t_dis.cdf(t_stat)>0.5 else 2*t_dis.cdf(t_stat)
    else:
        t_record = np.zeros(feat_size)
    #print(p_value)
    return p_value, t_record

def fx_model(width = 22, height = 22, feat_size = FEAT_SIZE_GLOBAL):
    """Build a 2D CNN for f(x)"""
    input1 = Input((width, height))
    x = tf.expand_dims(input1, axis=-1)
    x = Conv2D(filters=64, kernel_size=2, activation="relu")(x)
    x = MaxPool2D(pool_size=2)(x)
    x = BN()(x)
    x = Conv2D(filters=32, kernel_size=2, activation="relu")(x)
    x = MaxPool2D(pool_size=2)(x)
    x = BN()(x)
    x = Flatten()(x)
    x = Dense(units=64, activation="relu", kernel_regularizer = l1_l2(l1_p, l2_p))(x)
    x = Dropout(0.3)(x)
    fx = Dense(units=feat_size, kernel_regularizer = l1_l2(l1_p, l2_p))(x)
    fx = relu(fx, alpha=0.3)
    # if FX_center:
    #     #fx = fx - tf.math.reduce_mean(fx, axis=0,keepdims=True)
    #     fx = BN()(fx)
    model = Model(inputs=input1, outputs=fx, name="fx")
    return model

def linear_model(feat_size, z_dim, alpha = 1e-6, response_mode =0):
    fx = Input((feat_size,))
    input2 = tf.keras.Input((z_dim, )) if FX_center else tf.keras.Input((z_dim+1, ))
    if FX_center:
        alphaI = tf.eye(z_dim)
    else:
        alphaI = np.eye(z_dim+1, dtype=np.float32)
        alphaI[0,0]=0
        alphaI = tf.convert_to_tensor(alphaI)
    beta = tf.linalg.inv(tf.transpose(input2)@input2+batch_size*alpha*alphaI)@tf.transpose(input2)@fx
    y = input2@beta
    if response_mode == 0:
        outputs = Dense(units=1,name="get_w", kernel_regularizer = l1_l2(l1_p, l2_p))(y)
    else:
        outputs = Dense(units=1, activation="sigmoid",name="get_w", kernel_regularizer = l1_l2(l1_p, l2_p))(y)
    model = Model(inputs=[fx, input2], outputs=outputs, name="lm")
    return model

def linear_model_skip(feat_size, z_dim, alpha = 0.001, response_mode=0):
    fx = Input((feat_size,))
    input2 = tf.keras.Input((z_dim, )) if FX_center else tf.keras.Input((z_dim+1, ))
    if response_mode == 0:
        outputs = Dense(units=1,name="get_w", kernel_regularizer = l1_l2(l1_p, l2_p))(fx)
    else:
        outputs = Dense(units=1, activation="sigmoid",name="get_w", kernel_regularizer = l1_l2(l1_p, l2_p))(fx)
    model = Model(inputs=[fx, input2], outputs=outputs, name="lm")
    return model


def SLS_train2(z_all, x_all, y_all, feat_size, alpha = 1e-6, response_mode=0):
    Z_train_val = z_all[:(N_train+N_val)]
    Z_test = z_all[(N_train+N_val):(N_train+N_val+N_test1)]
    Z_test_sum = z_all[(N_train+N_val):(N_train+N_val+N_test1)]
    Z_ref = z_all[(N_train+N_val+N_test1):]
    x_all = x_all.reshape(len(x_all),20*20)
    x_train_val = x_all[:(N_train+N_val)]
    x_test = x_all[(N_train+N_val):(N_train+N_val+N_test1)]
    y_train_val = y_all[:(N_train+N_val)]
    y_test = y_all[(N_train+N_val):(N_train+N_val+N_test1)]
    y_test_sum = y_all[(N_train+N_val):(N_train+N_val+N_test1)]
    x_dim = x_test.shape[1]
    z_dim = Z_test.shape[1]
    Z_mean = np.expand_dims(Z_train_val.mean(axis=0),axis=0)
    Z_train_val1 = Z_train_val - np.repeat(Z_mean, repeats = Z_train_val.shape[0], axis=0)
    Z_test1 = Z_test - np.repeat(Z_mean, repeats = Z_test.shape[0], axis=0)
    Z_ref1 = Z_ref - np.repeat(np.expand_dims(Z_ref.mean(axis=0),axis=0), repeats = Z_ref.shape[0], axis=0)
    if FX_center == False:
        Z_train_val1 = np.concatenate([np.ones((Z_train_val.shape[0],1)), Z_train_val1], axis=-1)
        Z_test1 = np.concatenate([np.ones((Z_test.shape[0],1)), Z_test1], axis=-1)
        Z_ref1 = np.concatenate([np.ones((Z_ref.shape[0],1)), Z_ref1], axis=-1)
    if FX_center:
        alphaI = np.eye(z_dim)
    else:
        alphaI = np.eye(z_dim+1)
        alphaI[0,0]=0
    #beta = np.linalg.inv(Z_train_val1.T@Z_train_val1+(N_train+N_val)*alpha*alphaI)@Z_train_val1.T@x_train_val
    mod1 = Ridge(alpha=0).fit(Z_train_val1,x_train_val)
    beta_zx = (mod1.coef_).T
    output1 = np.zeros(2*feat_size+1)
    summary_data = summary_data_ob(Z_test_sum, y_test_sum)
    Z_test2 = Z_test1 - Z_test1.mean(axis=0)
    X_hat_train = Z_train_val1@beta_zx
    _, feat_size_eff = scipy.linalg.pinv(X_hat_train.T@X_hat_train,return_rank=True)
    print(beta_zx.shape[1], feat_size_eff)
    p_test_f, _ = MVIWAS(summary_data, Z_test2, beta_zx, n_G = N_test1, feat_size = beta_zx.shape[1], feat_size_eff=feat_size_eff,SLS_flag=True)
    #p_value_indv, t_record_indv = MVIWAS(summary_data, Z_test1, beta_zx, q=64*64, SLS_flag=True)
    output1[0] = p_test_f
    print("SLS: indv", p_test_f)
    p_value_sum, _ = MVIWAS(summary_data, Z_ref1, beta_zx, n_G = N_test1, feat_size = beta_zx.shape[1], feat_size_eff=feat_size_eff, SLS_flag=True)
    summary_out = np.concatenate([p_value_sum[0], np.zeros(feat_size)])
    print("SLS: sum", p_value_sum)
    #output1[0] = p_test_f
    return output1, 0, summary_out

def DF_train(z_all, x_all, y_all, feat_size, alpha = 1e-4, response_mode = 0, skip_flag = False, r = 0):
    Z_train = z_all[:N_train]
    Z_val = z_all[N_train:(N_train+N_val)]
    Z_test = z_all[(N_train+N_val):(N_train+N_val+N_test1)]
    Z_test_sum = np.copy(z_all[(N_train+N_val):(N_train+N_val+N_test1)])
    Z_ref = z_all[(N_train+N_val+N_test1):]
    x_train = x_all[:N_train]
    x_val = x_all[N_train:(N_train+N_val)]
    x_test = x_all[(N_train+N_val):(N_train+N_val+N_test1)]
    y_train = y_all[:N_train]
    y_val = y_all[N_train:(N_train+N_val)]
    y_test = y_all[(N_train+N_val):(N_train+N_val+N_test1)]
    y_test_sum = np.copy(y_all[(N_train+N_val):(N_train+N_val+N_test1)])
    z_dim = Z_test.shape[1]
    Z_mean = np.expand_dims(Z_train.mean(axis=0),axis=0)
    Z_train1 = Z_train - np.repeat(Z_mean, repeats = Z_train.shape[0], axis=0)
    # Z_train2 = Z_train-np.mean(Z_train,axis=0,keepdims=True)
    Z_val1 = Z_val - np.repeat(Z_mean, repeats = Z_val.shape[0], axis=0)
    Z_test1 = Z_test - np.repeat(Z_mean, repeats = Z_test.shape[0], axis=0)
    Z_ref1 = Z_ref - np.repeat(np.expand_dims(Z_ref.mean(axis=0),axis=0), repeats = Z_ref.shape[0], axis=0)
    # Z_ref2 = Z_ref-np.mean(Z_ref,axis=0,keepdims=True)
    if FX_center == False:
        Z_train1 = np.concatenate([np.ones((Z_train.shape[0],1)), Z_train1], axis=-1)
        Z_val1 = np.concatenate([np.ones((Z_val.shape[0],1)), Z_val1], axis=-1)
        Z_test1 = np.concatenate([np.ones((Z_test.shape[0],1)), Z_test1], axis=-1)
        Z_ref1 = np.concatenate([np.ones((Z_ref.shape[0],1)), Z_ref1], axis=-1)
    if FX_center:
        alphaI = np.eye(z_dim)
    else:
        alphaI = np.eye(z_dim+1)
        alphaI[0,0]=0
    Z_train_inv = np.linalg.inv(Z_train1.T@Z_train1+N_train*alpha*alphaI)
    Z_train_part = np.linalg.inv(Z_train1.T@Z_train1+N_train*alpha*alphaI)@Z_train1.T
    width = x_all.shape[1]
    height = x_all.shape[2]
    input_x = Input((width, height))
    input_z = Input((z_dim)) if FX_center else Input((z_dim+1))
    fx_m = fx_model(width = width, height = height, feat_size = feat_size)
    if skip_flag:
        l_m = linear_model_skip(feat_size = feat_size, z_dim = z_dim, alpha = alpha, response_mode = response_mode)
    else:
        l_m = linear_model(feat_size = feat_size, z_dim = z_dim, alpha = alpha, response_mode = response_mode)
    fx = fx_m(input_x)
    output = l_m([fx,input_z])
    model2 = Model(inputs = [input_x, input_z], outputs = output)
    if response_mode == 0:
        model2.compile(optimizer=Adam(lr=learning_rate), loss="mse")
    else:
        model2.compile(optimizer=Adam(lr=learning_rate), loss="binary_crossentropy")
    CE_val_record = np.ones(epochs)*1000 if response_mode==0 else np.zeros(epochs)
    CE_prev_1 = 1000
    CE_prev_2 = 1000
    for epoch in range(epochs):
        model2.fit([x_train, Z_train1],y_train, batch_size = batch_size, epochs = 1, verbose = 0, shuffle=True)
        model2.save_weights(save_dir+"model_weights_"+str(r)+"_"+str(epoch)+"_.h5")
        fx_m.save_weights(save_dir+"model_weights_"+str(r)+"_"+str(epoch)+"_fx_m.h5")
        l_m.save_weights(save_dir+"model_weights_"+str(r)+"_"+str(epoch)+"_l_m.h5")
        fx_hat = fx_m.predict(x_train, batch_size = batch_size) #(# of train, q)
        # fx_hat does not need to center as Z is centered
        # if FX_center:
        #     fx_hat = fx_hat - fx_hat.mean(axis=0, keepdims=True)
        beta = Z_train_part@fx_hat #(p,q)
        l_m_weights = l_m.get_layer("get_w").get_weights()
        np.save(save_dir+"beta0324__"+str(r)+"_"+str(epoch)+".npy",beta)
        np.save(save_dir+"l_m_weights0322__"+str(r)+"_"+str(epoch)+".npy",l_m_weights)
        if skip_flag:
            pred_val = fx_m.predict(x_val, batch_size = batch_size)@l_m_weights[0]+l_m_weights[1]
        else:
            pred_val = Z_val1@beta@l_m_weights[0]+l_m_weights[1]
        if response_mode!=0:
            pred_val = 1/(1+np.exp(-pred_val))
        pred_val = pred_val[:,0]
        if response_mode==0:
            tmp_mse = np_mse(y_val, pred_val)
            CE_val_record[epoch] = tmp_mse
            if CE_prev_1<tmp_mse and CE_prev_2<tmp_mse:
                break
            CE_prev_2 = CE_prev_1
            CE_prev_1 = tmp_mse
        else:
            tmp_auc = K.eval(tf.keras.metrics.AUC()(y_val,pred_val))
            # the following does not test
            CE_val_record[epoch] = tmp_auc
            if CE_prev_1>tmp_auc and CE_prev_1>tmp_auc:
                break
            CE_prev_2 = CE_prev_1
            CE_prev_1 = tmp_auc
        #print(CE_val_record[epoch])
    np.save(save_dir+"CE_val_record_"+str(r)+"_.npy",CE_val_record)
    if response_mode==0:
        epoch_id = np.argmin(CE_val_record)
    else:
        epoch_id = np.argmax(CE_val_record)
    print(epoch_id)
    model2.load_weights(save_dir+"model_weights_"+str(r)+"_"+str(epoch_id)+"_.h5")
    fx_m.load_weights(save_dir+"model_weights_"+str(r)+"_"+str(epoch_id)+"_fx_m.h5")
    l_m.load_weights(save_dir+"model_weights_"+str(r)+"_"+str(epoch_id)+"_l_m.h5")
    fx_hat = fx_m.predict(x_train, batch_size = len(x_train)) #(# of train, q)
    # if FX_center:
    #     fx_hat = fx_hat - fx_hat.mean(axis=0, keepdims=True)
    beta = Z_train_part@fx_hat #(1+p,q)
    l_m_weights = l_m.get_layer("get_w").get_weights()
    np.save(save_dir+"beta_save0317v1_"+str(r)+"_.npy",beta)
    if skip_flag:
        fx_test = fx_m.predict(x_test, batch_size = batch_size)
    else:
        fx_test= Z_test1@beta
    pred_test = fx_test@l_m_weights[0]+l_m_weights[1]
    if response_mode!=0:
        #pred_test = 1/(1+np.exp(-pred_test))
        p_value2t2 = np.zeros(feat_size)
        for k in range(feat_size):
            if not np.all(fx_test[:,k]==fx_test[0,k]):
                #p_value2t2[k] = scipy.stats.linregress(pred_test[:,k], y_test)[3]
                p_value2t2[k] = Logit(y_test, add_constant(fx_test[:,k])).fit().pvalues[1]
            else:
                p_value2t2[k]=-1
        feature_coef = np.corrcoef(np.transpose(fx_test))
        feature_del = np.zeros(feat_size)
        corr_thr = 0.8
        for t in range(feat_size):
            for s in range(t+1,feat_size):
                if abs(feature_coef[t,s])>corr_thr:
                    del1 = t if p_value2t2[t]>p_value2t2[s] else s
                    feature_del[del1]=1
                    #print(t,s, chosen_pv[t], chosen_pv[s], del1)
        fx_test_sh = fx_test[:,feature_del==0]
        feat_sh_size = fx_test_sh.shape[1]
        if FX_center:
            r_matrix = np.eye(feat_sh_size)
            y_test1 = y_test - np.mean(y_test)
            ols_test = Logit(y_test1, fx_test_sh).fit()
            p_test_f = ols_test.f_test(r_matrix).pvalue
        else:
            r_matrix = np.zeros((feat_sh_size, feat_sh_size+1))
            for i in range(feat_sh_size):
                r_matrix[i, i+1] = 1
            ols_test = Logit(y_test, add_constant(fx_test_sh)).fit()
        #ols_test = Logit(y_test, add_constant(fx_test_sh)).fit()
            p_test_f = ols_test.f_test(r_matrix).pvalue
    else:
        fx_test_red = list()
        for k in range(feat_size):
            if not np.all(fx_test[:,k]==fx_test[0,k]):
                fx_test_red.append(fx_test[:,k])
        fx_test_red = np.array(fx_test_red).T
        if fx_test_red.shape[0]==0:
            p_test_f = 1.0
            p_value2t2 = np.ones(feat_size)
            p_value2t3 = np.ones(feat_size)
        else:
            ols_test = OLS(y_test, add_constant(fx_test_red)).fit()
            #p_test_f = ols_test.f_pvalue
            p_test_f = ols_test.f_test(np.eye(fx_test_red.shape[1]+1)[1:]).pvalue
            p_value2t2 = np.ones(feat_size)*(-1)
            p_value2t2[:ols_test.pvalues[1:].shape[0]] = ols_test.pvalues[1:]
            p_value2t3 = np.ones(feat_size)
            for k in range(feat_size):
                if not np.all(fx_test[:,k]==fx_test[0,k]):
                    p_value2t3[k] = stats.linregress(fx_test[:,k], y_test)[3]
        #############
    print(p_test_f)
    print(p_value2t2)
    pred_test = pred_test[:,0]
    test_mse = np_mse(pred_test, y_test)
    summary_data = summary_data_ob(Z_test_sum, y_test_sum)
    if skip_flag:
        summary_out = np.zeros(feat_size+1)
    else:
        p_value_sum, t_record_sum = MVIWAS(summary_data, Z_ref1, beta, n_G = N_test1, feat_size = feat_size,feat_size_eff=feat_size)
        print(p_value_sum)
        print(t_record_sum)
        summary_out = np.concatenate([p_value_sum[0], t_record_sum])
    K.clear_session()
    return np.concatenate([np.array([p_test_f]), p_value2t2,p_value2t3]), test_mse, summary_out

def simulate_inv3(z_all, B, beta, D, sigma=1.0, response_mode = 0):
    N = z_all.shape[0]
    fx = z_all@B+np.repeat(np.expand_dims(u_all,axis=-1), FEAT_SIZE_REAL, axis=1)+0.5*sigma*np.random.randn(N, FEAT_SIZE_REAL)
    x_train_val_test = np.zeros((N_train+N_val+N_test1, 20, 20))
    for i in range(x_train_val_test.shape[0]):
        # x_start = 3 if fx[i,0]>0 else 15
        # y_start = 3 if fx[i,1]>0 else 15
        # x_train_val[i,x_start:(x_start+3),y_start:(y_start+6)] = np.sqrt(np.abs(fx[i,0]))
        # x_train_val[i,(x_start+3):(x_start+6),y_start:(y_start+6)] = np.sqrt(np.abs(fx[i,1]))
        y_mid0 = 5 if fx[i,0]>0 else 15
        y_mid1 = 5 if fx[i,1]>0 else 15
        radius0 = np.random.randint(2,5)
        radius1 = np.random.randint(2,5)
        x_train_val_test[i,(5-radius0):(5+radius0),(y_mid0-radius0):(y_mid0+radius0)] = np.sqrt(np.abs(fx[i,0]))
        x_train_val_test[i,(15-radius1):(15+radius1),(y_mid1-radius1):(y_mid1+radius1)] = np.sqrt(np.abs(fx[i,1]))
    #fx = z_all@B+np.repeat(np.expand_dims(u_all,axis=-1), FEAT_SIZE_REAL, axis=1)+sigma*np.random.randn(N, FEAT_SIZE_REAL)
    x_train_val_test += np.random.randn(N_train+N_val+N_test1, 20, 20)*0.1
    LT = fx@beta*D+u_all
    if response_mode == 0:
        y_all = LT+sigma*np.random.randn(N)
    else:
        y_all = np.random.binomial(1,1/(1+np.exp(-LT)),N)
    return z_all, x_train_val_test, y_all

z_cov = np.zeros((Z_dim,Z_dim))
z_cov_tmp = np.ones((int(Z_dim/10),int(Z_dim/10)))*0.1

for j in range(int(Z_dim/10)):
    z_cov_tmp[j,j]=1.0

for j in range(10):
    z_cov[j*5:(j+1)*5,j*5:(j+1)*5] = z_cov_tmp

for run_mode in [0,1,2,3,4,5]:
    save_dir = save_dir_ori+"run"+str(run_mode)+"/"
    if run_mode ==0:
        method_list = [0,1,2] # 0: DF-2SLS; 1: 2SLS; 2: DF
        response_list = [0] # 0: linear; 1: binary
    elif run_mode in [3,5]:
        method_list = [0,1] # 0: DF-2SLS; 1: 2SLS; 2: DF
        response_list = [0] # 0: linear; 1: binary
    else:
        method_list = [0] # 0: DF-2SLS; 1: 2SLS; 2: DF
        response_list = [0] # 0: linear; 1: binary
    if run_mode==6:
        D_list = np.array([[0.0,0.05]])
    else:
        D_list = np.array([[0.0, 0.02, 0.03, 0.05],[0.0, 0.02, 0.03, 0.05]])
    if run_mode == 0: # basic type
        batch_size = 32
        N_train = 800
        FEAT_SIZE = FEAT_SIZE_GLOBAL
        beta = np.array([-0.1, 0.2, -0.2, 0.1])
    elif run_mode == 1: # FEAT_SIZE to the real one
        batch_size = 32
        N_train = 800
        FEAT_SIZE=FEAT_SIZE_REAL
        beta = np.array([-0.1, 0.2, -0.2, 0.1])
    elif run_mode == 2: # batch_size to 16
        batch_size = 16
        N_train = 800
        FEAT_SIZE = FEAT_SIZE_GLOBAL
        beta = np.array([-0.1, 0.2, -0.2, 0.1])
    elif run_mode == 3: # silence
        batch_size = 32
        N_train = 800
        FEAT_SIZE = FEAT_SIZE_GLOBAL
        beta = np.array([0.0, 0.2, -0.2, 0.1])
    elif run_mode == 4: # larger training size
        batch_size = 32
        N_train = 1600
        FEAT_SIZE = FEAT_SIZE_GLOBAL
        beta = np.array([-0.1, 0.2, -0.2, 0.1])
    elif run_mode == 5: # weak IV
        batch_size = 32
        N_train = 800
        FEAT_SIZE = FEAT_SIZE_GLOBAL
        beta = np.array([-0.1, 0.2, -0.2, 0.1])
    beta = beta[:FEAT_SIZE_REAL]
    #u_ori = u_ori-u_ori.mean()
    # Response, Method, D, Repeat
    pvalue_table1 = np.zeros((len(response_list), len(method_list), len(D_list[0]), RepeatTimes, 2*FEAT_SIZE+1))
    pvalue_table2 = np.zeros((len(response_list), len(method_list), len(D_list[0]), RepeatTimes, FEAT_SIZE+1))
    mse_table = np.zeros((len(response_list), len(method_list),len(D_list[0]), RepeatTimes))
    np.random.seed(100)
    for r in range(RepeatTimes):
        N_all = N_train+N_val+N_test1+N_test3
        z_all = np.random.multivariate_normal(np.zeros(Z_dim),z_cov,size=N_all)
        u_all = np.random.randn(N_all)
        B = np.random.randn(Z_dim, FEAT_SIZE_REAL)
        if run_mode==5:
            B[:int(Z_dim/2),:] = np.random.randn(int(Z_dim/2), FEAT_SIZE_REAL)*0.1
        for response_mode in response_list:
            for dd in range(len(D_list[0])):
                D = D_list[response_mode, dd]
                z_all, x_all, y_all = simulate_inv3(z_all, B, beta, D, response_mode = response_mode)
                for method_id_idx in range(len(method_list)):
                    method_id = method_list[method_id_idx]
                    print("Repeat "+str(r)+" D is "+str(D)+ " response "+str(response_mode)+" mehthod "+ str(method_id))
                    # set response_mode = 0 as we still treat the binary outcome as quantitative
                    if method_id == 0:
                        p_value1, test_mse, p_value2 = DF_train(z_all, x_all, y_all, alpha = 1e-6, response_mode = 0, skip_flag=False, r = r, feat_size = FEAT_SIZE)
                    elif method_id == 1:
                        p_value1, test_mse, p_value2 = SLS_train2(z_all, x_all, y_all, alpha = 1e-6, response_mode = 0, feat_size = FEAT_SIZE)
                    else:
                        p_value1, test_mse, p_value2 = DF_train(z_all, x_all, y_all, alpha = 1e-6, response_mode = 0, skip_flag=True, r = r, feat_size = FEAT_SIZE)
                    pvalue_table1[response_mode, method_id_idx, dd, r,:] = p_value1
                    pvalue_table2[response_mode, method_id_idx, dd, r,:] = p_value2
                    mse_table[response_mode, method_id_idx, dd, r] = test_mse
        if (r+1)%10 == 0:
            print("r is", r)
            print((pvalue_table1[:,:,:,:(r+1),0]<0.05).mean(axis=3))
            print((pvalue_table2[:,:,:,:(r+1),0]<0.05).mean(axis=3))
            np.save(save_dir+"pvalue_table1_"+str(r)+".npy", pvalue_table1)
            np.save(save_dir+"pvalue_table2_"+str(r)+".npy", pvalue_table2)
            np.save(save_dir+"mse_table_"+str(r)+".npy", mse_table)

import numpy as np
from scipy.stats import cauchy, chi2
r = 99
run_mode = 5
np.set_printoptions(precision=3)
save_dir = "simul_0216v5/"
save_dir1 = save_dir+"run"+str(run_mode)+"/"
pvalue_table1 = np.load(save_dir1+"pvalue_table1_"+str(r)+".npy") # not
pvalue_table2 = np.load(save_dir1+"pvalue_table2_"+str(r)+".npy") # summary_statistics
mse_table = np.load(save_dir1+"mse_table_"+str(r)+".npy")
(pvalue_table1[:,:,:,:(r+1),0]<0.05).mean(axis=3)
(pvalue_table2[:,:,:,:(r+1),0]<0.05).mean(axis=3)
