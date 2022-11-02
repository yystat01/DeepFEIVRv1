import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, ZeroPadding1D, GlobalAveragePooling1D, Dense, Flatten, Activation
from tensorflow.keras.layers import Add, Dropout, Concatenate, Dot, Lambda
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization as BN
from scipy.stats import chi2
from scipy import stats


save_dir = "simul_1028v2/"
epochs = 15
learning_rate = 1e-4
RepeatTimes = 500
FEAT_SIZE = 3
FEAT_SIZE_REAL = 4
FX_center = True
batch_size = 80
N_train = 800
N_val = 200
N_test1 = 4000
N_test2 = 4000
N_test3 = 8000 # reference
l1_p = 1e-6
l2_p = 1e-5

def np_mse(y_true, y_pred):
    return ((y_true-y_pred)**2).mean()

def simulate(z_all, u_all, B, beta, D, sigma=1.0, response_mode = 0):
    N = z_all.shape[0]
    z_dim = z_all.shape[1]
    fx = z_all@B+np.repeat(u_all, [2,2], axis=1)+sigma*np.random.randn(N, FEAT_SIZE_REAL)
    LT = fx@beta*D+u_all.mean(axis=1)
    if response_mode == 0:
        y_all = LT+sigma*np.random.randn(N)
    else:
        y_all = np.random.binomial(1,1/(1+np.exp(-LT)),N)
    return y_all

def summary_data_ob(Z_test_sum, y_test_sum):
    summary_data = np.zeros((Z_test_sum.shape[1],2))
    for j in range(Z_test_sum.shape[1]):
        Z_test_tmp = Z_test_sum[:,j]
        ols = stats.linregress(Z_test_tmp, y_test_sum)
        summary_data[j,0] = ols[0]
        summary_data[j,1] = ols[4]
    return summary_data

def MVIWAS(summary_data, Z_ref1, W, n_G = N_test2, q = FEAT_SIZE, SLS_flag = False):
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
        WZZW_inv = np.linalg.inv(WZZW+1e-7*np.eye(64*64))
    else:
        WZZW_inv = np.linalg.inv(WZZW)
    beta_hat = WZZW_inv@W.T@ZTY_G
    beta_hat = np.expand_dims(beta_hat, axis=-1)
    sigma2_hat = 1/(n_G-q)*(YTY_G-beta_hat.T@W.T@ZTY_G)
    var_hat_beta_hat = WZZW_inv*sigma2_hat
    dis = chi2(q)
    if SLS_flag:
        p_value = 1-dis.cdf(beta_hat.T@np.linalg.inv(var_hat_beta_hat+1e-16*np.eye(64*64))@beta_hat)
    else:
        p_value = 1-dis.cdf(beta_hat.T@np.linalg.inv(var_hat_beta_hat+1e-16*np.eye(FEAT_SIZE))@beta_hat)
    if SLS_flag==False:
        t_dis = stats.t(n_G-q)
        t_record = np.zeros(q)
        for j in range(q):
            t_stat = beta_hat[j]/np.sqrt(var_hat_beta_hat[j,j])
            t_record[j] = 2*(1-t_dis.cdf(t_stat)) if t_dis.cdf(t_stat)>0.5 else 2*t_dis.cdf(t_stat)
    else:
        t_record = np.zeros(FEAT_SIZE)
    return p_value, t_record

def fx_model(width = 64, height = 64, feat_size = FEAT_SIZE):
    """Build a 2D CNN for f(x)"""
    input1 = Input((width, height))
    x = tf.expand_dims(input1, axis=-1)
    x = Conv2D(filters=32, kernel_size=3, activation="relu", kernel_regularizer = l1_l2(l1_p, l2_p))(x)
    x = MaxPool2D(pool_size=2)(x)
    x = BN()(x)
    x = Conv2D(filters=32, kernel_size=3, activation="relu", kernel_regularizer = l1_l2(l1_p, l2_p))(x)
    x = MaxPool2D(pool_size=2)(x)
    x = BN()(x)
    x = Conv2D(filters=64, kernel_size=3, activation="relu", kernel_regularizer = l1_l2(l1_p, l2_p))(x)
    x = MaxPool2D(pool_size=2)(x)
    x = BN()(x)
    x = GlobalAveragePooling2D()(x)
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

def SLS_train(z_all, x_all, y_all, alpha = 1e-6, response_mode=0):
    Z_train = z_all[:N_train]
    Z_val = z_all[N_train:(N_train+N_val)]
    Z_test = z_all[(N_train+N_val):(N_train+N_val+N_test1)]
    Z_test_sum = z_all[(N_train+N_val):(N_train+N_val+N_test1)]
    Z_ref = z_all[(N_train+N_val+N_test1):]
    x_all = x_all.reshape(len(x_all),64*64)
    x_train = x_all[:N_train]
    x_val = x_all[N_train:(N_train+N_val)]
    x_test = x_all[(N_train+N_val):(N_train+N_val+N_test1)]
    y_train = y_all[:N_train]
    y_val = y_all[N_train:(N_train+N_val)]
    y_test = y_all[(N_train+N_val):(N_train+N_val+N_test1)]
    y_test_sum = y_all[(N_train+N_val):(N_train+N_val+N_test1)]
    x_dim = x_test.shape[1]
    z_dim = Z_test.shape[1]
    Z_mean = np.expand_dims(Z_train.mean(axis=0),axis=0)
    Z_train1 = Z_train - np.repeat(Z_mean, repeats = Z_train.shape[0], axis=0)
    Z_val1 = Z_val - np.repeat(Z_mean, repeats = Z_val.shape[0], axis=0)
    Z_test1 = Z_test - np.repeat(Z_mean, repeats = Z_test.shape[0], axis=0)
    Z_ref1 = Z_ref - np.repeat(np.expand_dims(Z_ref.mean(axis=0),axis=0), repeats = Z_ref.shape[0], axis=0)
    if FX_center == False:
        Z_train1 = np.concatenate([np.ones((Z_train.shape[0],1)), Z_train1], axis=-1)
        Z_val1 = np.concatenate([np.ones((Z_val.shape[0],1)), Z_val1], axis=-1)
        Z_test1 = np.concatenate([np.ones((Z_test.shape[0],1)), Z_test1], axis=-1)
        Z_ref1 = np.concatenate([np.ones((Z_ref.shape[0],1)), Z_ref1], axis=-1)
    if FX_center:
        alphaI = np.eye(x_dim)
    else:
        alphaI = np.eye(x_dim+1)
        alphaI[0,0]=0
    beta = np.linalg.inv(Z_train1.T@Z_train1)@Z_train1.T@x_train
    summary_data = summary_data_ob(Z_test_sum, y_test_sum)
    p_value_sum, t_record_sum = MVIWAS(summary_data, Z_ref1, beta, q=64*64, SLS_flag=True)
    print("SLS", p_value_sum)
    summary_out = np.concatenate([np.array([p_value_sum]), t_record_sum])
    return np.zeros(FEAT_SIZE+1), 0, summary_out


def DF_train(z_all, x_all, y_all, alpha = 1e-4, response_mode = 0, skip_flag = False, r = 0):
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
    fx_m = fx_model(width = width, height = height)
    if skip_flag:
        l_m = linear_model_skip(feat_size = FEAT_SIZE, z_dim = z_dim, alpha = alpha, response_mode = response_mode)
    else:
        l_m = linear_model(feat_size = FEAT_SIZE, z_dim = z_dim, alpha = alpha, response_mode = response_mode)
    fx = fx_m(input_x)
    output = l_m([fx,input_z])
    model2 = Model(inputs = [input_x, input_z], outputs = output)
    if response_mode == 0:
        model2.compile(optimizer=Adam(lr=learning_rate), loss="mse")
    else:
        model2.compile(optimizer=Adam(lr=learning_rate), loss="binary_crossentropy")
    CE_val_record = np.zeros(epochs)
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
        np.save(save_dir+"l_m_weights0324__"+str(r)+"_"+str(epoch)+".npy",l_m_weights)
        if skip_flag:
            pred_val = fx_m.predict(x_val, batch_size = batch_size)@l_m_weights[0]+l_m_weights[1]
        else:
            pred_val = Z_val1@beta@l_m_weights[0]+l_m_weights[1]
        if response_mode!=0:
            pred_val = 1/(1+np.exp(-pred_val))
        pred_val = pred_val[:,0]
        if response_mode==0:
            CE_val_record[epoch] = np_mse(y_val, pred_val)
        else:
            CE_val_record[epoch] = K.eval(tf.keras.metrics.AUC()(y_val,pred_val))
        print(CE_val_record[epoch])
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
        p_value2t2 = np.zeros(FEAT_SIZE)
        for k in range(FEAT_SIZE):
            if not np.all(fx_test[:,k]==fx_test[0,k]):
                #p_value2t2[k] = scipy.stats.linregress(pred_test[:,k], y_test)[3]
                p_value2t2[k] = Logit(y_test, add_constant(fx_test[:,k])).fit().pvalues[1]
            else:
                p_value2t2[k]=-1
        feature_coef = np.corrcoef(np.transpose(fx_test))
        feature_del = np.zeros(FEAT_SIZE)
        corr_thr = 0.8
        for t in range(FEAT_SIZE):
            for s in range(t+1,FEAT_SIZE):
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
        ols_test = OLS(y_test, add_constant(fx_test)).fit()
        #p_test_f = ols_test.f_pvalue
        p_test_f = ols_test.f_test(np.eye(FEAT_SIZE+1)[1:]).pvalue
        p_value2t2 = ols_test.pvalues[1:]
        # if FX_center:
        #     # y_test1 = y_test - np.mean(y_test)
        #     # ols_test = OLS(y_test1, fx_test).fit()
        # else:
        #     ols_test = OLS(y_test, add_constant(fx_test)).fit()
        #p_test_f = ols_test.f_pvalue
        # p_test_f = ols_test.f_test(np.eye(FEAT_SIZE)).pvalue
        # #########
        # p_value2t2 = np.zeros(FEAT_SIZE)
        # for k in range(FEAT_SIZE):
        #     if not np.all(fx_test[:,k]==fx_test[0,k]):
        #         p_value2t2[k] = stats.linregress(fx_test[:,k], y_test)[3]
        #     else:
        #         p_value2t2[k]=-1
        #############
    print(p_test_f)
    print(p_value2t2)
    pred_test = pred_test[:,0]
    test_mse = np_mse(pred_test, y_test)
    if skip_flag:
        summary_out = np.zeros(FEAT_SIZE+1)
    else:
        summary_data = summary_data_ob(Z_test_sum, y_test_sum)
        p_value_sum, t_record_sum = MVIWAS(summary_data, Z_ref1, beta)
        print(p_value_sum)
        print(t_record_sum)
        summary_out = np.concatenate([p_value_sum[0], t_record_sum])
    K.clear_session()
    return np.concatenate([np.array([p_test_f]), p_value2t2]), test_mse, summary_out

D_list = np.array([[0.0, 0.1, 0.2, 0.5],[0.0, 0.1, 0.2, 0.5]])
method_list = [0] # 0: DF-2SLS; 1: 2SLS
response_list = [0, 1] # 0: linear; 1: binary
data_ori = np.load("ZZ/dsprites_imgv1.npy")
value = np.load("ZZ/dsprites_valv1.npy")
z_ori = np.array([value[:,0], value[:,1], value[:,3], value[:,4]])
z_ori = np.moveaxis(z_ori, 0,-1).astype(float)
u_ori = np.array([value[:,2]==1,value[:,2]==2])
u_ori = np.moveaxis(u_ori, 0,-1).astype(float)
u_ori[:,0] = u_ori[:,0]-u_ori[:,0].mean()
u_ori[:,1] = u_ori[:,1]-u_ori[:,1].mean()
# Response, Method, D, Repeat
pvalue_table1 = np.zeros((len(response_list), len(method_list), len(D_list[0]), RepeatTimes, FEAT_SIZE+1))
pvalue_table2 = np.zeros((len(response_list), len(method_list), len(D_list[0]), RepeatTimes, FEAT_SIZE+1))
mse_table = np.zeros((len(response_list), len(method_list),len(D_list[0]), RepeatTimes))
for r in range(RepeatTimes):
    all_id = np.random.choice(np.arange(len(z_ori)),N_train+N_val+N_test1+N_test3,replace=False)
    x_all = data_ori[all_id]
    z_all = z_ori[all_id]
    u_all = u_ori[all_id]
    z_dim = z_all.shape[1]
    B = np.random.randn(z_dim, FEAT_SIZE_REAL)
    beta = np.array([-0.1, -0.2, 0.1, 0.2])
    for response_mode in response_list:
        for dd in range(len(D_list[0])):
            D = D_list[response_mode, dd]
            y_all = simulate(z_all, u_all, B, beta, D, response_mode = response_mode)
            for method_id in method_list:
                print("Repeat "+str(r)+" D is "+str(D)+ " response "+str(response_mode)+" mehthod "+ str(method_id))
                # set response_mode = 0 as we still treat the binary outcome as quantitative
                if method_id == 0:
                    p_value1, test_mse, p_value2 = DF_train(z_all, x_all, y_all, alpha = 1e-6, response_mode = 0, skip_flag=False, r = r)
                elif method_id == 1:
                    p_value1, test_mse, p_value2 = SLS_train(z_all, x_all, y_all, alpha = 1e-6, response_mode = 0)
                else:
                    p_value1, test_mse, p_value2 = DF_train(z_all, x_all, y_all, alpha = 1e-6, response_mode = 0, skip_flag=True, r = r)
                pvalue_table1[response_mode, method_id, dd, r,:] = p_value1
                pvalue_table2[response_mode, method_id, dd, r,:] = p_value2
                mse_table[response_mode, method_id, dd, r] = test_mse
    if (r+1)%10 == 0:
        print("r is", r)
        print((pvalue_table1[:,:,:,:r,0]<0.05).mean(axis=3))
        print((pvalue_table2[:,:,:,:r,0]<0.05).mean(axis=3))
        np.save(save_dir+"pvalue_table1_"+str(r)+".npy", pvalue_table1)
        np.save(save_dir+"pvalue_table2_"+str(r)+".npy", pvalue_table2)
        np.save(save_dir+"mse_table_"+str(r)+".npy", mse_table)
