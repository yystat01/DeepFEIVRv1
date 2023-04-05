import numpy as np
import scipy.stats
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, ZeroPadding1D, GlobalAveragePooling1D, Dense, Flatten, Activation
from tensorflow.keras.layers import Add, Dropout, Concatenate, Dot, Lambda, Conv3D, MaxPool3D, GlobalAveragePooling3D
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.layers import LayerNormalization as LN
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from tensorflow.keras.losses import BinaryCrossentropy
import os
import random
from scipy import ndimage
os.environ["CUDA_VISIBLE_DEVICES"]="0"


skip_flag = False
summary_flag = "IDP"
save_str = "Feb08V8" #save_str = "Oct27V3"
save_dir =  save_str+"_skip/" if skip_flag else save_str+"_idp/"
FX_center = True
testing_mode = True
epochs = 15
learning_rate = 1e-4
FEAT_SIZE = 20
batch_size = 16
l1_p = 1e-6
l2_p = 1e-5

def np_binary_cross_entropy(y_true, y_pred, a, b):
    return -(a*(1-y_true)*np.log(1-y_pred+1e-8)+b*y_true*np.log(y_pred+1e-8)).mean()

def myMSE(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


def fx_model(width=192, height=192, depth=160, feat_size = FEAT_SIZE):
    """Build a 3D CNN for fx (follow Dipnil)"""
    input1 = Input((width, height, depth, 3))
    x = Conv3D(filters=16, kernel_size=3, activation="relu")(input1)
    x = MaxPool3D(pool_size=2)(x)
    x = BN()(x)
    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = BN()(x)
    x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = BN()(x)
    x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = BN()(x)
    x = GlobalAveragePooling3D()(x)
    x = Dense(units=256, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(units=128, activation="relu")(x)
    #x = Dropout(0.3)(x)
    fx = Dense(units=feat_size, kernel_regularizer = l1_l2(l1_p, l2_p))(x)
    fx = relu(fx, alpha=0.3)
    # if FX_center:
    #     fx = fx - tf.math.reduce_mean(fx, axis=0,keepdims=True)
    model = Model(inputs=input1, outputs=fx, name="fx")
    return model

def linear_model_s(feat_size, z_dim, cov_dim, alpha = 0.001):
    fx = Input((feat_size,))
    input_z = tf.keras.Input((z_dim, )) if FX_center else tf.keras.Input((z_dim+1, ))
    input_cov = tf.keras.Input((cov_dim, ))
    if FX_center:
        alphaI = tf.eye(z_dim+cov_dim)
    else:
        alphaI = np.eye(z_dim+cov_dim+1, dtype=np.float32)
        alphaI[0,0]=0
        alphaI = tf.convert_to_tensor(alphaI)
    input_z_cov = tf.concat([input_z,input_cov],axis=-1)
    beta = tf.linalg.inv(tf.transpose(input_z_cov)@input_z_cov+alpha*alphaI)@tf.transpose(input_z_cov)@fx
    fx_est = input_z_cov@beta
    y = tf.concat([fx_est,input_cov],axis=-1)
    outputs = Dense(units=1,name="get_w",kernel_regularizer = l1_l2(l1_p, l2_p))(y)
    model = Model(inputs=[fx, input_z, input_cov], outputs=outputs, name="lm")
    return model

def linear_model_skip(feat_size, z_dim, cov_dim, alpha = 0.001):
    fx = Input((feat_size,))
    input_z = tf.keras.Input((z_dim, )) if FX_center else tf.keras.Input((z_dim+1, ))
    input_cov = tf.keras.Input((cov_dim, ))
    y = tf.concat([fx,input_cov],axis=-1)
    outputs = Dense(units=1, name="get_w", kernel_regularizer = l1_l2(l1_p, l2_p))(y)
    model = Model(inputs=[fx, input_z, input_cov], outputs=outputs, name="lm")
    return model

def shink_ZZ(ZZ, label, ZZ_name, pvalue_thr = 1e-3, corr_thr = 0.8): #5e-8 #0.001, 0.2, 282
    N, Q = ZZ.shape
    chosen_record = np.zeros(Q)
    if summary_flag=="No":
        pvalue_thr = pvalue_thr/10000
    for q in range(Q):
        chosen_record[q] = scipy.stats.linregress(ZZ[:,q], label)[3]
    print("period1")
    ZZc = ZZ[:,chosen_record<pvalue_thr] #if np.all(chosen_record<pvalue_thr) else ZZ[:100]
    chosen_pv = chosen_record[chosen_record<pvalue_thr] #if np.all(chosen_record<pvalue_thr) else chosen_record[:100]
    ZZ_name_c = ZZ_name[chosen_record<pvalue_thr]
    ZZ_coef = np.corrcoef(np.transpose(ZZc))
    np.save("0306pvalues.npy",chosen_record)
    print("period2")
    Q1 = ZZ_coef.shape[1]
    ZZ_del = np.zeros(Q1)
    for t in range(Q1):
        for s in range(t+1,Q1):
            if abs(ZZ_coef[t,s])>corr_thr:
                del1 = t if chosen_pv[t]>chosen_pv[s] else s
                ZZ_del[del1]=1
    final = ZZc[:,ZZ_del==0]
    print(final.shape)
    index = np.arange(Q)[chosen_record<pvalue_thr][ZZ_del==0] #if np.all(chosen_record<pvalue_thr) else np.arange(Q)[:100][ZZ_del==0]
    return final, index

def model_train(Image_data, label, ZZ, ZZ_name, cov,
                image_id_meta, alpha = 0.001, response_mode = 0, r_state=0, train_mode=2, gene_ind=1, chr=19):
    np.random.seed(r_state)
    all_id = np.random.choice(np.arange(len(ZZ)), len(ZZ), replace=False)
    print(all_id)
    np.save(save_dir+"all_id_"+str(r_state)+".npy",all_id)
    train_val_thre = int(len(ZZ)*0.9/batch_size)*batch_size
    n_test = len(all_id)-train_val_thre
    if testing_mode:
        train_id = all_id[:int(len(ZZ)*0.8/batch_size)*batch_size]
        val_id = all_id[(int(len(ZZ)*0.8/batch_size)*batch_size):train_val_thre]
        test_id = all_id[train_val_thre:]
    else:
        train_id = all_id[:train_val_thre]
        val_id = all_id[train_val_thre:]
        test_id = np.copy(all_id[train_val_thre:]) # we name it test, but actually it is still val, we use summary statistics as the test set
    x_train = Image_data[train_id]
    y_train = label[train_id]
    Z_train = ZZ[train_id]
    cov_train = cov[train_id]
    x_val = Image_data[val_id]
    y_val = label[val_id]
    Z_val = ZZ[val_id]
    cov_val = cov[val_id]
    x_test = np.copy(Image_data[test_id])
    y_test = np.copy(label[test_id])
    Z_test = np.copy(ZZ[test_id])
    print("label mean:", label.mean())
    print("label test mean: ",y_test.mean())
    cov_test = cov[test_id]
    image_id_meta_all = image_id_meta[all_id]
    np.save(save_dir+"image_id_all_0512_v2"+str(r_state)+".npy",image_id_meta_all)
    Z_train, z_index = shink_ZZ(Z_train, y_train, ZZ_name)
    np.save(save_dir+"z_index_0517_v1"+str(r_state)+".npy",z_index)
    ZZ_name_new = ZZ_name[z_index]
    np.save(save_dir+"ZZ_name_0517_v1"+str(r_state)+".npy",ZZ_name_new)
    np.savetxt(save_dir+"ZZ_name_0517_v1"+str(r_state)+".txt",ZZ_name_new,fmt="%s")
    Z_val = Z_val[:,z_index]
    Z_test = Z_test[:,z_index]
    z_dim = Z_train.shape[1]
    cov_dim = cov_train.shape[1]
    Z_mean = np.expand_dims(Z_train.mean(axis=0),axis=0)
    Z_train1 = Z_train - np.repeat(Z_mean, repeats = Z_train.shape[0], axis=0)
    Z_val1 = Z_val - np.repeat(Z_mean, repeats = Z_val.shape[0], axis=0)
    Z_test1 = Z_test - np.repeat(Z_mean, repeats = Z_test.shape[0], axis=0)
    cov_mean = np.expand_dims(cov_train.mean(axis=0),axis=0)
    cov_train1 = cov_train - np.repeat(cov_mean, repeats = cov_train.shape[0], axis=0)
    cov_val1 = cov_val - np.repeat(cov_mean, repeats = cov_val.shape[0], axis=0)
    cov_test1 = cov_test - np.repeat(cov_mean, repeats = cov_test.shape[0], axis=0)
    Z_cov_train1 = np.concatenate([Z_train1,cov_train1],axis=1)
    Z_cov_val1 = np.concatenate([Z_val1,cov_val1],axis=1)
    Z_cov_test1 = np.concatenate([Z_test1,cov_test1],axis=1)
    if FX_center==False:
        Z_train1 = np.concatenate([np.ones((Z_train.shape[0],1)), Z_train1], axis=-1)
        Z_val1 = np.concatenate([np.ones((Z_val.shape[0],1)), Z_val1], axis=-1)
        Z_test1 = np.concatenate([np.ones((Z_test.shape[0],1)), Z_test1], axis=-1)
    if FX_center:
        alphaI = np.eye(z_dim+cov_dim)
    else:
        alphaI = np.eye(z_dim+cov_dim+1)
        alphaI[0,0]=0
    Z_cov_train_part1 = np.linalg.inv(Z_cov_train1.T@Z_cov_train1+alpha*train_val_thre*alphaI)@Z_cov_train1.T
    #Pz_train1 = Z_train1@np.linalg.inv(Z_train1.T@Z_train1+alpha*train_val_thre*alphaI)@Z_train1.T
    #Qw_train1 = np.eye(Z_train1.shape[0])-cov_train1@np.linalg.inv(cov_train1.T@cov_train1+alpha*alphaI_cov)@cov_train1.T
    x_all = np.concatenate([x_train, x_val], axis=0)
    Z_all1 = np.concatenate([Z_train1, Z_val1], axis=0)
    cov_all1 = np.concatenate([cov_train1, cov_val1], axis=0)
    Z_cov_all1 = np.concatenate([Z_cov_train1, Z_cov_val1], axis=0)
    #Z_cov_all1 = np.concatenate([Z_cov_train1, Z_cov_val1], axis=0)
    width, height, depth =x_train.shape[1], x_train.shape[2], x_train.shape[3]
    input_x = Input((width, height, depth, 3))
    input_z = Input((z_dim)) if FX_center else Input((z_dim+1))
    input_cov = Input((cov_dim))
    fx_m = fx_model(width=width, height=height, depth=depth,feat_size = FEAT_SIZE)
    if skip_flag:
        l_m = linear_model_skip(feat_size = FEAT_SIZE, z_dim = z_dim, cov_dim = cov_dim, alpha = alpha/train_val_thre*batch_size)
    else:
        l_m = linear_model_s(feat_size = FEAT_SIZE, z_dim = z_dim, cov_dim = cov_dim, alpha = alpha/train_val_thre*batch_size)
    fx = fx_m(input_x)
    output = l_m([fx,input_z, input_cov])
    model2 = Model(inputs = [input_x, input_z,input_cov], outputs = output)
    model2.compile(optimizer=Adam(lr=learning_rate), loss="mse")
    CE_val_record = np.zeros(epochs)+10000
    for epoch in range(epochs):
        model2.fit([x_train, Z_train1,cov_train1],y_train, batch_size = batch_size, epochs = 1, verbose = 1, shuffle=True)
        model2.save_weights(save_dir+"model_weights"+str(epoch)+"_"+str(r_state)+".tf")
        fx_m.save_weights(save_dir+"model_weights"+str(epoch)+"_"+str(r_state)+"fx_m.tf")
        l_m.save_weights(save_dir+"model_weights"+str(epoch)+"_"+str(r_state)+"l_m.tf")
        if epoch>2:
            if skip_flag == False:
                fx_hat = fx_m.predict(x_train, batch_size = int(batch_size/2)) #(# of train, q)
                # if FX_center:
                #     fx_hat = fx_hat - fx_hat.mean(axis=0, keepdims=True)
                beta = Z_cov_train_part1@fx_hat #(1+p,q)
                #np.save(save_dir+"beta0324_"+str(epoch)+".npy",beta)
            l_m_weights = l_m.get_layer("get_w").get_weights()
            np.save(save_dir+"l_m_weights0324_"+str(epoch)+".npy",l_m_weights)
            if skip_flag:
                pred_val = model2.predict([x_val, Z_val1, cov_val1],batch_size = int(batch_size/2))
            else:
                pred_val = np.concatenate([Z_cov_val1@beta, cov_val1],axis=-1)@l_m_weights[0]+l_m_weights[1]
            pred_val = pred_val[:,0]
            CE_val_record[epoch] = myMSE(y_val,pred_val)
        print(CE_val_record[epoch])
    np.save(save_dir+"CE_val_record.npy",CE_val_record)
    epoch_id = np.argmin(CE_val_record)
    print(epoch_id)
    #print(epoch_id_s)
    model2.load_weights(save_dir+"model_weights"+str(epoch_id)+"_"+str(r_state)+".tf")
    fx_m.load_weights(save_dir+"model_weights"+str(epoch_id)+"_"+str(r_state)+"fx_m.tf")
    l_m.load_weights(save_dir+"model_weights"+str(epoch_id)+"_"+str(r_state)+"l_m.tf")
    model2.save_weights(save_dir+"model_weights_best_"+str(r_state)+".tf")
    fx_m.save_weights(save_dir+"model_weights_best_"+str(r_state)+"fx_m.tf")
    l_m.save_weights(save_dir+"model_weights_best_"+str(r_state)+"l_m.tf")
    fx_hat = fx_m.predict(x_train, batch_size = batch_size) #(# of train, q)
    # if FX_center:
    #     fx_hat = fx_hat - fx_hat.mean(axis=0, keepdims=True)
    beta = Z_cov_train_part1@fx_hat #(1+p,q)
    l_m_weights = l_m.get_layer("get_w").get_weights()
    np.save(save_dir+"beta_save0317v1_"+str(r_state)+".npy",beta)
    fx_hat_all = fx_m.predict(x_all, batch_size = batch_size)
    np.save(save_dir+"fx_hat_0512_"+str(r_state)+".npy",fx_hat_all)
    if skip_flag==False:
        fx_hat_all_af = Z_cov_all1@beta
        np.save(save_dir+"fx_hat_0512_"+str(r_state)+"_af_.npy",fx_hat_all_af)
    if skip_flag:
        pred_train= Z_cov_train1@beta
        np.save(save_dir+"pred_train_"+str(r_state)+".npy",pred_train)
        np.save(save_dir+"y_train_"+str(r_state)+".npy",y_train)
        pred_train_label = np.concatenate([pred_train,cov_train1],axis=-1)@l_m_weights[0]+l_m_weights[1]
        pred_val= Z_val1@beta
        np.save(save_dir+"pred_val_"+str(r_state)+".npy",pred_val)
        np.save(save_dir+"y_val_"+str(r_state)+".npy",y_val)
        pred_val_label = np.concatenate([pred_val,cov_val1],axis=-1)@l_m_weights[0]+l_m_weights[1]
        pred_test= Z_test1@beta
        np.save(save_dir+"pred_test_"+str(r_state)+".npy",pred_test)
        np.save(save_dir+"y_test_"+str(r_state)+".npy",y_test)
        #pred_test_label = np.concatenate([fx_m.predict(x_test, batch_size = batch_size),cov_test1],axis=-1)@l_m_weights[0]+l_m_weights[1]
        pred_test_label = model2.predict([x_test, Z_test1, cov_test1],batch_size = int(batch_size/2))
    else:
        pred_train= Z_cov_train1@beta
        np.save(save_dir+"pred_train_"+str(r_state)+".npy",pred_train)
        np.save(save_dir+"y_train_"+str(r_state)+".npy",y_train)
        pred_train_label = np.concatenate([pred_train, cov_train1],axis=-1)@l_m_weights[0]+l_m_weights[1]
        pred_val= Z_cov_val1@beta
        np.save(save_dir+"pred_val_"+str(r_state)+".npy",pred_val)
        np.save(save_dir+"y_val_"+str(r_state)+".npy",y_val)
        pred_val_label = np.concatenate([pred_val, cov_val1],axis=-1)@l_m_weights[0]+l_m_weights[1]
        pred_test= Z_cov_test1@beta
        np.save(save_dir+"pred_test_"+str(r_state)+".npy",pred_test)
        np.save(save_dir+"y_test_"+str(r_state)+".npy",y_test)
        pred_test_label = np.concatenate([pred_test, cov_test1],axis=-1)@l_m_weights[0]+l_m_weights[1]
    pred_val_s_label = model2.predict([x_val, Z_val1, cov_val1], batch_size = batch_size)
    pred_test_s_label = model2.predict([x_test, Z_test1,cov_test1], batch_size = batch_size)
    pred_train_label = pred_train_label[:,0]
    pred_val_label = pred_val_label[:,0]
    pred_test_label = pred_test_label[:,0]
    pred_val_s_label = pred_val_s_label[:,0]
    pred_test_s_label = pred_test_s_label[:,0]
    print(pred_test_label.shape)
    np.save(save_dir+"pred_train_"+str(r_state)+".npy",pred_train)
    np.save(save_dir+"y_train_"+str(r_state)+".npy",y_train)
    np.save(save_dir+"y_val_"+str(r_state)+".npy",y_val)
    np.save(save_dir+"y_test_"+str(r_state)+".npy",y_test)
    np.save(save_dir+"pred_train_"+str(r_state)+"_label.npy",pred_train_label)
    np.save(save_dir+"pred_val_"+str(r_state)+"_label.npy",pred_val_label)
    np.save(save_dir+"pred_val_s_"+str(r_state)+"_label.npy",pred_val_s_label)
    np.save(save_dir+"pred_test_"+str(r_state)+"_label.npy",pred_test_label)
    np.save(save_dir+"pred_test_s_"+str(r_state)+"_label.npy",pred_test_s_label)
    np.save(save_dir+"Z_train_"+str(r_state)+".npy",Z_train1)
    np.save(save_dir+"Z_val_"+str(r_state)+".npy",Z_val1)
    np.save(save_dir+"Z_test_"+str(r_state)+".npy",Z_test1)
    np.save(save_dir+"cov_train_"+str(r_state)+".npy",cov_train)
    np.save(save_dir+"cov_val_"+str(r_state)+".npy",cov_val)
    #np.save(save_dir+"cov_test_"+str(r_state)+".npy",cov_test)
    np.save(save_dir+"X_test_"+str(r_state)+".npy",x_test)
    #np.save(save_dir+"p_value2t2_"+str(r_state)+".npy",p_value2t2)
    K.clear_session()
    print("final results")
    print(np.array([p_test_f]), p_value2t2)
    return 0

r_state_list =
sss = 3
Image_data = np.load("brainImg_data/brainImg1017_v1.npy")
label = np.load("brainImg_data/label1017_v1.npy") # 175
ZZ = np.load("SNP/adni_snp1017_v1.npy")
ZZ_name = np.load("SNP/snp_name1017_v1.npy",allow_pickle=True)

image_id_meta = np.load("brainImg_data/image_id_meta1017_v1.npy") # image_id number only
cov = np.load("brainImg_data/covariates_1017_v1.npy",allow_pickle=True)[:,1:(-2)].astype(float)[:,np.array([0,1,2])]
for r_state in r_state_list:
    model_train(Image_data, label, ZZ, ZZ_name, cov, image_id_meta = image_id_meta, alpha=1e-3,r_state = r_state)
