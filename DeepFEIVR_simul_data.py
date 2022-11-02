import numpy as np
import tensorflow_datasets as tfds
N = 737280
batch_s = 1000
data_ite = tfds.as_numpy(tfds.load("dsprites")['train'].batch(batch_s))
image_list = list()
value_list = list()
for i in range(int(N/batch_s)+1):
    tmp = next(data_ite)
    image_list.append(tmp['image'][:,:,:,0])
    value_tmp = np.array([tmp['value_orientation'], tmp['value_scale'],
                        tmp['value_shape'],tmp['value_x_position'],tmp['value_y_position'],
                        tmp['label_orientation'], tmp['label_scale'],
                                            tmp['label_shape'],tmp['label_x_position'],tmp['label_y_position']])
    value_list.append(np.moveaxis(value_tmp,0,-1))
    if i%10 == 0:
        print(i)
data = np.concatenate(image_list)[:N]
value = np.concatenate(value_list)[:N]
np.save("ZZ/dsprites_imgv1.npy",data)
np.save("ZZ/dsprites_valv1.npy",value)
