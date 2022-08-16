from scipy.io import loadmat
import h5py
import numpy as np
from keras.applications.vgg19 import VGG19
from keras.models import Model

path = ''
# l: labels, y: txt, i:images
l_path = 'nus-wide-tc10-lall.mat'
y_path = 'nus-wide-tc10-yall.mat'
i_path = '/IAll/nus-wide-tc10-iall.mat'
l = loadmat(path+l_path)
y = h5py.File(path+y_path)
i = h5py.File(path+i_path)

print(l.keys())
print(y.keys())
print(i.keys())

l = l['LAll']
y = y['YAll']
i = i['IAll']
print(l.shape, y.shape, i.shape)

# 加载 VGG 19
vgg = VGG19(weights='imagenet')

vgg = Model(vgg.input, vgg.get_layer('fc2').output)  # 取 4096-D feature 输出

all_img =[]
for e in range(len(i)):
    img = i[e].T
    print(img.shape)
    img = np.expand_dims(img, axis=0)
    all_img.append(vgg.predict(img))
all_img = np.vstack(all_img)
print("images shape:", all_img.shape)
np.save(path+'nus_vgg19.npy', all_img)
print('image save successful!')

y_np = np.asarray(y)
print(y_np.shape)
l_np = np.asarray(l)
print(l_np.shape)
y_np = y_np.transpose()
print(y_np.shape)
np.save(path+"label.npy", l_np)
np.save(path+"tags.npy", y_np)
