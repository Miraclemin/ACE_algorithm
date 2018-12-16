# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

num = np.load('num.npy')

pxpy = num / num.sum()
px = num.sum(1) / num.sum()
py = num.sum(0) / num.sum()

py_px = np.zeros_like(num)     #  py|x
for i in range(10):
    py_px[i] = pxpy[i] / px[i]
    
px_py = np.zeros_like(num)   # px|y
for i in range(15):
    px_py[:, i] = pxpy[:, i] / py[i]
    
gy = np.random.rand(1, 15).reshape(15 ,1)   # randomly initialize gy, fx
fx = np.random.rand(1, 10).reshape(10 ,1)


def normalize(gy, py):         # noramlize
    gy = gy - py.dot(gy)
    gy = gy / np.sqrt(py.dot(gy*gy))
    return gy

s = 0
s_ = 1e-4
while np.abs(s_ - s) >= 1e-10 :    # end when E[fxgy] stop increase
    s = s_
    fx = py_px.dot(gy)        #fx = E[gy|X =x]
    gy = px_py.T.dot(fx)      #gy = E[fx|Y =y]
    
    gy = normalize(gy, py)
    
    s_ = fx.T.dot(pxpy).dot(gy)  # E[fxgy]
    
    

second_sv = np.sqrt(s).item()   # why square root of s  is  sencond sigular
                                # value of B, think about it


print('second_sv : {}'.format(second_sv))       
plt.plot(np.arange(15), gy, c = 'r')
plt.xlabel('y')
plt.ylabel('gy')
#plt.savefig('result1.png')
    
