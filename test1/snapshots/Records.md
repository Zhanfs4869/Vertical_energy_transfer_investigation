# Experiment Records
## Movie1: 
foring为`np.sin(2*np.pi*16*z+2*np.pi*16*x)+np.sin(..-..)+np.cos(..+..)+np.cos(..-..)`乘上`np.sin(100t)`(在ex方向)，`np.cos(100t)`(在ez方向)  
`Ro=0.01, Rey=300, Nfc=10000`
## Movie2: 
foring为白噪声滤波后结果，滤波核为
`k0=2*np.pi*16,dk=2*np.pi*2,ring_filter=np.exp(-((K-k0)**2)/(2*dk**2))`
，滤波后乘上`np.sin(100t)`(在ex方向与ez方向)，在时间上存在间歇性，每隔`rng.exponential(scale=0.01)`关闭或开启forcing  
`Ro=0.01, Rey=300, Nfc=10000`
## Movie3:
滤波核修改为
```python
k0 = 16
dk = 2
ring_filter = 0.5*(-np.tanh((KZ+0.95*k0)/dk)+np.tanh((KZ-0.95*k0)/dk)+2)*np.exp(-(np.abs(KZ)-k0)**2)/(2*dk**2)
ring_filter*= np.exp(-(np.abs(KX)-k0)**2)/(2*dk**2)
fc *= ring_filter
forcing['c'] = fc
fg = forcing['g']
rms = np.sqrt(np.mean(fg**2))
forcing['g'] *= 1 / (rms + 1e-30)
```
滤波后仍乘上`np.sin(100t)`(在ex方向与ez方向)，时间间歇性同上，无量纲数设置同上，实际计算的雷诺数大概是0.01，偏低

## Movie4(4_1与4_2)
滤波核与Movie3相同，但滤波后不再乘上`np.sin(100t)`，时间间歇性同上。  
`Ro=0.01, Rey=300, Nfc=10`

## Movie5
滤波核中`ring_filter`字段相较Movie3修改为`ring_filter = np.exp(-((K-k0)**2)/ (2*dk**2))`，其中`K=np.sqrt(KX**2+KZ**2)`,时间依赖性与无量纲参数设置同Movie4,实际计算的雷诺数大致为1的量级

## Movie8
`Ro=0.1, Rey=1000, Nfc=np.sqrt(1000), eta=1000`

## Movie9
`Ro=0.01, Rey=10000, Nfc=10, eta=10000`

## Movie10
新的控制方程，forcing的频谱里没有使用ap
```python
Lx, Lz = 1, 1         #The range of the box
Nx, Nz = 512, 128     #Number of grid points
Ro = 0.1
Rey = 3000
Nfc = np.sqrt(10000)
ap = np.sqrt(0.1)
ap2=(1/ap)**2
N2 = Nfc**2*ap**2*Ro
eta = 300
```



