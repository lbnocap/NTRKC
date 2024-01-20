# -*- coding: utf-8 -*-
import numpy as np  
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev
import time
import math
import pandas as pd
import  hotspot
np.seterr(divide='ignore', invalid='ignore')





M=128
pi=np.pi
time_st=time.time()
x0=0
x_end=1
x=np.linspace(x0,x_end,M+2,dtype=float)
hx=x[1]-x[0]
bt=2.5
af=1
R=5
segam=20

def generate_column_vector(M):
    # 生成 M^2 x 1 的零向量
    vector = np.zeros((M**2, 1))

    # 设置最后 M 个元素为 1
    vector[-M:] = 1

    return vector


          
def generate_column_vector2(M):
    # 生成 M^2 x 1 的零向量
    vector = np.zeros((M**2, 1))

    # 设置每第 M 个元素为 1
    vector[M-1::M] = 1

    return vector
A,y=hotspot.hotspot(bt,M,hx)
def fun1(x,z):
    b1=bt*generate_column_vector(M)/(hx**2)
    b2=bt*generate_column_vector2(M)/(hx**2)
    b3=np.zeros((M**2,1))
    for i in range(M**2):
         b3[i]=R*(1+af-z[i])*np.exp(segam*(1-1/z[i]))/(af*segam)
   
    U=np.dot(A,z).reshape((M**2,1))
  
    return U+b1+b2+b3

print(A)

def ro(x,y):
    e=1e-12;ln=len(y)
    Rv=y.copy()
    for j in range(ln):
        if y[j]==0:
            Rv[j]=e/2
        else:
            Rv[j]=y[j]*(1+e/2)
    e=max(e,e*np.linalg.norm(Rv,ord=2))
    Rv1=y.copy()
    f1=fun1(x,Rv1) 
    f2=fun1(x,Rv)
    Rv1=Rv+e*(f1-f2)/(np.linalg.norm(f1-f2))
    Rv1=Rv1.reshape((ln,1))
    f1=fun1(x,Rv1)
    R=np.linalg.norm(f1-f2)/e
    Rr=R
    fg=R;fg1=0
    while fg > 1e-4*R and fg1<20:
        Rv1=Rv+e*(f1-f2)/np.linalg.norm(f1-f2)
        f1=fun1(x,Rv1)
        R=np.linalg.norm(f1-f2)/e
        fg=np.abs(R-Rr)
        fg1+=1
        Rr=R 
    if fg1==20:
        R=1.2*R
    return R,fg1

RKCv2= np.load(r'C:\Users\A204-7\Desktop\RKC\RKC\twostepRKCv1.npz', allow_pickle=True)
cs = RKCv2['cs']
us1=RKCv2['us1']
vs1=RKCv2['vs1']
vs=RKCv2['vs']
us=RKCv2['us']
yts=RKCv2['yts']

RKCv24= np.load(r'C:\Users\A204-7\Desktop\RKC\RKC\RKC2.npz', allow_pickle=True)
RKC2cs = RKCv24['cs']
RKC2us1=RKCv24['us1']
RKC2vs1=RKCv24['vs1']
RKC2vs=RKCv24['vs']
RKC2us=RKCv24['us']

N=M**2

def RKC2(fun1,t0,h,u0,s): 
    h1=h
    tc=[t0] #t的初始
    y1=u0 
    counter1=0
    while counter1==0:
        c=RKC2cs[s,0]
        u1=RKC2us1[s,0]
        u=RKC2us[s,0]
        v1=RKC2vs1[s,0]
        v=RKC2vs[s,0]
        k0=np.zeros((N,1))
        k1=np.zeros((N,1))
        k2=np.zeros((N,1))
        k3=np.zeros((N,1))
        ky0=np.zeros((N,1))
        ky1=np.zeros((N,1))
        k0=y1[:,-1].copy()
        k0=k0.reshape((N,1))
        ky0=fun1(tc[-1],k0)
        k1=k0+u1[1] *h *ky0
        ky1=fun1(tc[-1]+u1[1]*h,k1) 
        k2=k1.copy()
        k1=k0.copy()
        for j in range(2,s+1):

            k3=u[j]*k2+v[j]*k1+(1-u[j]-v[j])*k0+u1[j]*h*ky1+v1[j]*h*ky0
            #if j==4:
                #print(k[4])
            ky1=fun1(tc[-1]+c[j]*h,k3)
            k1=k2.copy()
            k2=k3.copy()
      
        
        yc=k3.copy()
            #err2=err(y[:,-1],yc,tc[-1],h1)
            #err1=np.linalg.norm(err2)/math.sqrt(2*M+2)
            #print(err1)
            # fac=0.8*((1/err1)**(1/3))
        counter1=1
    return yc



def RKC(fun1,t0,t_end,h,u0,s): 
    h1=h
    tc=[t0] #t的初始
    y=u0
    counter=0
    fg1=0
    nfe=0
    s_max=0
    y2=np.zeros((N,1))
    while tc[-1]<t_end:
        c=cs[s,0]
        u1=us1[s,0]
        u=us[s,0]
        v1=vs1[s,0]
        v=vs[s,0]
        yt=yts[s,0]
        if counter==1:
            h=yt*h1
        nfe=s+nfe+fg1+3
        k0=np.zeros((N,1))
        k1=np.zeros((N,1))
        k2=np.zeros((N,1))
        k3=np.zeros((N,1))
        ky0=np.zeros((N,1))
        ky1=np.zeros((N,1))
        
        k0=y.copy()
        k0=k0.reshape((N,1))
        ky0=fun1(tc[-1],k0)
        k1=k0+u1[1] *h *ky0
        ky1=fun1(tc[-1]+u1[1]*h,k1)
    
        k2=k1.copy()
        k1=k0.copy()
        for j in range(2,s+1):

            k3=u[j]*k2+v[j]*k1+(1-u[j]-v[j])*k0+u1[j]*h*ky1+v1[j]*h*ky0
            #if j==4:
                #print(k[4])
            ky1=fun1(tc[-1]+c[j]*h,k3)
            k1=k2.copy()
            k2=k3.copy()
        r=1
        bn=(1+r)/(yt*(1+r*yt))
        bf1=(r**2)*(1-yt)/(1+yt*r)
        b0=1-bn-bf1
        
        if counter==0:
            #yc=RKC2(fun1,t0,h,u0,s)
            yc=k3.copy()
            # print(yc)
            #err2=err(y[:,-1],yc,tc[-1],h1)
            #err1=np.linalg.norm(err2)/math.sqrt(2*M+2)
            #print(err1)
            # fac=0.8*((1/err1)**(1/3))
            y2=y.copy()
            y=yc.copy()
            
            counter=1
            tc.append(tc[-1]+h1)
            pu,fg1=ro(tc[-1]+h1,yc)
            s2=math.sqrt(h1*pu/0.4)
            s=math.ceil(s2)
            if s_max<s:
                   s_max=s
            if s<3:
                    s=3
            if s>250:
                s=250
            
         
            


           
        else :
            k02=y2.copy()
            k02=k02.reshape((N,1))
            yb=k3.copy()
        
            yc=bf1*k02+b0*k0+bn*yb
            pu,fg1=ro(tc[-1]+h1,yc)
            tc.append(tc[-1]+h1) 
            if tc[-1] + h1 > t_end:
                 h1 = t_end -tc[-1]
                
            s2=np.sqrt(h1*pu/0.4)                                           
            s=math.ceil(s2)
            if s<3:
                s=3
            if s>s_max:
                s_max=s
            if s>250:
                s=250  
            #h=h1
            y2=y.copy()
            y=yc.copy()

    return np.array(tc),np.array(y),np.array(y2),nfe,s_max
t0=0
t_end=1.55
h=0.00001
eig3,fg1=ro(0,y)
s2=np.sqrt(h*eig3/0.4)                                           
s=math.ceil(s2)
print(s)
print('eig:',eig3)
#print(fun1(x,y))
#print(y)
if s<=3:
    s=3
#tc1,y1,nfe1,s_max1=RKC2(fun1,t0,t_end,0.0001,y,s)
tc,y,y2,nfe,s_max=RKC(fun1,t0,t_end,h,y,s)
#mse = np.mean((np.array(y[1:M,-1]) - np.array(solu[1:M]))**2)
#mae = np.mean(np.abs(np.array(y[1:M,-1]) - np.array(solu[1:M])
# ))
#err=sum([(x - y) ** 2 for x, y in zip(y[:,-2], y1[:,-2])] )/ len(y1[:,-2])
#print("terrpr:",np.sqrt(err))
time_end=time.time()
print(time_end-time_st)
print("twos")
print("h;",h)
print("bt:",bt)
print(tc)
print("步数：",len(tc))
print("评估次数：",nfe)
print("s_max:",s_max)

#print(y[:,3])

solu1=np.load('hotspotsolu0.000001.npy')
err=sum([(x - y) ** 2 for x, y in zip(y[0:3*M,-1], solu1[0:3*M])] )/ len(solu1[0:3*M])
print("err:",np.sqrt(err))
