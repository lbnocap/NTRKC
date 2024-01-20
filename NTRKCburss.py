# -*- coding: utf-8 -*-
import numpy as np  
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev
import time
import math
import burss
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')





M=128
pi=np.pi
time_st=time.time()
x0=0
x_end=1
x1=np.linspace(x0,x_end,M+1,dtype=float)
x=np.linspace(x1[1],x_end,M,dtype=float)
hx=x[1]-x[0]
bt=0.04
af=2
A1,A2,y0=burss.burss(af,bt,M,hx,x)
def fun1(x,z):
    b1=np.zeros((M**2,1))
    b2=np.zeros((M**2,1))
    e=np.ones((M**2,1))
    u=z[0:M**2].copy()
    v=z[M**2:2*M**2].copy()
    u=u.reshape((M**2,1))
    v=v.reshape((M**2,1))
    b1=-2*u+(u**2)*v+1.3*e
    b2=u-(u**2)*v
    b=np.concatenate((b1, b2))
    b=b.reshape((2*M**2,1))
    U1=np.dot(A1,u)
    U2=np.dot(A2,v) 
    U=np.concatenate((U1, U2))
  
    return U+b

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
    while fg > 1e-4*R and fg1<40:
        Rv1=Rv+e*(f1-f2)/np.linalg.norm(f1-f2)
        f1=fun1(x,Rv1)
        R=np.linalg.norm(f1-f2)/e
        fg=np.abs(R-Rr)
        fg1+=1
        Rr=R 
    if fg1==40:
        R=1.2*R
    return R,fg1


widetwoRKCv2 = np.load(r'C:\Users\A204-7\Desktop\RKC\RKC\widetwostepRKCv3.npz', allow_pickle=True)
cs = widetwoRKCv2['cs']
us1=widetwoRKCv2['us1']
vs1=widetwoRKCv2['vs1']
vs=widetwoRKCv2['vs']
us=widetwoRKCv2['us']
bs=widetwoRKCv2['bs']
xxs=widetwoRKCv2['xxs']

RKCv2= np.load(r'C:\Users\A204-7\Desktop\RKC\RKC\RKC2v1.npz', allow_pickle=True)
RKC2cs = RKCv2['cs']
RKC2us1=RKCv2['us1']
RKC2vs1=RKCv2['vs1']
RKC2vs=RKCv2['vs']
RKC2us=RKCv2['us']
N=2*M**2

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
    
    tc=[t0] #t的初始
    y=u0
    counter=0
    fg1=0
    nfe=0
    s_max=0
    h1=h
    lop=0
    yb0=np.zeros((N,1))
    y2=np.zeros((N,1))
    while tc[-1]<t_end:
        c=cs[s,0]
        u1=us1[s,0]
        u=us[s,0]
        v1=vs1[s,0]
        v=vs[s,0]
        xx=xxs[s,0]
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
            #if j==8:
              #  print(k3)
            ky1=fun1(tc[-1]+c[j]*h,k3)
            k1=k2.copy()
            k2=k3.copy()
        xx1=xx[0]
        xx2=xx[1]
        xx3=xx[2]
        xx4=xx[3]
            
        if counter==0:
            yc=RKC2(fun1,t0,h,u0,s)
           
           # print("yc:",yc)
          

            #err2=err(y[:,-1],yc,h1)
            #err1=np.linalg.norm(err2)/math.sqrt(3*M+3)
           # print(yb0)y, yc))
            yb0=k3.copy()
            
            counter+=1
            y2=y.copy()
            y=yc.copy()
            tc.append(tc[-1]+h1)
            pu,fg1=ro(tc[-1]+h1,yc)
            s2=math.sqrt(h1*pu/0.4)
            s=math.ceil(s2)
            if s_max<s:
                   s_max=s
            if s>250:
                s=250

            if s<3:
                s=3  
        else :
            k02=y2.copy()
            k02=k02.reshape((N,1))
            yb=k3.copy()    
            yc=xx1*k02+xx2*yb0+xx3*k0+xx4*yb
            yb0=yb.copy()
            pu,fg1=ro(tc[-1]+h1,yc)
            tc.append(tc[-1]+h1)
            if pu>lop:
                lop=pu
            if tc[-1] + h1 > t_end:
                 h1 = t_end -tc[-1]
                 h=h1  
            s2=np.sqrt(h1*pu/0.4)                                           
            s=math.ceil(s2)
            if s<3:
                s=3
            if s>s_max:
                s_max=s 
            if s>250:
                s=250
            #h=h1
            #err2=err(y[:,-1],yc,h1)
            #err1=np.linalg.norm(err2)/math.sqrt(3*M+3)
            #print(err1)
            y2=y.copy()
            y=yc.copy()
    return np.array(tc),np.array(y),np.array(y2),nfe,s_max,lop


    return np.array(tc),np.array(y),np.array(y2),nfe,s_max
t0=0
t_end=1
print("af:",af)
for i in range(1):
  h=0.051
  eig3,fg1=ro(0,y0)
  s2 = math.sqrt(h * eig3 / 0.4)
  s = math.ceil(s2)
  print(s)
  print('eig:',eig3)
#print(fun1(0,y))
#print(y)
  if s<=3:
     s=3
#tc1,y1,nfe1,s_max1=RKC2(fun1,t0,t_end,0.0001,y,s)
  tc,y,y2,nfe,s_max,lop=RKC(fun1,t0,t_end,h,y0,s)
  time_end=time.time()
  print(time_end-time_st)
  print(tc)
  print("步数：",len(tc))
  print("评估次数：",nfe)
  print("s_max:",s_max)
  print("lop:",lop)
  solu1=np.load('burssyt1.5solu0.0001v1.npy')
  err=sum([(x - y) ** 2 for x, y in zip(y, solu1)] )/ len(solu1)
  err3=sum([(x - y2) ** 2 for x, y2 in zip(y2, solu1)] )/ len(solu1)
  print("err:",np.sqrt(err))             
  print("err3:",np.sqrt(err3))  