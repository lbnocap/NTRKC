import numpy as np   #rkc2定步长 Robertson equation
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev
import time
import copy
import math
import pandas as pd



#np.seterr(divide='ignore', invalid='ignore')
pi=math.pi
M=2000
time_st=time.time()
x0=0
x_end=100
x=np.linspace(x0,x_end,M+1,dtype=float)
hx=x[1]-x[0]
af=0.139
bt=2.54
yt=0.008
BB=np.zeros((2*(M-1),2*(M-1)))
A=np.zeros((M-1,M-1)) 
B=np.zeros((M-1,M-1)) 
y0=np.zeros((2*(M-1),1))

for i in range(0,M-1):
    if i==0: 
        A[0][0],A[0][1]=-2/(hx**2),1/(hx**2)

    elif 0<i<M-2:
    
        A[i][i-1],A[i][i],A[i][i+1]=1/(hx**2),-2/(hx**2),1/(hx**2)
      
    elif i==M-2:
        A[M-2][M-2],A[M-2][M-3]=-2/(hx**2),1/(hx**2)
    
       
BB[0:M-1,0:M-1],BB[M-1:2*(M-1),M-1:2*(M-1)]=A,B

N=2*(M-1)

def fun1(x,z):
    b=np.zeros((2*(M-1),1))
    u=z[0:M-1]
    v=z[M-1:2*(M-1)]
    for j in range(0,M-1):
        if j==0:
          b[j]=-u[j]*(u[j]-af)*(u[j]-1)-v[j]-0.3/(hx**2)
          b[M-1+j]=yt*(u[j]-bt*v[j])
      
        if 0<j<M-2:
             b[j]=-u[j]*(u[j]-af)*(u[j]-1)-v[j]
             b[M-1+j]=yt*(u[j]-bt*v[j])
        elif j==M-2:
             b[j]=-u[j]*(u[j]-af)*(u[j]-1)-v[j]
             b[M-1+j]=yt*(u[j]-bt*v[j])

    b=b.reshape((2*(M-1),1))
    U=np.dot(BB,z).reshape((2*(M-1),1))

    return U+b



def err(x,y,tc,h):
    x1=x.reshape((2*M,1))
    y1=y.reshape((2*M,1))
    z1=12*(x1-y1)
    return 0.1*(z1+6*h*(fun1(tc+h,x1)+fun1(tc+h,y1)))



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
            s2=math.sqrt(h1*pu/0.40)
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
            s2=np.sqrt(h1*pu/0.40)                                           
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
    return np.array(tc),np.array(y),np.array(y2),nfe,s_max


t0=0
t_end=2.5
for i in range(1):
   h=0.05
   eig3,fg1=ro(0,y0)
   print('eig:',eig3)
   eig1,abcd=np.linalg.eig(A)
   eig2=np.max(np.abs(eig1))
   print('eig2:',eig2)
   s2=np.sqrt(h*eig3/0.40)                                           
   s=int(s2)
   f=fun1(0,y0)
   #print(f)
   if s<=3:
      s=3
   tc,y,y2,nfe,s_max=RKC(fun1,t0,t_end,h,y0,s)
   print(tc)
   #err2=err(y,y2,0,h)
   #print(solu)
   #err1=np.linalg.norm(err2)/math.sqrt(3*M)
   #print("err1:",err1)
   solu1=np.load('FHNM2000solu_0.000011v1.npy')
   err=sum([(x - y) ** 2 for x,y in zip(y[0:2*M,-1], solu1[0:2*M])] )/ len(solu1[0:2*M])
   print("err:",np.sqrt(err))
   err3=sum([(x - y) ** 2 for x, y in zip(y2[0:2*M,-1], solu1[0:2*M])] )/ len(solu1[0:2*M])
   print("err3:",np.sqrt(err3))
   #obsolu=y.ravel())
   print("NTRKCv3")
   print("h;",h)
   print("bt:",bt)
   print("af:",af)
   print("yt",yt)

   print('nfe:',nfe)
  #obsolu=y.ravel()
  #plt.show()