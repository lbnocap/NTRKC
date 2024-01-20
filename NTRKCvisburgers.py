import numpy as np   #rkc2定步长 Robertson equation
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev
import time
import copy
import math
import pandas as pd



np.seterr(divide='ignore', invalid='ignore')
pi=math.pi
M=1500
time_st=time.time()
x0=-pi
x_end=pi
x=np.linspace(x0,x_end,M+1,dtype=float)
hx=x[1]-x[0]
yt=0.2
t0=0


t_end=1
A=np.zeros((M,M)) 
B1=np.zeros((M,M))
B2=np.zeros((M,M)) 
solu=np.zeros((M,1))
y=np.zeros((M,1))
B3=np.zeros((M,M))
for i in range(0,M):
    if i==0: 
        B1[0][0],B1[0][1]=-yt*2/(hx**2),yt/(hx**2)
        B2[0][0]=-1/(hx)
        B3[0][0]=-1/(hx)
        y[i]=np.sin(x[i+1])
        solu[i]=np.exp(-yt*t_end)*np.sin(x[i+1])
    if i==1:
         B1[i][i-1],B1[i][i],B1[i][i+1]=yt/(hx**2),-2*yt/(hx**2),yt/(hx**2)
         B2[i][i-1],B2[i][i]=2/hx,-3/(2*hx)
         B3[i][i],B3[i][i-1]=-1/hx,1/hx
         y[i]=np.sin(x[i+1])
         solu[i]=np.exp(-yt*t_end)*np.sin(x[i+1])

    elif 1<i<M-1:
    
        B1[i][i-1],B1[i][i],B1[i][i+1]=yt/(hx**2),-2*yt/(hx**2),yt/(hx**2)
        B2[i][i-1],B2[i][i]=2/hx,-3/(2*hx)
        B2[i][i-2]=-1/(2*hx)
        B3[i][i],B3[i][i-1]=-1/hx,1/hx
        y[i]=np.sin(x[i+1])
        solu[i]=np.exp(-yt*t_end)*np.sin(x[i+1])
      
    elif i==M-1:
        B1[M-1][M-1],B1[M-1][M-2]=-2*yt/(hx**2),yt/(hx**2)
        B1[M-1][0]=yt/(hx**2)
        B2[i][i-1],B2[i][i]=2/hx,-(3)/(2*hx)
        B2[i][i-2]=-1/(2*hx)
        B3[i][i],B3[i][i-1]=-1/hx,1/hx
        y[i]=np.sin(x[i+1])
        solu[i]=np.exp(-yt*t_end)*np.sin(x[i+1])
    
       

def g(t,x):
    return np.exp(-2*yt*t)*np.sin(2*x)/2
def fun1(t,z):

    b=np.dot(B3,z).reshape((M,1))
    b1=b*z
    #b1=b1.reshape((M,1))
    U=np.dot(B1,z).reshape((M,1))
    b2=np.zeros((M,1))
    for i in range(0,M):
        b2[i]=g(t,x[i+1])
    return U+b1+b2



def err(x,y,tc,h):
    x1=x.reshape((M-1,1))
    y1=y.reshape((M-1,1))
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
        R=1.1*R
    return R,fg1




def ro(x,y):
    e=1e-8;ln=len(y)
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
        R=1.1*R
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
        k0=np.zeros((M,1))
        k1=np.zeros((M,1))
        k2=np.zeros((M,1))
        k3=np.zeros((M,1))
        ky0=np.zeros((M,1))
        ky1=np.zeros((M,1))
        k0=y1[:,-1].copy()
        k0=k0.reshape((M,1))
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
    yb0=np.zeros((M,1))
    y2=np.zeros((M,1))
    while tc[-1]<t_end:
        c=cs[s,0]
        u1=us1[s,0]
        u=us[s,0]
        v1=vs1[s,0]
        v=vs[s,0]
        xx=xxs[s,0]
        nfe=s+nfe+fg1+3
        k0=np.zeros((M,1))
        k1=np.zeros((M,1))
        k2=np.zeros((M,1))
        k3=np.zeros((M,1))
        ky0=np.zeros((M,1))
        ky1=np.zeros((M,1))
        k0=y.copy()
        k0=k0.reshape((M,1))
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
            k02=k02.reshape((M,1))
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



h=1/(20*4)
eig3,fg1=ro(0,y)
#print('eig:',eig3)
#eig1,abcd=np.linalg.eig(A)
#eig2=np.max(np.abs(eig1))
print('eig2:',eig3)
s2=np.sqrt(h*eig3/0.42)                                           
s=math.ceil(s2)
f=fun1(0,y)
#print(f)
if s<=5:
    s=5
tc,y,y2,nfe,s_max,lop=RKC(fun1,t0,t_end,h,y,s)
#err=sum([(x - y) ** 2 for x, y in zip(y[1:M,-1], solu[1:M])] )/ len(solu[1:M])
print(tc)
print("NTRKC")
print("h;",h)
print("lop:",lop)
print("s:",s)
#print("err:",np.sqrt(err))
print("nfe:",nfe)
solu1=np.load('buegerssolu_0.00001.npy')
err=sum([(x - y) ** 2 for x, y in zip(y[0:M-1,-1], solu[0:M-1])] )/ len(solu[0:M-1])
err3=sum([(x - y2) ** 2 for x, y2 in zip(y2[0:M-1,-1], solu[0:M-1])] )/ len(solu[0:M-1])
print("err:",np.sqrt(err))
print("er3:",np.sqrt(err3))
plt.plot(x[1:M], y[0:M-1,-1],'red')
plt.plot(x[1:M], solu[0:M-1],'blue')
#plt.title(' t=2 af=0.1 beta=0.05  numberical solutions of RKC')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.legend()
plt.show()
