import numpy as np  
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev
import time
import copy
import math
import pandas as pd



np.seterr(divide='ignore', invalid='ignore')
pi=math.pi
M=1
time_st=time.time()
t0=0
t_end=30
y0=np.zeros((3*M,1))
y0[0]=1
y0[1]=2
y0[2]=3

def fun1(t,z):
    b=np.zeros((3*M,1))
    b[0]=77.27*(z[1]+z[0]*(1-8.375*(1e-06)*z[0]-z[1]))
    b[1]=(z[2]-(1+z[0])*z[1])/77.27
    b[2]=0.161*(z[0]-z[2])

    return b

N=3*M

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




RKCv2= np.load(r'C:\Users\A204-7\Desktop\RKC\RKC\RKC2v1.npz', allow_pickle=True)
cs = RKCv2['cs']
us1=RKCv2['us1']
vs1=RKCv2['vs1']
vs=RKCv2['vs']
us=RKCv2['us']



def RKC(fun1,t0,t_end,h,u0,s,MM): 
    h1=h
    tc=[t0] #t的初始
    #solu=np.zeros((MM,1))
    solu=[1]
    y=u0
    counter=0
    fg1=0
    nfe=0
    s_max=0
    y2=np.zeros((N,1))
    
    ii=1
    while tc[-1]<t_end:
        c=cs[s,0]
        u1=us1[s,0]
        u=us[s,0]
        v1=vs1[s,0]
        v=vs[s,0]
       
        nfe=s+nfe+fg1+3
        k0=np.zeros((N,1))
        k1=np.zeros((N,1))
        k2=np.zeros((N,1))
        k3=np.zeros((N,1))
        ky0=np.zeros((N,1))
        ky1=np.zeros((N,1))
        k0=y[:,-1].copy()
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
        y2=y.copy()
        y =yc.copy()
        counter=counter+1
        tc.append(tc[-1]+h1)
        solu.append(float(y[0].copy()))
        pu,fg1=ro(tc[-1]+h1,yc)
        if tc[-1] + h1 > t_end:
                 h1 = t_end -tc[-1]
                 h=h1
        s2=math.sqrt(h1*pu/0.55)
        s=math.ceil(s2)
       # solu[ii]=y[1]
        ii=ii+1
        if s_max<s:
                   s_max=s
        if s<2:
                    s=2
        if s>250:
                s=250  
    
         
            
    return np.array(tc),np.array(y),np.array(y2),nfe,s_max,solu
t0=0
t_end=30
h=0.00011
eig3,fg1=ro(0,y0)
#eig1,abcd=np.linalg.eig(A)
#eig2=np.max(np.abs(eig1))
s2 = math.sqrt(h * eig3 / 0.55)
s = math.ceil(s2)
print(s)
#print('eig:',eig2)
#print(fun1(0,y))
#print(y)
if s<=1:
    s=2
#tc1,y1,nfe1,s_max1=RKC2(fun1,t0,t_end,0.0001,y,s)
MM=np.ceil(30/h)
MM=int(MM)
tc,y,y2,nfe,s_max,solu=RKC(fun1,t0,t_end,h,y0,s,MM+1)
time_end=time.time()
print(time_end-time_st)
print(tc)
print("步数：",len(tc))
print("评估次数：",nfe)
print("s_max:",s_max)
tc=np.array(tc)
#print(solu)
#solu=np.array(solu)
#err=sum([(x - y) ** 2 for x, y in zip(y[0:M-1,-1], solu[0:M-1])] )/ len(solu[0:M-1])
#print("err:",np.sqrt(err))
plt.plot(tc, solu,'red')
#plt.plot(x[1:M], solu[0:M-1],'blue')
#plt.title(' t=2 af=0.1 beta=0.05  numberical solutions of RKC')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.legend()
Robsolu=y.ravel()
Robsolu1=y2.ravel()
solu1=np.load('OREGOsolut_end30v1.npy')
err=sum([(x - y) ** 2 for x, y in zip(y, solu1)] )/ len(solu1)
err3=sum([(x - y2) ** 2 for x, y2 in zip(y2, solu1)] )/ len(solu1)
print("err:",np.sqrt(err))             
print("err3:",np.sqrt(err3))
#df = pd.DataFrame({'OREGOsolut_end30v1': Robsolu})
#df1 = pd.DataFrame({'OREGOsolut_end30v2': Robsolu1})

#np.save('OREGOsolut_end30v1.npy',Robsolu)
#np.save('OREGOsolut_end30v2.npy',Robsolu1)
plt.show()