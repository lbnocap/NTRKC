import numpy as np  

import sys
if sys.maxsize >2**32:
    print(64)
else:
    print(32)















''' 
def burss(af,bt,M,hx,x):
        u01=np.zeros((M**2,1))
        v01=np.zeros((M**2,1))
        A=np.zeros(((M)**2,(M)**2))
        A1=np.zeros(((M)**2,(M)**2))
        A2=np.zeros(((M)**2,(M)**2))
        B=np.zeros((M,M))
        B1=np.zeros((M,M))
        B2=np.zeros((M,M))
        B3=np.zeros((M,M))
        C1=np.zeros((M,M))
        C2=np.zeros((M,M))
        D1=np.zeros((M,M))
        D2=np.zeros((M,M))
        D3=np.zeros((M,M))
        G1=np.zeros((M,M))
        G2=np.zeros((M,M))
        E=np.eye(M)
        AA=np.zeros((2*(M**2),2*(M**2)))
        for i in range(0,M):
            if i==0:
                for j in range(0,M):
                
                  u01[j]=22*x[i]*((1-x[i])**(3/2))
                  v01[j]=27*x[j]*((1-x[j])**(3/2))

                B[0][0],B[0][1]=-4,1
                B[0][M-1]=1
                B1[0][0],B1[0][M-1]=1/2,1/2
                B2[0][0]=-1
                C1[0][0],C1[0][M-1]=1/2,1/2
                C2[0][0]=-1
                D1[0][0],D1[0][M-1]=1.1,-2/5
                D2[0][0]=-0.7
                G1[0][0],G1[0][M-1]=1.1,-2/5
                G2[0][0]=-0.7

            elif i==1:
                for j in range(0,M):
                  if j==0:
                    u01[(i*M)+j]=22*x[i]*((1-x[i])**(3/2))
                    v01[(i*M)+j]=27*x[j]*((1-x[j])**(3/2))
                  if  j>0: 
                    u01[(i*M)+j]=22*x[i]*((1-x[i])**(3/2))
                    v01[(i*M)+j]=27*x[j]*((1-x[j])**(3/2))
                B[i][i],B[i][i+1]=-4,1
                B[i][i-1]=1
                B1[i][i],B1[i][i-1]=3/4,1
                B1[i][M-1]=-1
                B2[i][i]=-2
                B3[i][i]=1/2
                C1[i][i],C1[i][i-1]=1/2,1/2
                C2[i][i]=-1
                D1[i][i],D1[i][i-1]=1.65,-4/5
                D1[i][M-1]=1/5
                D2[i][i]=-1.4
                D3[i][i]=7/20
                G1[i][i],G1[i][i-1]=1.1,-2/5
                G2[i][i]=-0.7
            elif 1<i<M-1:
                for j in range(0,M):
                  if j==0:
                    u01[(i*M)+j]=22*x[i]*((1-x[i])**(3/2))
                    v01[(i*M)+j]=27*x[j]*((1-x[j])**(3/2))
                  if  j>0: 
                    u01[(i*M)+j]=22*x[i]*((1-x[i])**(3/2))
                    v01[(i*M)+j]=27*x[j]*((1-x[j])**(3/2))
                B[i][i],B[i][i+1]=-4,1
                B[i][i-1]=1
                B1[i][i],B1[i][i-1]=3/4,1
                B1[i][i-2]=-1
                B2[i][i]=-2
                B3[i][i]=1/2
                C1[i][i],C1[i][i-1]=1/2,1/2
                C2[i][i]=-1
                D1[i][i],D1[i][i-1]=1.65,-4/5
                D1[i][i-2]=1/5
                D2[i][i]=-1.4
                D3[i][i]=7/20
                G1[i][i],G1[i][i-1]=1.1,-2/5
                G2[i][i]=-0.7
            elif i==M-1:
                for j in range(0,M):
                   if j==0:
                    u01[(i*M)+j]=22*x[i]*((1-x[i])**(3/2))
                    v01[(i*M)+j]=27*x[j]*((1-x[j])**(3/2))
                   if  j>0: 
                    u01[(i*M)+j]=22*x[i]*((1-x[i])**(3/2))
                    v01[(i*M)+j]=27*x[j]*((1-x[j])**(3/2))
                B[i][i],B[i][0]=-4,1
                B[i][i-1]=1
                B1[i][i],B1[i][i-1]=3/4,1
                B1[i][i-2]=-1
                B2[i][i]=-2
                B3[i][i]=1/2
                C1[i][i],C1[i][i-1]=1/2,1/2
                C2[i][i]=-1
                D1[i][i],D1[i][i-1]=1.65,-4/5
                D1[i][i-2]=1/5
                D2[i][i]=-1.4
                D3[i][i]=7/20
                G1[i][i],G1[i][i-1]=1.1,-2/5
                G2[i][i]=-0.7
        


        for j in range(0,M):
         if j==0:
            A[0:M,0:M],A[0:M,M:2*(M)]=B,E
            A[0:M,(M-1)*M:M*M]=E
            A1[0:M,0:M],A1[0:M,(M-1)*M:M*M]=C1,C2
            A2[0:M,0:M],A2[0:M,(M-1)*M:M*M]=G1,G2
         elif j==1:
            A[j*(M):(j+1)*(M),(j-1)*(M):(j)*(M)]=E
            A[j*(M):(j+1)*(M),(j)*(M):(j+1)*(M)]=B
            A[j*(M):(j+1)*(M),(j+1)*(M):(j+2)*(M)]=E
            A1[j*(M):(j+1)*(M),(j-1)*(M):(j)*(M)]=B2
            A1[j*(M):(j+1)*(M),(j)*(M):(j+1)*(M)]=B1
            A1[j*(M):(j+1)*(M),(M-1)*(M):(M)*(M)]=B3
            A2[j*(M):(j+1)*(M),(j-1)*(M):(j)*(M)]=D2
            A2[j*(M):(j+1)*(M),(j)*(M):(j+1)*(M)]=D1
            A2[j*(M):(j+1)*(M),(M-1)*(M):(M)*(M)]=D3

            
         elif 1<j<M-1:
            A[j*(M):(j+1)*(M),(j-1)*(M):(j)*(M)]=E
            A[j*(M):(j+1)*(M),(j)*(M):(j+1)*(M)]=B
            A[j*(M):(j+1)*(M),(j+1)*(M):(j+2)*(M)]=E
            A1[j*(M):(j+1)*(M),(j-1)*(M):(j)*(M)]=B2
            A1[j*(M):(j+1)*(M),(j)*(M):(j+1)*(M)]=B1
            A1[j*(M):(j+1)*(M),(j-2)*(M):(j-1)*(M)]=B3
            A2[j*(M):(j+1)*(M),(j-1)*(M):(j)*(M)]=D2
            A2[j*(M):(j+1)*(M),(j)*(M):(j+1)*(M)]=D1
            A2[j*(M):(j+1)*(M),(j-2)*(M):(j-1)*(M)]=D3
         elif j==M-1:
            A[j*(M):(j+1)*(M),(j-1)*(M):(j)*(M)]=E
            A[j*(M):(j+1)*(M),(j)*(M):(j+1)*(M)]=B
            A[j*(M):(j+1)*(M),(0)*(M):(1)*(M)]=E
            A1[j*(M):(j+1)*(M),(j-1)*(M):(j)*(M)]=B2
            A1[j*(M):(j+1)*(M),(j)*(M):(j+1)*(M)]=B1
            A1[j*(M):(j+1)*(M),(j-2)*(M):(j-1)*(M)]=B3
            A2[j*(M):(j+1)*(M),(j-1)*(M):(j)*(M)]=D2
            A2[j*(M):(j+1)*(M),(j)*(M):(j+1)*(M)]=D1
            A2[j*(M):(j+1)*(M),(j-2)*(M):(j-1)*(M)]=D3
        AA[0:M**2,0:M**2]=A*bt/(hx**2)+A1*af/hx
        AA[M**2:2*M**2,M**2:2*M**2]=A*bt/(hx**2)+A2*af/hx
        y=np.concatenate((u01, v01))
        return AA,y
        '''