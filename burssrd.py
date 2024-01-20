import numpy as np  
     
def burss(bt,M,hx,x):
        u01=np.zeros((M**2,1))
        v01=np.zeros((M**2,1))
        A=np.zeros(((M)**2,(M)**2))
        B=np.zeros((M,M))
        E=np.eye(M)
        AA=np.zeros((2*(M**2),2*(M**2)))
        g=np.zeros((M**2,1))
        for i in range(0,M):
            if i==0:
                for j in range(0,M):
                
                  u01[j]=22*x[i]*((1-x[i])**(3/2))
                  v01[j]=27*x[j]*((1-x[j])**(3/2))
                  if (x[j]-0.3)**2+(x[i]-0.6)**2<=0.01:
                     g[(i*M)+j]=5

                B[0][0],B[0][1]=-4,1
                B[0][M-1]=1
              

            elif i==1:
                for j in range(0,M):
                  if j==0:
                    u01[(i*M)+j]=22*x[i]*((1-x[i])**(3/2))
                    v01[(i*M)+j]=27*x[j]*((1-x[j])**(3/2))
                  if  j>0: 
                    u01[(i*M)+j]=22*x[i]*((1-x[i])**(3/2))
                    v01[(i*M)+j]=27*x[j]*((1-x[j])**(3/2))
                  if (x[j]-0.3)**2+(x[i]-0.6)**2<=0.01:
                     g[(i*M)+j]=5
                B[i][i],B[i][i+1]=-4,1
                B[i][i-1]=1
            
            elif 1<i<M-1:
                for j in range(0,M):
                  if j==0:
                    u01[(i*M)+j]=22*x[i]*((1-x[i])**(3/2))
                    v01[(i*M)+j]=27*x[j]*((1-x[j])**(3/2))
                  if  j>0: 
                    u01[(i*M)+j]=22*x[i]*((1-x[i])**(3/2))
                    v01[(i*M)+j]=27*x[j]*((1-x[j])**(3/2))
                  if (x[j]-0.3)**2+(x[i]-0.6)**2<=0.01:
                     g[(i*M)+j]=5
                B[i][i],B[i][i+1]=-4,1
                B[i][i-1]=1
            
            elif i==M-1:
                for j in range(0,M):
                   if j==0:
                    u01[(i*M)+j]=22*x[i]*((1-x[i])**(3/2))
                    v01[(i*M)+j]=27*x[j]*((1-x[j])**(3/2))
                   if  j>0: 
                    u01[(i*M)+j]=22*x[i]*((1-x[i])**(3/2))
                    v01[(i*M)+j]=27*x[j]*((1-x[j])**(3/2))
                   if (x[j]-0.3)**2+(x[i]-0.6)**2<=0.01:
                     g[(i*M)+j]=5
                B[i][i],B[i][0]=-4,1
                B[i][i-1]=1
              
        


        for j in range(0,M):
         if j==0:
            A[0:M,0:M],A[0:M,M:2*(M)]=B,E
            A[0:M,(M-1)*M:M*M]=E
      
         elif j==1:
            A[j*(M):(j+1)*(M),(j-1)*(M):(j)*(M)]=E
            A[j*(M):(j+1)*(M),(j)*(M):(j+1)*(M)]=B
            A[j*(M):(j+1)*(M),(j+1)*(M):(j+2)*(M)]=E
      

            
         elif 1<j<M-1:
            A[j*(M):(j+1)*(M),(j-1)*(M):(j)*(M)]=E
            A[j*(M):(j+1)*(M),(j)*(M):(j+1)*(M)]=B
            A[j*(M):(j+1)*(M),(j+1)*(M):(j+2)*(M)]=E
  
         elif j==M-1:
            A[j*(M):(j+1)*(M),(j-1)*(M):(j)*(M)]=E
            A[j*(M):(j+1)*(M),(j)*(M):(j+1)*(M)]=B
            A[j*(M):(j+1)*(M),(0)*(M):(1)*(M)]=E
        AA[0:M**2,0:M**2]=A*bt/(hx**2)
        AA[M**2:2*M**2,M**2:2*M**2]=A*bt/(hx**2)
        B=B*bt/(hx**2)
        A=A*bt/(hx**2)
        y=np.concatenate((u01, v01))
        return A,B,E,y,g