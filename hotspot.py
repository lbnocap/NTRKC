import numpy as np
def hotspot(bt,M,hx):
    A=np.zeros((M**2,M**2))
    B=np.zeros((M,M))
    y=np.ones((M**2,1))
    E=np.eye(M)

    for i in range(M):
      if i==0:
        B[0][0],B[0][1]=-3,1
      elif 0<i<M-1:
        B[i][i-1],B[i][i],B[i][i+1]=1,-4,1
      elif i==M-1:
        B[i][i-1],B[i][i]=1,-4
    for j in range(M):
     if j==0:
           A[0:M,0:M],A[0:M,M:2*M]=B+E,E
     elif 0<i<M-1:
          A[j*(M):(j+1)*(M),(j-1)*(M):(j)*(M)]=E
          A[j*(M):(j+1)*(M),(j)*(M):(j+1)*(M)]=B
          A[j*(M):(j+1)*(M),(j+1)*(M):(j+2)*(M)]=E
     elif  i==M-1:
          A[j*(M):(j+1)*(M),(j-1)*(M):(j)*(M)]=E
          A[j*(M):(j+1)*(M),(j)*(M):(j+1)*(M)]=B
    return bt*A/(hx**2),y
         
      