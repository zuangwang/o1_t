
import numpy as np
import random 
import matplotlib.pyplot as plt
import math


iter=6000
N=20
N_row=500
N_col=100
n=100
tau=1
rho=5
rho1=1
rho2=2
rho3=3
rho4=4
rho5=5
unit=10

parameter=np.ones(rho)
parameter[0]=1
parameter[1]=1.5
parameter[2]=2
parameter[3]=2.5
parameter[4]=3


unit_vector=unit*np.ones(n)

B=np.zeros((N,N_row,N_col))
for s1 in range(0,N):
    for s2 in range(0,N_row):
        for s3 in range(0,N_col):
            B[s1][s2][s3]=random.random()


beta=np.zeros(iter)

Result_my1=np.zeros((rho,iter))
Result_my2=np.zeros((rho,iter))
Result_my3=np.zeros((rho,iter))


ave_X_my1=np.zeros((rho,n))
ave_X_my2=np.zeros((rho,n))

X_my1=0*np.zeros((rho,N,n))
X_my2=0*np.zeros((rho,N,n))

A=np.zeros((rho,N,N_row,N_col))
b=np.zeros((rho,N,N_row))
C=np.zeros((rho,N,n,n))
D=np.zeros((rho,N,n))


for k in range(0,rho):
    for s1 in range(0,N):
        for s2 in range(0,N_row):
            for s3 in range(0,N_col):
                A[k][s1][s2][s3]=math.pow(s1+1,parameter[k])*B[s1][s2][s3]
    for sa1 in range(0,N):
        b[k][sa1]=np.dot(A[k][sa1],unit_vector)
    
L=np.zeros((rho,N))
step_size1=np.zeros((rho,N))
step_size2=np.zeros(rho)
L_average=np.zeros(rho)

for k in range(0,rho):
    for s in range(0,N):
        L[k][s]=np.linalg.norm(np.dot(np.transpose(A[k][s]),A[k][s]))
        step_size1[k][s]=(1/L[k][s])-math.pow(10,-15)
    
    L_average[k]=0
    for i in range(0,N):
        L_average[k]=L_average[k]+np.linalg.norm(np.dot(np.transpose(A[k][i]),A[k][i]))
    L_average[k]=L_average[k]/N

    step_size2[k]=1/L_average[k]
    

C=np.zeros((rho,N,n,n))
D=np.zeros((rho,N,n))

Opt1=np.zeros((rho,n,n))
Opt2=np.zeros((rho,n))

X_OPT=np.zeros((rho,n)) 



for k in range(0,rho):
    for s in range(0,N):
        C[k][s]=np.dot(np.transpose(A[k][s]),A[k][s])
        Opt1[k]=Opt1[k]+C[k][s]
        D[k][s]=np.dot(np.transpose(A[k][s]),b[k][s])
        Opt2[k]=Opt2[k]+D[k][s]
        X_OPT[k]=np.dot(np.linalg.inv(Opt1[k]),Opt2[k])

def obj(A,b,XX):
    result=0
    for p in range(0,N):
        result=result+np.linalg.norm(np.dot(A[p],XX)-b[p])*np.linalg.norm(np.dot(A[p],XX)-b[p])
    final_result=result/(2*N)
    return final_result

ave_X_my=np.zeros((iter,n))

for k in range(0,rho):
    obj_timely=obj(A[k],b[k],X_OPT[k])
    for t in range(0,iter):
        print(t)
        Result_my1[k][t]=obj(A[k],b[k],ave_X_my1[k])-obj_timely
        Result_my2[k][t]=obj(A[k],b[k],ave_X_my2[k])-obj_timely
        Result_my3[k][t]=Result_my1[k][t]/Result_my2[k][t]
    
        print(Result_my1[k][t])
        print(Result_my2[k][t])
        print(Result_my3[k][t])
    
        for i in range(0,N):     
            X_my1[k][i]=ave_X_my1[k]
            X_my2[k][i]=ave_X_my2[k]
        
            for kk in range(0,tau):
                X_my1[k][i]=X_my1[k][i]-step_size1[k][i]*(np.dot(np.dot(np.transpose(A[k][i]),A[k][i]),X_my1[k][i])-np.dot(np.transpose(A[k][i]),b[k][i]))
                X_my2[k][i]=X_my2[k][i]-step_size2[k]*(np.dot(np.dot(np.transpose(A[k][i]),A[k][i]),X_my2[k][i])-np.dot(np.transpose(A[k][i]),b[k][i]))
            
            
        ave_X_my1[k]=np.zeros(n)
        ave_X_my2[k]=np.zeros(n)
    
    
        for j in range(0,N):
            ave_X_my1[k]=ave_X_my1[k]+X_my1[k][j]
            ave_X_my2[k]=ave_X_my2[k]+X_my2[k][j]
        
        ave_X_my1[k]=ave_X_my1[k]/N
        ave_X_my2[k]=ave_X_my2[k]/N
    
    
np.savetxt('paper_algorithm1_local_rho1.txt',Result_my1[0],fmt="%f",delimiter=" ") 
np.savetxt('paper_algorithm1_global_rho1.txt',Result_my2[0],fmt="%f",delimiter=" ")
np.savetxt('paper_compare_rho1.txt',Result_my3[0],fmt="%f",delimiter=" ")

np.savetxt('paper_algorithm1_local_rho2.txt',Result_my1[1],fmt="%f",delimiter=" ") 
np.savetxt('paper_algorithm1_global_rho2.txt',Result_my2[1],fmt="%f",delimiter=" ")
np.savetxt('paper_compare_rho2.txt',Result_my3[1],fmt="%f",delimiter=" ")
    

np.savetxt('paper_algorithm1_local_rho3.txt',Result_my1[2],fmt="%f",delimiter=" ") 
np.savetxt('paper_algorithm1_global_rho3.txt',Result_my2[2],fmt="%f",delimiter=" ")
np.savetxt('paper_compare_rho3.txt',Result_my3[2],fmt="%f",delimiter=" ")


np.savetxt('paper_algorithm1_local_rho4.txt',Result_my1[3],fmt="%f",delimiter=" ") 
np.savetxt('paper_algorithm1_global_rho4.txt',Result_my2[3],fmt="%f",delimiter=" ")
np.savetxt('paper_compare_rho4.txt',Result_my3[3],fmt="%f",delimiter=" ")


np.savetxt('paper_algorithm1_local_rho5.txt',Result_my1[4],fmt="%f",delimiter=" ") 
np.savetxt('paper_algorithm1_global_rho5.txt',Result_my2[4],fmt="%f",delimiter=" ")
np.savetxt('paper_compare_rho5.txt',Result_my3[4],fmt="%f",delimiter=" ")

