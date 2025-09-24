
import numpy as np
import random 
import matplotlib.pyplot as plt
import math


iter=1500
N=20
N_row=500
N_col=100
n=100

#The parameter tau is the local training period. It can be changed.

#tau=2 
#tau=3
#tau=4
#tau=5
#tau=6
tau=7


beta=np.zeros(iter)

Result_my1=np.zeros(iter)
Result_my2=np.zeros(iter)
Result_my3=np.zeros(iter)
Result_my4=np.zeros(iter)

GG_my1=np.zeros(n)
GG_AVG_my1=np.zeros(n)
ave_X_my1=np.zeros(n)

GG_my2=np.zeros(n)
GG_AVG_my2=np.zeros(n)
ave_X_my2=np.zeros(n)

GG_my3=np.zeros(n)
GG_AVG_my3=np.zeros(n)
ave_X_my3=np.zeros(n)



X_my1=np.zeros((N,n))
X_my2=np.zeros((N,n))
X_my3=np.zeros((N,n))


A=np.zeros((N,N_row,N_col))
b=np.zeros((N,N_row))
C=np.zeros((N,n,n))
D=np.zeros((N,n))
for s1 in range(0,N):
    for s2 in range(0,N_row):
        b[s1][s2]=random.random()
        for s3 in range(0,N_col):
            A[s1][s2][s3]=random.random()
            
for i in range(0,N_row):
    A[0][i][1]=A[0][i][0]
    
L=np.zeros(N)
for i in range(0,N):
    L[i]=np.linalg.norm(np.dot(np.transpose(A[i]),A[i]))
    
L_max=L[0]
for i in range(0,N):
    if L[i]>L_max:
        L_max=L[i]

L_average=0
for i in range(0,N):
    L_average=L_average+np.linalg.norm(np.dot(np.transpose(A[i]),A[i]))
    
L_average=L_average/N
step_size_example=(2*tau)/(5*L_average*tau*tau-L_average*tau)

step_size1=step_size_example-math.pow(10,-15)
for i in range(0,N):
    Sample=1/L[i]
    if Sample<step_size_example:
        step_size1=Sample-math.pow(10,-15)
    

step_size2=1/(10*L_max*tau)

step_size3=1/(18*L_max*tau)


Opt1=0*C[1]
Opt2=0*D[1]


for sss in range(0,N):
    C[sss]=np.dot(np.transpose(A[sss]),A[sss])
    Opt1=Opt1+C[sss]
    D[sss]=np.dot(np.transpose(A[sss]),b[sss])
    Opt2=Opt2+D[sss]
    
X_OPT=np.dot(np.linalg.inv(Opt1),Opt2)

def obj(XX):
    result=0
    for p in range(0,N):
        result=result+np.linalg.norm(np.dot(A[p],XX)-b[p])*np.linalg.norm(np.dot(A[p],XX)-b[p])
    final_result=result/(2*N)
    return final_result

ave_X_my=np.zeros(n)
for t in range(0,iter):
    print(t)
    print(obj(ave_X_my1)-obj(X_OPT))
    print(obj(ave_X_my2)-obj(X_OPT))
    print(obj(ave_X_my3)-obj(X_OPT))
    
    Result_my1[t]=obj(ave_X_my1)-obj(X_OPT)
    Result_my2[t]=obj(ave_X_my2)-obj(X_OPT)
    Result_my3[t]=obj(ave_X_my3)-obj(X_OPT)
    
    for i in range(0,N):     
        X_my1[i]=ave_X_my1
        X_my2[i]=ave_X_my2
        X_my3[i]=ave_X_my3
        
        GG_FITST_my1=np.dot(np.dot(np.transpose(A[i]),A[i]),X_my1[i])-np.dot(np.transpose(A[i]),b[i])
        GG_FITST_my2=np.dot(np.dot(np.transpose(A[i]),A[i]),X_my2[i])-np.dot(np.transpose(A[i]),b[i])
        GG_FITST_my3=np.dot(np.dot(np.transpose(A[i]),A[i]),X_my3[i])-np.dot(np.transpose(A[i]),b[i])
        
        for k in range(0,tau):
            X_my1[i]=X_my1[i]-step_size1*(GG_AVG_my1-GG_FITST_my1+np.dot(np.dot(np.transpose(A[i]),A[i]),X_my1[i])-np.dot(np.transpose(A[i]),b[i]))
            X_my2[i]=X_my2[i]-step_size2*(GG_AVG_my2-GG_FITST_my2+np.dot(np.dot(np.transpose(A[i]),A[i]),X_my2[i])-np.dot(np.transpose(A[i]),b[i]))
            X_my3[i]=X_my3[i]-step_size3*(GG_AVG_my3-GG_FITST_my3+np.dot(np.dot(np.transpose(A[i]),A[i]),X_my3[i])-np.dot(np.transpose(A[i]),b[i]))
            
            
    ave_X_my1=np.zeros(n)
    GG_my1=np.zeros(n)
    ave_X_my2=np.zeros(n)
    GG_my2=np.zeros(n)
    ave_X_my3=np.zeros(n)
    GG_my3=np.zeros(n)
    
    for j in range(0,N):
        ave_X_my1=ave_X_my1+X_my1[j]
        ave_X_my2=ave_X_my2+X_my2[j]
        ave_X_my3=ave_X_my3+X_my3[j]
        
    ave_X_my1=ave_X_my1/N
    ave_X_my2=ave_X_my2/N
    ave_X_my3=ave_X_my3/N
    
    for ii in range(0,N):       
        GG_my1=GG_my1+np.dot(np.dot(np.transpose(A[ii]),A[ii]),ave_X_my1)-np.dot(np.transpose(A[ii]),b[ii])
        GG_my2=GG_my2+np.dot(np.dot(np.transpose(A[ii]),A[ii]),ave_X_my2)-np.dot(np.transpose(A[ii]),b[ii])
        GG_my3=GG_my3+np.dot(np.dot(np.transpose(A[ii]),A[ii]),ave_X_my3)-np.dot(np.transpose(A[ii]),b[ii])
        
    GG_AVG_my1=GG_my1/N
    GG_AVG_my2=GG_my2/N
    GG_AVG_my3=GG_my3/N



#Reference_38 corresponds to Mitra et al. (2021b); 
#Reference_42 corresponds to Mitra et al. (2021a);


#For different parameters tau, the name of the txt file should be changed accordingly.

#For tau=2:
#np.savetxt('paper_algorithm2_TAU2.txt',Result_my1,fmt="%f",delimiter=" ") 
#np.savetxt('Reference_38_TAU2.txt',Result_my2,fmt="%f",delimiter=" ")
#np.savetxt('Reference_42_TAU2.txt',Result_my3,fmt="%f",delimiter=" ")

#For tau=3:
#np.savetxt('paper_algorithm2_TAU3.txt',Result_my1,fmt="%f",delimiter=" ") 
#np.savetxt('Reference_38_TAU3.txt',Result_my2,fmt="%f",delimiter=" ")
#np.savetxt('Reference_42_TAU3.txt',Result_my3,fmt="%f",delimiter=" ")

#For tau=4:
#np.savetxt('paper_algorithm2_TAU4.txt',Result_my1,fmt="%f",delimiter=" ") 
#np.savetxt('Reference_38_TAU4.txt',Result_my2,fmt="%f",delimiter=" ")
#np.savetxt('Reference_42_TAU4.txt',Result_my3,fmt="%f",delimiter=" ")

#For tau=5:
#np.savetxt('paper_algorithm2_TAU5.txt',Result_my1,fmt="%f",delimiter=" ") 
#np.savetxt('Reference_38_TAU5.txt',Result_my2,fmt="%f",delimiter=" ")
#np.savetxt('Reference_42_TAU5.txt',Result_my3,fmt="%f",delimiter=" ")

#For tau=6:
#np.savetxt('paper_algorithm2_TAU6.txt',Result_my1,fmt="%f",delimiter=" ") 
#np.savetxt('Reference_38_TAU6.txt',Result_my2,fmt="%f",delimiter=" ")
#np.savetxt('Reference_42_TAU6.txt',Result_my3,fmt="%f",delimiter=" ")

#For tau=7:
np.savetxt('paper_algorithm2_TAU7.txt',Result_my1,fmt="%f",delimiter=" ") 
np.savetxt('Reference_38_TAU7.txt',Result_my2,fmt="%f",delimiter=" ")
np.savetxt('Reference_42_TAU7.txt',Result_my3,fmt="%f",delimiter=" ")
