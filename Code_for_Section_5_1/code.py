

import numpy as np
import random 
import matplotlib.pyplot as plt
import math


iter=800
N=20
N_row=500
N_col=100
n=100
unit=10


#The parameter tau is the local training period. It can be changed.

tau=2 
#tau=3
#tau=4
#tau=5
#tau=6

Result_my1=np.zeros(iter)
Result_my2=np.zeros(iter)
Result_my3=np.zeros(iter)
Result_my4=np.zeros(iter)
Result_my5=np.zeros(iter)
Result_my6=np.zeros(iter)
Result_my7=np.zeros(iter)

GG_my1=np.zeros(n)
GG_AVG_my1=np.zeros(n)
ave_X_my1=np.zeros(n)

GG_my2=np.zeros(n)
GG_AVG_my2=np.zeros(n)
ave_X_my2=np.zeros(n)

GG_my3=np.zeros(n)
GG_AVG_my3=np.zeros(n)
ave_X_my3=np.zeros(n)

GG_my4=np.zeros(n)
GG_AVG_my4=np.zeros(n)
ave_X_my4=np.zeros(n)

GG_my5=np.zeros(n)
GG_AVG_my5=np.zeros(n)
ave_X_my5=np.zeros(n)

GG_my6=np.zeros(n)
GG_AVG_my6=np.zeros(n)
ave_X_my6=np.zeros(n)

GG_my7=np.zeros(n)
GG_AVG_my7=np.zeros(n)
ave_X_my7=np.zeros(n)


X_my1=0*np.zeros((N,n))
X_my2=0*np.zeros((N,n))
X_my3=0*np.zeros((N,n))
X_my4=0*np.zeros((N,n))
X_my5=0*np.zeros((N,n))
X_my6=0*np.zeros((N,n))
X_my7=0*np.zeros((N,n))


A=np.zeros((N,N_row,N_col))
b=np.zeros((N,N_row))
C=np.zeros((N,n,n))
D=np.zeros((N,n))
for s1 in range(0,N):
    for s2 in range(0,N_row):
        for s3 in range(0,N_col):
            A[s1][s2][s3]=random.random()
                    
            
for i in range(0,N_row):
    A[0][i][1]=A[0][i][0]
    
    

x0=unit*np.ones(n)

for i in range(0,N):
    b[i]=np.dot(A[i],x0)
    
L=np.zeros(N)

step_size1=np.zeros(N)

for i in range(0,N):
    L[i]=np.linalg.norm(np.dot(np.transpose(A[i]),A[i]))
    step_size1[i]=(1/L[i])-math.pow(10,-15)

L_Max=L[0]
L_Min=L[0]
for i in range(0,N):
    if L[i]>L_Max:
        L_Max=L[i]
    if L[i]<L_Min:
        L_Min=L[i]
        
L_average=0
for i in range(0,N):
    L_average=L_average+L[i]
L_average=L_average/N
        

average_matrix=A[0]
for i in range(1,N):
    average_matrix=average_matrix+A[i]
    
average_matrix=average_matrix/N

sigma=L_Max/(np.linalg.norm(np.dot(np.transpose(average_matrix),average_matrix)))
    


step_size2=(8*tau)/(L_average*(2*tau+sigma*(tau-1))*(2*tau+sigma*(tau-1))+4*sigma*L_average*tau*(tau-1))

for i in range(0,N):
    sample=1/(tau*np.linalg.norm(np.dot(np.transpose(A[i]),A[i])))
    if step_size2>sample:
        step_size2=sample


    
step_size3=1/(2*L_Max)

step_size5=1/(4*L_Max)
    
step_size6=(2*tau)/(5*L_average*tau*tau-L_average*tau)-math.pow(10,-15)
if ((2*tau)/(5*L_average*tau*tau-L_average*tau))>(1/L_Max):
    step_size6=(1/L_Max)-math.pow(10,-15)
    
step_size7=1/(10*L_Max*tau)


Opt1=0*C[1]
Opt2=0*D[1]


c=2*tau*tau
gamma1=1/(2*L_Max*c)
a=2*L_Max*c
bBBB=20*tau*L_Max

if a<bBBB:
    gamma1=1/(20*tau*L_Max)

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
    print(obj(ave_X_my4)-obj(X_OPT))
    print(obj(ave_X_my5)-obj(X_OPT))
    print(obj(ave_X_my6)-obj(X_OPT))
    print(obj(ave_X_my7)-obj(X_OPT))
    
    Result_my1[t]=obj(ave_X_my1)-obj(X_OPT)
    Result_my2[t]=obj(ave_X_my2)-obj(X_OPT)
    Result_my3[t]=obj(ave_X_my3)-obj(X_OPT)
    Result_my4[t]=obj(ave_X_my4)-obj(X_OPT)
    Result_my5[t]=obj(ave_X_my5)-obj(X_OPT)
    Result_my6[t]=obj(ave_X_my6)-obj(X_OPT)
    Result_my7[t]=obj(ave_X_my7)-obj(X_OPT)
    
    
    
    for i in range(0,N):     
        X_my1[i]=ave_X_my1
        X_my2[i]=ave_X_my2
        X_my3[i]=ave_X_my3
        X_my4[i]=ave_X_my4
        X_my5[i]=ave_X_my5
        X_my6[i]=ave_X_my6
        X_my7[i]=ave_X_my7
        
        
        GG_FITST_my6=np.dot(np.dot(np.transpose(A[i]),A[i]),X_my6[i])-np.dot(np.transpose(A[i]),b[i])
        GG_FITST_my7=np.dot(np.dot(np.transpose(A[i]),A[i]),X_my7[i])-np.dot(np.transpose(A[i]),b[i])
        
        
        for k in range(0,tau):
            X_my1[i]=X_my1[i]-step_size1[i]*(np.dot(np.dot(np.transpose(A[i]),A[i]),X_my1[i])-np.dot(np.transpose(A[i]),b[i]))
            X_my2[i]=X_my2[i]-step_size2*(np.dot(np.dot(np.transpose(A[i]),A[i]),X_my2[i])-np.dot(np.transpose(A[i]),b[i]))
            X_my3[i]=X_my3[i]-step_size3*(np.dot(np.dot(np.transpose(A[i]),A[i]),X_my3[i])-np.dot(np.transpose(A[i]),b[i]))
            X_my5[i]=X_my5[i]-step_size5*(np.dot(np.dot(np.transpose(A[i]),A[i]),X_my5[i])-np.dot(np.transpose(A[i]),b[i]))
            X_my6[i]=X_my6[i]-step_size6*(GG_AVG_my6-GG_FITST_my6+np.dot(np.dot(np.transpose(A[i]),A[i]),X_my6[i])-np.dot(np.transpose(A[i]),b[i]))
            X_my7[i]=X_my7[i]-step_size7*(GG_AVG_my7-GG_FITST_my7+np.dot(np.dot(np.transpose(A[i]),A[i]),X_my7[i])-np.dot(np.transpose(A[i]),b[i]))
            gamma2=(np.linalg.norm(np.dot(A[i],X_my4[i])-b[i])*np.linalg.norm(np.dot(A[i],X_my4[i])-b[i]))/(2*c*(np.linalg.norm(np.dot(np.dot(np.transpose(A[i]),A[i]),X_my4[i])-np.dot(np.transpose(A[i]),b[i])))*(np.linalg.norm(np.dot(np.dot(np.transpose(A[i]),A[i]),X_my4[i])-np.dot(np.transpose(A[i]),b[i]))))
            gamma=gamma1
            if gamma2<gamma1:
                gamma=gamma2
            X_my4[i]=X_my4[i]-gamma*(np.dot(np.dot(np.transpose(A[i]),A[i]),X_my4[i])-np.dot(np.transpose(A[i]),b[i]))
                 
    ave_X_my1=np.zeros(n)
    ave_X_my2=np.zeros(n)
    ave_X_my3=np.zeros(n)
    ave_X_my4=np.zeros(n)
    ave_X_my5=np.zeros(n)
    ave_X_my6=np.zeros(n)
    ave_X_my7=np.zeros(n)
   
    for j in range(0,N):
        ave_X_my1=ave_X_my1+X_my1[j]
        ave_X_my2=ave_X_my2+X_my2[j]
        ave_X_my3=ave_X_my3+X_my3[j]
        ave_X_my4=ave_X_my4+X_my4[j]
        ave_X_my5=ave_X_my5+X_my5[j]
        ave_X_my6=ave_X_my6+X_my6[j]
        ave_X_my7=ave_X_my7+X_my7[j]
        
    ave_X_my1=ave_X_my1/N
    ave_X_my2=ave_X_my2/N
    ave_X_my3=ave_X_my3/N
    ave_X_my4=ave_X_my4/N
    ave_X_my5=ave_X_my5/N
    ave_X_my6=ave_X_my6/N
    ave_X_my7=ave_X_my7/N
    
    GG_my6=np.zeros(n)
    GG_my7=np.zeros(n)
    
    for ii in range(0,N):       
        GG_my6=GG_my6+np.dot(np.dot(np.transpose(A[ii]),A[ii]),ave_X_my6)-np.dot(np.transpose(A[ii]),b[ii])
        GG_my7=GG_my7+np.dot(np.dot(np.transpose(A[ii]),A[ii]),ave_X_my7)-np.dot(np.transpose(A[ii]),b[ii])
        
    GG_AVG_my6=GG_my6/N
    GG_AVG_my7=GG_my7/N
    

#For different parameters tau, the name of the txt file should be changed accordingly.

#Reference_41 corresponds to Qin et al. (2022); 
#Reference_24 corresponds to Mukherjee et al. (2023);
#Reference_44 corresponds to Khaled et al. (2020); 
#Reference_38 corresponds to Mitra et al. (2021b); 

#For tau=2:
np.savetxt('paper_algorithm1_(5)_TAU2.txt',Result_my1,fmt="%f",delimiter=" ") 
np.savetxt('paper_algorithm1_(4)_TAU2.txt',Result_my2,fmt="%f",delimiter=" ")
np.savetxt('Reference_41_TAU2.txt',Result_my3,fmt="%f",delimiter=" ")
np.savetxt('Reference_24_TAU2.txt',Result_my4,fmt="%f",delimiter=" ") 
np.savetxt('Reference_44_TAU2.txt',Result_my5,fmt="%f",delimiter=" ")
np.savetxt('paper_algorithm2_TAU2.txt',Result_my6,fmt="%f",delimiter=" ") 
np.savetxt('Reference_38_TAU2.txt',Result_my7,fmt="%f",delimiter=" ") 


#For tau=3:
#np.savetxt('paper_algorithm1_(5)_TAU3.txt',Result_my1,fmt="%f",delimiter=" ") 
#np.savetxt('paper_algorithm1_(4)_TAU3.txt',Result_my2,fmt="%f",delimiter=" ")
#np.savetxt('Reference_41_TAU3.txt',Result_my3,fmt="%f",delimiter=" ")
#np.savetxt('Reference_24_TAU3.txt',Result_my4,fmt="%f",delimiter=" ") 
#np.savetxt('Reference_44_TAU3.txt',Result_my5,fmt="%f",delimiter=" ")
#np.savetxt('paper_algorithm2_TAU3.txt',Result_my6,fmt="%f",delimiter=" ") 
#np.savetxt('Reference_38_TAU3.txt',Result_my7,fmt="%f",delimiter=" ") 

#For tau=4:
#np.savetxt('paper_algorithm1_(5)_TAU4.txt',Result_my1,fmt="%f",delimiter=" ") 
#np.savetxt('paper_algorithm1_(4)_TAU4.txt',Result_my2,fmt="%f",delimiter=" ")
#np.savetxt('Reference_41_TAU4.txt',Result_my3,fmt="%f",delimiter=" ")
#np.savetxt('Reference_24_TAU4.txt',Result_my4,fmt="%f",delimiter=" ") 
#np.savetxt('Reference_44_TAU4.txt',Result_my5,fmt="%f",delimiter=" ")
#np.savetxt('paper_algorithm2_TAU4.txt',Result_my6,fmt="%f",delimiter=" ") 
#np.savetxt('Reference_38_TAU4.txt',Result_my7,fmt="%f",delimiter=" ") 

#For tau=5:
#np.savetxt('paper_algorithm1_(5)_TAU5.txt',Result_my1,fmt="%f",delimiter=" ") 
#np.savetxt('paper_algorithm1_(4)_TAU5.txt',Result_my2,fmt="%f",delimiter=" ")
#np.savetxt('Reference_41_TAU5.txt',Result_my3,fmt="%f",delimiter=" ")
#np.savetxt('Reference_24_TAU5.txt',Result_my4,fmt="%f",delimiter=" ") 
#np.savetxt('Reference_44_TAU5.txt',Result_my5,fmt="%f",delimiter=" ")
#np.savetxt('paper_algorithm2_TAU5.txt',Result_my6,fmt="%f",delimiter=" ") 
#np.savetxt('Reference_38_TAU5.txt',Result_my7,fmt="%f",delimiter=" ") 


#For tau=6:
#np.savetxt('paper_algorithm1_(5)_TAU6.txt',Result_my1,fmt="%f",delimiter=" ") 
#np.savetxt('paper_algorithm1_(4)_TAU6.txt',Result_my2,fmt="%f",delimiter=" ")
#np.savetxt('Reference_41_TAU6.txt',Result_my3,fmt="%f",delimiter=" ")
#np.savetxt('Reference_24_TAU6.txt',Result_my4,fmt="%f",delimiter=" ") 
#np.savetxt('Reference_44_TAU6.txt',Result_my5,fmt="%f",delimiter=" ")
#np.savetxt('paper_algorithm2_TAU6.txt',Result_my6,fmt="%f",delimiter=" ") 
#np.savetxt('Reference_38_TAU6.txt',Result_my7,fmt="%f",delimiter=" ") 