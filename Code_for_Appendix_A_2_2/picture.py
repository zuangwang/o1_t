

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
def readFile(path):
    f = open(path)
    first_ele = True
    for data in f.readlines():
        data = data.strip('\n')
        nums = data.split(" ")
        if first_ele:
            nums = [float(x) for x in nums ]
            matrix = np.array(nums)
            first_ele = False
        else:
            nums = [float(x) for x in nums ]
            matrix = np.c_[matrix,nums]
    return matrix
    f.close()


if __name__ == '__main__':
    
    #Next
    
    D1_1=np.transpose(readFile('paper_algorithm1_local_rho1.txt'))
    D2_1=np.transpose(readFile('paper_algorithm1_global_rho1.txt'))
    
    k_1=4999
    kk_1=2500
    
    BASE1_1=np.zeros(k_1)
    BASE2_1=np.zeros(k_1)
    BASE3_1=np.zeros(kk_1)
    BASE4_1=np.zeros(kk_1)
    
    BASE5_1=np.zeros(k_1)
    BASE6_1=np.zeros(k_1)
    BASE7_1=np.zeros(kk_1)
    BASE8_1=np.zeros(kk_1)
    
    for j in range(0,k_1-1):
        BASE1_1[j]=j
        BASE2_1[j]=D1_1[j+1]
        
        
    for j in range(0,k_1-1):
        BASE5_1[j]=j
        BASE6_1[j]=D2_1[j+1]
        
    k0_1=500
    
    for jj in range(0,kk_1):
        BASE3_1[jj]=BASE1_1[k0_1+jj]
        BASE4_1[jj]=BASE2_1[k0_1+jj]
         
    for jj in range(0,kk_1):
        BASE7_1[jj]=BASE5_1[k0_1+jj]
        BASE8_1[jj]=BASE6_1[k0_1+jj]
          
    x_1 = BASE3_1
    y_1 = BASE4_1
    
    x2_1=BASE7_1
    y2_1=BASE8_1
    
    #Next
    
    D1_2=np.transpose(readFile('paper_algorithm1_local_rho2.txt'))
    D2_2=np.transpose(readFile('paper_algorithm1_global_rho2.txt'))
    
    k_2=4999
    kk_2=2500
    
    BASE1_2=np.zeros(k_2)
    BASE2_2=np.zeros(k_2)
    BASE3_2=np.zeros(kk_2)
    BASE4_2=np.zeros(kk_2)
    
    BASE5_2=np.zeros(k_2)
    BASE6_2=np.zeros(k_2)
    BASE7_2=np.zeros(kk_2)
    BASE8_2=np.zeros(kk_2)
    
    for j in range(0,k_2-1):
        BASE1_2[j]=j
        BASE2_2[j]=D1_2[j+1]
        
        
    for j in range(0,k_2-1):
        BASE5_2[j]=j
        BASE6_2[j]=D2_2[j+1]
        
    k0_2=1000
    
    for jj in range(0,kk_2):
        BASE3_2[jj]=BASE1_2[k0_2+jj]
        BASE4_2[jj]=BASE2_2[k0_2+jj]
         
    for jj in range(0,kk_2):
        BASE7_2[jj]=BASE5_2[k0_2+jj]
        BASE8_2[jj]=BASE6_2[k0_2+jj]
          
    x_2 = BASE3_2
    y_2 = BASE4_2
    
    x2_2=BASE7_2
    y2_2=BASE8_2
    
    #Next
    
    D1_3=np.transpose(readFile('paper_algorithm1_local_rho3.txt'))
    D2_3=np.transpose(readFile('paper_algorithm1_global_rho3.txt'))
    
    k_3=4999
    kk_3=2500
    
    BASE1_3=np.zeros(k_3)
    BASE2_3=np.zeros(k_3)
    BASE3_3=np.zeros(kk_3)
    BASE4_3=np.zeros(kk_3)
    
    BASE5_3=np.zeros(k_3)
    BASE6_3=np.zeros(k_3)
    BASE7_3=np.zeros(kk_3)
    BASE8_3=np.zeros(kk_3)
    
    for j in range(0,k_3-1):
        BASE1_3[j]=j
        BASE2_3[j]=D1_3[j+1]
        
        
    for j in range(0,k_3-1):
        BASE5_3[j]=j
        BASE6_3[j]=D2_3[j+1]
        
    k0_3=1500
    
    for jj in range(0,kk_3):
        BASE3_3[jj]=BASE1_3[k0_3+jj]
        BASE4_3[jj]=BASE2_3[k0_3+jj]
         
    for jj in range(0,kk_3):
        BASE7_3[jj]=BASE5_3[k0_3+jj]
        BASE8_3[jj]=BASE6_3[k0_3+jj]
          
    x_3 = BASE3_3
    y_3 = BASE4_3
    
    x2_3=BASE7_3
    y2_3=BASE8_3
    
    #Next
    
    D1_4=np.transpose(readFile('paper_algorithm1_local_rho4.txt'))
    D2_4=np.transpose(readFile('paper_algorithm1_global_rho4.txt'))
    
    k_4=4999
    kk_4=2500
    
    BASE1_4=np.zeros(k_4)
    BASE2_4=np.zeros(k_4)
    BASE3_4=np.zeros(kk_4)
    BASE4_4=np.zeros(kk_4)
    
    BASE5_4=np.zeros(k_4)
    BASE6_4=np.zeros(k_4)
    BASE7_4=np.zeros(kk_4)
    BASE8_4=np.zeros(kk_4)
    
    for j in range(0,k_4-1):
        BASE1_4[j]=j
        BASE2_4[j]=D1_4[j+1]
        
        
    for j in range(0,k_4-1):
        BASE5_4[j]=j
        BASE6_4[j]=D2_4[j+1]
        
    k0_4=2000
    
    for jj in range(0,kk_4):
        BASE3_4[jj]=BASE1_4[k0_4+jj]
        BASE4_4[jj]=BASE2_4[k0_4+jj]
         
    for jj in range(0,kk_4):
        BASE7_4[jj]=BASE5_4[k0_4+jj]
        BASE8_4[jj]=BASE6_4[k0_4+jj]
          
    x_4 = BASE3_4
    y_4 = BASE4_4
    
    x2_4=BASE7_4
    y2_4=BASE8_4
    
    #Next
    
    D1_5=np.transpose(readFile('paper_algorithm1_local_rho5.txt'))
    D2_5=np.transpose(readFile('paper_algorithm1_global_rho5.txt'))
    
    k_5=4999
    kk_5=2500
    
    BASE1_5=np.zeros(k_5)
    BASE2_5=np.zeros(k_5)
    BASE3_5=np.zeros(kk_5)
    BASE4_5=np.zeros(kk_5)
    
    BASE5_5=np.zeros(k_5)
    BASE6_5=np.zeros(k_5)
    BASE7_5=np.zeros(kk_5)
    BASE8_5=np.zeros(kk_5)
    
    for j in range(0,k_5-1):
        BASE1_5[j]=j
        BASE2_5[j]=D1_5[j+1]
        
        
    for j in range(0,k_5-1):
        BASE5_5[j]=j
        BASE6_5[j]=D2_5[j+1]
        
    k0_5=2490
    
    for jj in range(0,kk_5):
        BASE3_5[jj]=BASE1_5[k0_5+jj]
        BASE4_5[jj]=BASE2_5[k0_5+jj]
         
    for jj in range(0,kk_5):
        BASE7_5[jj]=BASE5_5[k0_5+jj]
        BASE8_5[jj]=BASE6_5[k0_5+jj]
          
    x_5 = BASE3_5
    y_5 = BASE4_5
    
    x2_5=BASE7_5
    y2_5=BASE8_5
    
    #Next
    
    D1=np.transpose(readFile('paper_compare_rho1.txt'))
    D2=np.transpose(readFile('paper_compare_rho2.txt'))
    D3=np.transpose(readFile('paper_compare_rho3.txt'))
    D4=np.transpose(readFile('paper_compare_rho4.txt'))
    D5=np.transpose(readFile('paper_compare_rho5.txt'))
    
    k=4999
    kk=4490
    
    BASE1=np.zeros(k)
    BASE2=np.zeros(k)
    BASE3=np.zeros(kk)
    BASE4=np.zeros(kk)
    
    BASE5=np.zeros(k)
    BASE6=np.zeros(k)
    BASE7=np.zeros(kk)
    BASE8=np.zeros(kk)
    
    BASE9=np.zeros(k)
    BASE10=np.zeros(k)
    BASE11=np.zeros(kk)
    BASE12=np.zeros(kk)
    
    BASE13=np.zeros(k)
    BASE14=np.zeros(k)
    BASE15=np.zeros(kk)
    BASE16=np.zeros(kk)
    
    BASE17=np.zeros(k)
    BASE18=np.zeros(k)
    BASE19=np.zeros(kk)
    BASE20=np.zeros(kk)
    
    for j in range(0,k-1):
        BASE1[j]=j
        BASE2[j]=D1[j+1]
        
    for j in range(0,k-1):
        BASE5[j]=j
        BASE6[j]=D2[j+1]
        
    for j in range(0,k-1):
        BASE9[j]=j
        BASE10[j]=D3[j+1]
    
    for j in range(0,k-1):
        BASE13[j]=j
        BASE14[j]=D4[j+1]
        
    for j in range(0,k-1):
        BASE17[j]=j
        BASE18[j]=D5[j+1]
        
    k0=500
    
    for jj in range(0,kk):
        BASE3[jj]=BASE1[k0+jj]
        BASE4[jj]=BASE2[k0+jj]
        
    for jj in range(0,kk):
        BASE7[jj]=BASE5[k0+jj]
        BASE8[jj]=BASE6[k0+jj]
        
    for jj in range(0,kk):
        BASE11[jj]=BASE9[k0+jj]
        BASE12[jj]=BASE10[k0+jj]
        
    for jj in range(0,kk):
        BASE15[jj]=BASE13[k0+jj]
        BASE16[jj]=BASE14[k0+jj]
        
    for jj in range(0,kk):
        BASE19[jj]=BASE17[k0+jj]
        BASE20[jj]=BASE18[k0+jj]
        
    x = BASE3
    y = BASE4
    
    x2=BASE7
    y2=BASE8
    
    x3=BASE11
    y3=BASE12
    
    x4=BASE15
    y4=BASE16
    
    x5=BASE19
    y5=BASE20
    
    
    #Next
    
    plt.figure(figsize=(35,15))
   
    f1=plt.subplot(2, 3, 1)
    
    plt.semilogy(x2_1,y2_1,label='Global Stepsize',color='r',linewidth=1.9,linestyle=':')
    plt.semilogy(x_1,y_1,label='Local Stepsize',color='b',linewidth=1.9,linestyle='-.')
    plt.legend(loc = 0, prop = {'size':25})
    plt.grid()
   # f1.set_title(r'$\rho=1$',fontsize=25,y=-0.2)
    
    plt.xticks(fontsize = 18, fontname = 'times new roman')
    plt.yticks(fontsize = 18, fontname = 'times new roman')
    plt.xlabel(r'Communication Rounds $t$ ($\rho=1$)',fontsize=25)
    plt.ylabel(r'Errors',fontsize=25)
    
    f2=plt.subplot(2, 3, 2)
    
    plt.semilogy(x2_2,y2_2,label='Global Stepsize',color='r',linewidth=1.9,linestyle=':')
    plt.semilogy(x_2,y_2,label='Local Stepsize',color='b',linewidth=1.9,linestyle='-.')
    plt.legend(loc = 0, prop = {'size':25})
    plt.grid()
   # f2.set_title(r'$\rho=1.5$',fontsize=25,y=-0.2)
    
    plt.xticks(fontsize = 18, fontname = 'times new roman')
    plt.yticks(fontsize = 18, fontname = 'times new roman')
    plt.xlabel(r'Communication Rounds $t$ ($\rho=1.5$)',fontsize=25)
    plt.ylabel(r'Errors',fontsize=25)
    
    f3=plt.subplot(2, 3, 3)
    
    plt.semilogy(x2_3,y2_3,label='Global Stepsize',color='r',linewidth=1.9,linestyle=':')
    plt.semilogy(x_3,y_3,label='Local Stepsize',color='b',linewidth=1.9,linestyle='-.')
    plt.legend(loc = 0, prop = {'size':25})
    plt.grid()
  #  f3.set_title(r'$\rho=2$',fontsize=25,y=-0.2)
    
    plt.xticks(fontsize = 18, fontname = 'times new roman')
    plt.yticks(fontsize = 18, fontname = 'times new roman')
    plt.xlabel(r'Communication Rounds $t$ ($\rho=2$)',fontsize=25)
    plt.ylabel(r'Errors',fontsize=25)
    
    f4=plt.subplot(2, 3, 4)
    
    plt.semilogy(x2_4,y2_4,label='Global Stepsize',color='r',linewidth=1.9,linestyle=':')
    plt.semilogy(x_4,y_4,label='Local Stepsize',color='b',linewidth=1.9,linestyle='-.')
    plt.legend(loc = 0, prop = {'size':25})
    plt.grid()
  #  f4.set_title(r'$\rho=2.5$',fontsize=25,y=-0.2)
    
    plt.xticks(fontsize = 18, fontname = 'times new roman')
    plt.yticks(fontsize = 18, fontname = 'times new roman')
    plt.xlabel(r'Communication Rounds $t$ ($\rho=2.5$)',fontsize=25)
    plt.ylabel(r'Errors',fontsize=25)
    
    f5=plt.subplot(2, 3, 5)
    
    plt.semilogy(x2_5,y2_5,label='Global Stepsize',color='r',linewidth=1.9,linestyle=':')
    plt.semilogy(x_5,y_5,label='Local Stepsize',color='b',linewidth=1.9,linestyle='-.')
    plt.legend(loc = 0, prop = {'size':25})
    plt.grid()
 #   f5.set_title(r'$\rho=3$',fontsize=25,y=-0.2)
    
    plt.xticks(fontsize = 18, fontname = 'times new roman')
    plt.yticks(fontsize = 18, fontname = 'times new roman')
    plt.xlabel(r'Communication Rounds $t$ ($\rho=3$)',fontsize=25)
    plt.ylabel(r'Errors',fontsize=25)
    
    f6=plt.subplot(2, 3, 6)
    
    plt.plot(x,y,label=r'$\rho=1$',color='y',linewidth=1.9,linestyle='-.')
    plt.plot(x2,y2,label=r'$\rho=1.5$',color='g',linewidth=1.9,linestyle=':')
    plt.plot(x3,y3,label=r'$\rho=2$',color='r',linewidth=1.9,linestyle='--')
    plt.plot(x4,y4,label=r'$\rho=2.5$',color='c',linewidth=1.9,linestyle='-')
    plt.plot(x5,y5,label=r'$\rho=3$',color='b',linewidth=1.9,linestyle='solid')
    
    plt.legend(loc = 0, prop = {'size':25})
    plt.grid()
    f6.set_title('Ratio Comparisons',fontsize=25,y=-0.2)
    
    plt.xticks(fontsize = 18, fontname = 'times new roman')
    plt.yticks(fontsize = 18, fontname = 'times new roman')
    plt.xlabel(r'Communication Rounds $t$',fontsize=25)
    plt.ylabel(r'Ratios $r(t)$',fontsize=25)
    
    
    
    plt.savefig(fname="compare3_all.jpg",format="jpg", bbox_inches='tight')
    
    plt.savefig(fname="compare3_all.pdf",format="pdf", bbox_inches='tight')
    plt.show() 
    

    
