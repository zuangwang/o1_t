

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
    #Reference_41 corresponds to Qin et al. (2022); 
    #Reference_24 corresponds to Mukherjee et al. (2023);
    #Reference_44 corresponds to Khaled et al. (2020); 
    #Reference_38 corresponds to Mitra et al. (2021b); 
    
    D1_1=np.transpose(readFile('paper_algorithm1_(5)_TAU2.txt'))
    D2_1=np.transpose(readFile('paper_algorithm1_(4)_TAU2.txt'))
    D3_1=np.transpose(readFile('Reference_41_TAU2.txt'))
    D4_1=np.transpose(readFile('Reference_44_TAU2.txt'))
    D5_1=np.transpose(readFile('paper_algorithm2_TAU2.txt'))
    D6_1=np.transpose(readFile('Reference_38_TAU2.txt'))
    D7_1=np.transpose(readFile('Reference_24_TAU2.txt'))
    
    k_1=799
    kk_1=500
    
    BASE1_1=np.zeros(k_1)
    BASE2_1=np.zeros(k_1)
    BASE3_1=np.zeros(kk_1)
    BASE4_1=np.zeros(kk_1)
    
    BASE5_1=np.zeros(k_1)
    BASE6_1=np.zeros(k_1)
    BASE7_1=np.zeros(kk_1)
    BASE8_1=np.zeros(kk_1)
    
    BASE9_1=np.zeros(k_1)
    BASE10_1=np.zeros(k_1)
    BASE11_1=np.zeros(kk_1)
    BASE12_1=np.zeros(kk_1)
    
    BASE13_1=np.zeros(k_1)
    BASE14_1=np.zeros(k_1)
    BASE15_1=np.zeros(kk_1)
    BASE16_1=np.zeros(kk_1)
    
    BASE17_1=np.zeros(k_1)
    BASE18_1=np.zeros(k_1)
    BASE19_1=np.zeros(kk_1)
    BASE20_1=np.zeros(kk_1)
    
    BASE21_1=np.zeros(k_1)
    BASE22_1=np.zeros(k_1)
    BASE23_1=np.zeros(kk_1)
    BASE24_1=np.zeros(kk_1)
    
    BASE25_1=np.zeros(k_1)
    BASE26_1=np.zeros(k_1)
    BASE27_1=np.zeros(kk_1)
    BASE28_1=np.zeros(kk_1)
    
    for j in range(0,k_1-1):
        BASE1_1[j]=j
        BASE2_1[j]=D1_1[j+1]
        
    for j in range(0,k_1-1):
        BASE5_1[j]=j
        BASE6_1[j]=D2_1[j+1]
        
    for j in range(0,k_1-1):
        BASE9_1[j]=j
        BASE10_1[j]=D3_1[j+1]
    
    for j in range(0,k_1-1):
        BASE13_1[j]=j
        BASE14_1[j]=D4_1[j+1]
        
    for j in range(0,k_1-1):
        BASE17_1[j]=j
        BASE18_1[j]=D5_1[j+1]
        
    for j in range(0,k_1-1):
        BASE21_1[j]=j
        BASE22_1[j]=D6_1[j+1]
        
    for j in range(0,k_1-1):
        BASE25_1[j]=j
        BASE26_1[j]=D7_1[j+1]
    
    
    k0_1=200
    
    for jj in range(0,kk_1):
        BASE3_1[jj]=BASE1_1[k0_1+jj]
        BASE4_1[jj]=BASE2_1[k0_1+jj]
      
        
    for jj in range(0,kk_1):
        BASE7_1[jj]=BASE5_1[k0_1+jj]
        BASE8_1[jj]=BASE6_1[k0_1+jj]
        
    for jj in range(0,kk_1):
        BASE11_1[jj]=BASE9_1[k0_1+jj]
        BASE12_1[jj]=BASE10_1[k0_1+jj]
        
        
    for jj in range(0,kk_1):
        BASE15_1[jj]=BASE13_1[k0_1+jj]
        BASE16_1[jj]=BASE14_1[k0_1+jj]
        
    for jj in range(0,kk_1):
        BASE19_1[jj]=BASE17_1[k0_1+jj]
        BASE20_1[jj]=BASE18_1[k0_1+jj]
        
    for jj in range(0,kk_1):
        BASE23_1[jj]=BASE21_1[k0_1+jj]
        BASE24_1[jj]=BASE22_1[k0_1+jj]
        
    for jj in range(0,kk_1):
        BASE27_1[jj]=BASE25_1[k0_1+jj]
        BASE28_1[jj]=BASE26_1[k0_1+jj]
    
    x_1 = BASE3_1
    y_1 = BASE4_1
    
    x2_1=BASE7_1
    y2_1=BASE8_1
    
    x3_1=BASE11_1
    y3_1=BASE12_1
    
    x4_1=BASE15_1
    y4_1=BASE16_1
    
    x5_1=BASE19_1
    y5_1=BASE20_1
    
    x6_1=BASE23_1
    y6_1=BASE24_1
    
    x7_1=BASE27_1
    y7_1=BASE28_1
    
   
   #Next
    
    
    D1_2=np.transpose(readFile('paper_algorithm1_(5)_TAU3.txt'))
    D2_2=np.transpose(readFile('paper_algorithm1_(4)_TAU3.txt'))
    D3_2=np.transpose(readFile('Reference_41_TAU3.txt'))
    D4_2=np.transpose(readFile('Reference_44_TAU3.txt'))
    D5_2=np.transpose(readFile('paper_algorithm2_TAU3.txt'))
    D6_2=np.transpose(readFile('Reference_38_TAU3.txt'))
    D7_2=np.transpose(readFile('Reference_24_TAU3.txt'))
   
    
    k_2=799
    kk_2=400
    
    BASE1_2=np.zeros(k_2)
    BASE2_2=np.zeros(k_2)
    BASE3_2=np.zeros(kk_2)
    BASE4_2=np.zeros(kk_2)
    
    BASE5_2=np.zeros(k_2)
    BASE6_2=np.zeros(k_2)
    BASE7_2=np.zeros(kk_2)
    BASE8_2=np.zeros(kk_2)
    
    BASE9_2=np.zeros(k_2)
    BASE10_2=np.zeros(k_2)
    BASE11_2=np.zeros(kk_2)
    BASE12_2=np.zeros(kk_2)
    
    BASE13_2=np.zeros(k_2)
    BASE14_2=np.zeros(k_2)
    BASE15_2=np.zeros(kk_2)
    BASE16_2=np.zeros(kk_2)
    
    BASE17_2=np.zeros(k_2)
    BASE18_2=np.zeros(k_2)
    BASE19_2=np.zeros(kk_2)
    BASE20_2=np.zeros(kk_2)
    
    BASE21_2=np.zeros(k_2)
    BASE22_2=np.zeros(k_2)
    BASE23_2=np.zeros(kk_2)
    BASE24_2=np.zeros(kk_2)
    
    BASE25_2=np.zeros(k_2)
    BASE26_2=np.zeros(k_2)
    BASE27_2=np.zeros(kk_2)
    BASE28_2=np.zeros(kk_2)
    
    
    for j in range(0,k_2-1):
        BASE1_2[j]=j
        BASE2_2[j]=D1_2[j+1]
        
        
    for j in range(0,k_1-1):
        BASE5_2[j]=j
        BASE6_2[j]=D2_2[j+1]
        
    for j in range(0,k_2-1):
        BASE9_2[j]=j
        BASE10_2[j]=D3_2[j+1]
    
    for j in range(0,k_2-1):
        BASE13_2[j]=j
        BASE14_2[j]=D4_2[j+1]
        
    for j in range(0,k_2-1):
        BASE17_2[j]=j
        BASE18_2[j]=D5_2[j+1]
        
    for j in range(0,k_2-1):
        BASE21_2[j]=j
        BASE22_2[j]=D6_2[j+1]
        
    for j in range(0,k_2-1):
        BASE25_2[j]=j
        BASE26_2[j]=D7_2[j+1]
    
    
    k0_2=200
    
    for jj in range(0,kk_2):
        BASE3_2[jj]=BASE1_2[k0_2+jj]
        BASE4_2[jj]=BASE2_2[k0_2+jj]
        
    for jj in range(0,kk_2):
        BASE7_2[jj]=BASE5_2[k0_2+jj]
        BASE8_2[jj]=BASE6_2[k0_2+jj]
        
    for jj in range(0,kk_2):
        BASE11_2[jj]=BASE9_2[k0_2+jj]
        BASE12_2[jj]=BASE10_2[k0_2+jj]
        
    for jj in range(0,kk_2):
        BASE15_2[jj]=BASE13_2[k0_2+jj]
        BASE16_2[jj]=BASE14_2[k0_2+jj]
        
    for jj in range(0,kk_2):
        BASE19_2[jj]=BASE17_2[k0_2+jj]
        BASE20_2[jj]=BASE18_2[k0_2+jj]
        
    for jj in range(0,kk_2):
        BASE23_2[jj]=BASE21_2[k0_2+jj]
        BASE24_2[jj]=BASE22_2[k0_2+jj]
        
    for jj in range(0,kk_2):
        BASE27_2[jj]=BASE25_2[k0_2+jj]
        BASE28_2[jj]=BASE26_2[k0_2+jj]
    
    x_2 = BASE3_2
    y_2 = BASE4_2
    
    x2_2=BASE7_2
    y2_2=BASE8_2
    
    x3_2=BASE11_2
    y3_2=BASE12_2
    
    x4_2=BASE15_2
    y4_2=BASE16_2
    
    x5_2=BASE19_2
    y5_2=BASE20_2
    
    x6_2=BASE23_2
    y6_2=BASE24_2
    
    x7_2=BASE27_2
    y7_2=BASE28_2
   
   #Next
    D1_3=np.transpose(readFile('paper_algorithm1_(5)_TAU4.txt'))
    D2_3=np.transpose(readFile('paper_algorithm1_(4)_TAU4.txt'))
    D3_3=np.transpose(readFile('Reference_41_TAU4.txt'))
    D4_3=np.transpose(readFile('Reference_44_TAU4.txt'))
    D5_3=np.transpose(readFile('paper_algorithm2_TAU4.txt'))
    D6_3=np.transpose(readFile('Reference_38_TAU4.txt'))
    D7_3=np.transpose(readFile('Reference_24_TAU4.txt'))
    
    
    k_3=799
    kk_3=300
    
    BASE1_3=np.zeros(k_3)
    BASE2_3=np.zeros(k_3)
    BASE3_3=np.zeros(kk_3)
    BASE4_3=np.zeros(kk_3)
    
    BASE5_3=np.zeros(k_3)
    BASE6_3=np.zeros(k_3)
    BASE7_3=np.zeros(kk_3)
    BASE8_3=np.zeros(kk_3)
    
    BASE9_3=np.zeros(k_3)
    BASE10_3=np.zeros(k_3)
    BASE11_3=np.zeros(kk_3)
    BASE12_3=np.zeros(kk_3)
    
    BASE13_3=np.zeros(k_3)
    BASE14_3=np.zeros(k_3)
    BASE15_3=np.zeros(kk_3)
    BASE16_3=np.zeros(kk_3)
    
    BASE17_3=np.zeros(k_3)
    BASE18_3=np.zeros(k_3)
    BASE19_3=np.zeros(kk_3)
    BASE20_3=np.zeros(kk_3)
    
    BASE21_3=np.zeros(k_3)
    BASE22_3=np.zeros(k_3)
    BASE23_3=np.zeros(kk_3)
    BASE24_3=np.zeros(kk_3)
    
    BASE25_3=np.zeros(k_3)
    BASE26_3=np.zeros(k_3)
    BASE27_3=np.zeros(kk_3)
    BASE28_3=np.zeros(kk_3)
    
    for j in range(0,k_3-1):
        BASE1_3[j]=j
        BASE2_3[j]=D1_3[j+1]
        
        
    for j in range(0,k_3-1):
        BASE5_3[j]=j
        BASE6_3[j]=D2_3[j+1]
        
    for j in range(0,k_3-1):
        BASE9_3[j]=j
        BASE10_3[j]=D3_3[j+1]
    
    for j in range(0,k_3-1):
        BASE13_3[j]=j
        BASE14_3[j]=D4_3[j+1]
        
    for j in range(0,k_3-1):
        BASE17_3[j]=j
        BASE18_3[j]=D5_3[j+1]
        
    for j in range(0,k_3-1):
        BASE21_3[j]=j
        BASE22_3[j]=D6_3[j+1]
        
    for j in range(0,k_3-1):
        BASE25_3[j]=j
        BASE26_3[j]=D7_3[j+1]
    
    
    k0_3=200
    
    for jj in range(0,kk_3):
        BASE3_3[jj]=BASE1_3[k0_3+jj]
        BASE4_3[jj]=BASE2_3[k0_3+jj]
      
    for jj in range(0,kk_3):
        BASE7_3[jj]=BASE5_3[k0_3+jj]
        BASE8_3[jj]=BASE6_3[k0_3+jj]
    
    for jj in range(0,kk_3):
        BASE11_3[jj]=BASE9_3[k0_3+jj]
        BASE12_3[jj]=BASE10_3[k0_3+jj]
        
    for jj in range(0,kk_3):
        BASE15_3[jj]=BASE13_3[k0_3+jj]
        BASE16_3[jj]=BASE14_3[k0_3+jj]
        
    for jj in range(0,kk_3):
        BASE19_3[jj]=BASE17_3[k0_3+jj]
        BASE20_3[jj]=BASE18_3[k0_3+jj]
        
    for jj in range(0,kk_3):
        BASE23_3[jj]=BASE21_3[k0_3+jj]
        BASE24_3[jj]=BASE22_3[k0_3+jj]
        
    for jj in range(0,kk_3):
        BASE27_3[jj]=BASE25_3[k0_3+jj]
        BASE28_3[jj]=BASE26_3[k0_3+jj]
    
    x_3 = BASE3_3
    y_3 = BASE4_3
    
    
    x2_3=BASE7_3
    y2_3=BASE8_3
    
    x3_3=BASE11_3
    y3_3=BASE12_3
    
    x4_3=BASE15_3
    y4_3=BASE16_3
    
    x5_3=BASE19_3
    y5_3=BASE20_3
    
    x6_3=BASE23_3
    y6_3=BASE24_3
    
    x7_3=BASE27_3
    y7_3=BASE28_3
   
   #Next
   
    D1_4=np.transpose(readFile('paper_algorithm1_(5)_TAU5.txt'))
    D2_4=np.transpose(readFile('paper_algorithm1_(4)_TAU5.txt'))
    D3_4=np.transpose(readFile('Reference_41_TAU5.txt'))
    D4_4=np.transpose(readFile('Reference_44_TAU5.txt'))
    D5_4=np.transpose(readFile('paper_algorithm2_TAU5.txt'))
    D6_4=np.transpose(readFile('Reference_38_TAU5.txt'))
    D7_4=np.transpose(readFile('Reference_24_TAU5.txt'))
   
    
    k_4=799
    kk_4=200
    
    BASE1_4=np.zeros(k_4)
    BASE2_4=np.zeros(k_4)
    BASE3_4=np.zeros(kk_4)
    BASE4_4=np.zeros(kk_4)
    
    BASE5_4=np.zeros(k_4)
    BASE6_4=np.zeros(k_4)
    BASE7_4=np.zeros(kk_4)
    BASE8_4=np.zeros(kk_4)
    
    BASE9_4=np.zeros(k_4)
    BASE10_4=np.zeros(k_4)
    BASE11_4=np.zeros(kk_4)
    BASE12_4=np.zeros(kk_4)
    
    BASE13_4=np.zeros(k_4)
    BASE14_4=np.zeros(k_4)
    BASE15_4=np.zeros(kk_4)
    BASE16_4=np.zeros(kk_4)
    
    BASE17_4=np.zeros(k_4)
    BASE18_4=np.zeros(k_4)
    BASE19_4=np.zeros(kk_4)
    BASE20_4=np.zeros(kk_4)
    
    BASE21_4=np.zeros(k_4)
    BASE22_4=np.zeros(k_4)
    BASE23_4=np.zeros(kk_4)
    BASE24_4=np.zeros(kk_4)
    
    BASE25_4=np.zeros(k_4)
    BASE26_4=np.zeros(k_4)
    BASE27_4=np.zeros(kk_4)
    BASE28_4=np.zeros(kk_4)
    
    for j in range(0,k_4-1):
        BASE1_4[j]=j
        BASE2_4[j]=D1_4[j+1]
        
        
    for j in range(0,k_4-1):
        BASE5_4[j]=j
        BASE6_4[j]=D2_4[j+1]
        
    for j in range(0,k_4-1):
        BASE9_4[j]=j
        BASE10_4[j]=D3_4[j+1]
    
    for j in range(0,k_4-1):
        BASE13_4[j]=j
        BASE14_4[j]=D4_4[j+1]
        
    for j in range(0,k_4-1):
        BASE17_4[j]=j
        BASE18_4[j]=D5_4[j+1]
        
    for j in range(0,k_4-1):
        BASE21_4[j]=j
        BASE22_4[j]=D6_4[j+1]
        
    for j in range(0,k_4-1):
        BASE25_4[j]=j
        BASE26_4[j]=D7_4[j+1]
    
    
    k0_4=200
    
    for jj in range(0,kk_4):
        BASE3_4[jj]=BASE1_4[k0_4+jj]
        BASE4_4[jj]=BASE2_4[k0_4+jj]
      
        
    for jj in range(0,kk_4):
        BASE7_4[jj]=BASE5_4[k0_4+jj]
        BASE8_4[jj]=BASE6_4[k0_4+jj]
        
    for jj in range(0,kk_4):
        BASE11_4[jj]=BASE9_4[k0_4+jj]
        BASE12_4[jj]=BASE10_4[k0_4+jj]
        
        
    for jj in range(0,kk_4):
        BASE15_4[jj]=BASE13_4[k0_4+jj]
        BASE16_4[jj]=BASE14_4[k0_4+jj]
        
    for jj in range(0,kk_4):
        BASE19_4[jj]=BASE17_4[k0_4+jj]
        BASE20_4[jj]=BASE18_4[k0_4+jj]
        
    for jj in range(0,kk_4):
        BASE23_4[jj]=BASE21_4[k0_4+jj]
        BASE24_4[jj]=BASE22_4[k0_4+jj]
        
    for jj in range(0,kk_4):
        BASE27_4[jj]=BASE25_4[k0_4+jj]
        BASE28_4[jj]=BASE26_4[k0_4+jj]
    
    x_4 = BASE3_4
    y_4 = BASE4_4
    
    x2_4=BASE7_4
    y2_4=BASE8_4
    
    x3_4=BASE11_4
    y3_4=BASE12_4
    
    x4_4=BASE15_4
    y4_4=BASE16_4
    
    x5_4=BASE19_4
    y5_4=BASE20_4
    
    x6_4=BASE23_4
    y6_4=BASE24_4
    
    x7_4=BASE27_4
    y7_4=BASE28_4
   
   #Next
    D1_5=np.transpose(readFile('paper_algorithm1_(5)_TAU6.txt'))
    D2_5=np.transpose(readFile('paper_algorithm1_(4)_TAU6.txt'))
    D3_5=np.transpose(readFile('Reference_41_TAU6.txt'))
    D4_5=np.transpose(readFile('Reference_44_TAU6.txt'))
    D5_5=np.transpose(readFile('paper_algorithm2_TAU6.txt'))
    D6_5=np.transpose(readFile('Reference_38_TAU6.txt'))
    D7_5=np.transpose(readFile('Reference_24_TAU6.txt'))
    
    k_5=799
    kk_5=150
    
    BASE1_5=np.zeros(k_5)
    BASE2_5=np.zeros(k_5)
    BASE3_5=np.zeros(kk_5)
    BASE4_5=np.zeros(kk_5)
    
    BASE5_5=np.zeros(k_5)
    BASE6_5=np.zeros(k_5)
    BASE7_5=np.zeros(kk_5)
    BASE8_5=np.zeros(kk_5)
    
    BASE9_5=np.zeros(k_5)
    BASE10_5=np.zeros(k_5)
    BASE11_5=np.zeros(kk_5)
    BASE12_5=np.zeros(kk_5)
    
    BASE13_5=np.zeros(k_5)
    BASE14_5=np.zeros(k_5)
    BASE15_5=np.zeros(kk_5)
    BASE16_5=np.zeros(kk_5)
    
    BASE17_5=np.zeros(k_5)
    BASE18_5=np.zeros(k_5)
    BASE19_5=np.zeros(kk_5)
    BASE20_5=np.zeros(kk_5)
    
    BASE21_5=np.zeros(k_5)
    BASE22_5=np.zeros(k_5)
    BASE23_5=np.zeros(kk_5)
    BASE24_5=np.zeros(kk_5)
    
    BASE25_5=np.zeros(k_5)
    BASE26_5=np.zeros(k_5)
    BASE27_5=np.zeros(kk_5)
    BASE28_5=np.zeros(kk_5)
    
    for j in range(0,k_5-1):
        BASE1_5[j]=j
        BASE2_5[j]=D1_5[j+1]
        
        
    for j in range(0,k_5-1):
        BASE5_5[j]=j
        BASE6_5[j]=D2_5[j+1]
        
    for j in range(0,k_5-1):
        BASE9_5[j]=j
        BASE10_5[j]=D3_5[j+1]
    
    for j in range(0,k_5-1):
        BASE13_5[j]=j
        BASE14_5[j]=D4_5[j+1]
        
    for j in range(0,k_5-1):
        BASE17_5[j]=j
        BASE18_5[j]=D5_5[j+1]
        
    for j in range(0,k_5-1):
        BASE21_5[j]=j
        BASE22_5[j]=D6_5[j+1]
        
    for j in range(0,k_5-1):
        BASE25_5[j]=j
        BASE26_5[j]=D7_5[j+1]
    
    
    k0_5=200
    
    for jj in range(0,kk_5):
        BASE3_5[jj]=BASE1_5[k0_5+jj]
        BASE4_5[jj]=BASE2_5[k0_5+jj]
      
        
    for jj in range(0,kk_5):
        BASE7_5[jj]=BASE5_5[k0_5+jj]
        BASE8_5[jj]=BASE6_5[k0_5+jj]
        
    
    
    for jj in range(0,kk_5):
        BASE11_5[jj]=BASE9_5[k0_5+jj]
        BASE12_5[jj]=BASE10_5[k0_5+jj]
        
        
    for jj in range(0,kk_5):
        BASE15_5[jj]=BASE13_5[k0_5+jj]
        BASE16_5[jj]=BASE14_5[k0_5+jj]
        
    for jj in range(0,kk_5):
        BASE19_5[jj]=BASE17_5[k0_5+jj]
        BASE20_5[jj]=BASE18_5[k0_5+jj]
        
    for jj in range(0,kk_5):
        BASE23_5[jj]=BASE21_5[k0_5+jj]
        BASE24_5[jj]=BASE22_5[k0_5+jj]
        
    for jj in range(0,kk_5):
        BASE27_5[jj]=BASE25_5[k0_5+jj]
        BASE28_5[jj]=BASE26_5[k0_5+jj]
    
    x_5 = BASE3_5
    y_5 = BASE4_5
    
    x2_5=BASE7_5
    y2_5=BASE8_5
    
    x3_5=BASE11_5
    y3_5=BASE12_5
    
    x4_5=BASE15_5
    y4_5=BASE16_5
    
    x5_5=BASE19_5
    y5_5=BASE20_5
    
    x6_5=BASE23_5
    y6_5=BASE24_5
    
    x7_5=BASE27_5
    y7_5=BASE28_5
   
   #Next
   
   
   
    
    
    plt.figure(figsize=(35,14))
    f1=plt.subplot(2, 3, 1)
    
    plt.semilogy(x5_1,y5_1,color='k',linewidth=1.9,linestyle='-')
    plt.semilogy(x2_1,y2_1,color='y',linewidth=1.9,linestyle='-.')
    plt.semilogy(x3_1,y3_1,color='r',linewidth=1.9,linestyle='--')
    plt.semilogy(x4_1,y4_1,color='g',linewidth=1.9,linestyle=':')
    plt.semilogy(x6_1,y6_1,color='m',linewidth=1.9,linestyle='dashed')
    plt.semilogy(x7_1,y7_1,color='c',linewidth=1.9,linestyle='dashdot')
    plt.semilogy(x_1,y_1, color='b',linewidth=1.9,linestyle='solid')
    
    
  #  f1.set_title(r'$\tau=2$',fontsize=25,y=-0.2)
    plt.grid()
    
    plt.xticks(fontsize = 18, fontname = 'times new roman')
    plt.yticks(fontsize = 18, fontname = 'times new roman')
    plt.xlabel(r'Communication Rounds $t$ ($\tau=2$)',fontsize=25)
    plt.ylabel(r'Errors $e(t)$',fontsize=25)
    
    
    f2=plt.subplot(2, 3, 2)
    
    plt.semilogy(x5_2,y5_2,color='k',linewidth=1.9,linestyle='-')
    plt.semilogy(x2_2,y2_2,color='y',linewidth=1.9,linestyle='-.')
    plt.semilogy(x3_2,y3_2,color='r',linewidth=1.9,linestyle='--')
    plt.semilogy(x4_2,y4_2,color='g',linewidth=1.9,linestyle=':')
    plt.semilogy(x6_2,y6_2,color='m',linewidth=1.9,linestyle='dashed')
    plt.semilogy(x7_2,y7_2,color='c',linewidth=1.9,linestyle='dashdot')
    plt.semilogy(x_2,y_2,color='b',linewidth=1.9,linestyle='solid')
    
    
  #  f2.set_title(r'$\tau=3$',fontsize=25,y=-0.2)
    plt.grid()
    
    plt.xticks(fontsize = 18, fontname = 'times new roman')
    plt.yticks(fontsize = 18, fontname = 'times new roman')
    plt.xlabel(r'Communication Rounds $t$ ($\tau=3$)',fontsize=25)
    plt.ylabel(r'Errors $e(t)$',fontsize=25)
    
    f3=plt.subplot(2, 3, 4)
    
    plt.semilogy(x5_3,y5_3,color='k',linewidth=1.9,linestyle='-')
    plt.semilogy(x2_3,y2_3,color='y',linewidth=1.9,linestyle='-.')
    plt.semilogy(x3_3,y3_3,color='r',linewidth=1.9,linestyle='--')
    plt.semilogy(x4_3,y4_3,color='g',linewidth=1.9,linestyle=':')
    plt.semilogy(x6_3,y6_3,color='m',linewidth=1.9,linestyle='dashed')
    plt.semilogy(x7_3,y7_3,color='c',linewidth=1.9,linestyle='dashdot')
    plt.semilogy(x_3,y_3,color='b',linewidth=1.9,linestyle='solid')
    
   # f3.set_title(r'$\tau=4$',fontsize=25,y=-0.2)
    plt.grid()
    
    plt.xticks(fontsize = 18, fontname = 'times new roman')
    plt.yticks(fontsize = 18, fontname = 'times new roman')
    plt.xlabel(r'Communication Rounds $t$ ($\tau=4$)',fontsize=25)
    plt.ylabel(r'Errors $e(t)$',fontsize=25)
    
    f4=plt.subplot(2, 3, 5)
    
    plt.semilogy(x7_4,y7_4,color='c',linewidth=1.9,linestyle='dashdot')
    plt.semilogy(x6_4,y6_4,color='m',linewidth=1.9,linestyle='dashed')
    plt.semilogy(x5_4,y5_4,color='k',linewidth=1.9,linestyle='-')
    plt.semilogy(x2_4,y2_4,color='y',linewidth=1.9,linestyle='-.')
    plt.semilogy(x4_4,y4_4,color='g',linewidth=1.9,linestyle=':')
    plt.semilogy(x3_4,y3_4,color='r',linewidth=1.9,linestyle='--')
    plt.semilogy(x_4,y_4, color='b',linewidth=1.9,linestyle='solid')
    

 #   f4.set_title(r'$\tau=5$',fontsize=25,y=-0.2)
    plt.grid()
    
    plt.xticks(fontsize = 18, fontname = 'times new roman')
    plt.yticks(fontsize = 18, fontname = 'times new roman')
    plt.xlabel(r'Communication Rounds $t$ ($\tau=5$)',fontsize=25)
    plt.ylabel(r'Errors $e(t)$',fontsize=25)
    
    f5=plt.subplot(2, 3, 6)
    
    plt.semilogy(x7_5,y7_5,label='Mukherjee et al. (2023)',color='c',linewidth=1.9,linestyle='dashdot')
    plt.semilogy(x6_5,y6_5,label='Mitra et al. (2021b)',color='m',linewidth=1.9,linestyle='dashed')
    plt.semilogy(x5_5,y5_5,label='Algorithm 2',color='k',linewidth=1.9,linestyle='-')
    plt.semilogy(x2_5,y2_5,label='Algorithm 1 with Universal Stepsizes',color='y',linewidth=1.9,linestyle='-.')
    plt.semilogy(x4_5,y4_5,label='Khaled et al. (2020)',color='g',linewidth=1.9,linestyle=':')
    plt.semilogy(x3_5,y3_5,label='Qin et al. (2022)',color='r',linewidth=1.9,linestyle='--')
    plt.semilogy(x_5,y_5,label='Algorithm 1 with Local Stepsizes', color='b',linewidth=1.9,linestyle='solid')
    
 #   f5.set_title(r'$\tau=6$',fontsize=25,y=-0.2)
    plt.legend(loc=2, bbox_to_anchor=(0.00,2.1),borderaxespad = 0,prop = {'size':25})
    plt.grid()
    
    plt.xticks(fontsize = 18, fontname = 'times new roman')
    plt.yticks(fontsize = 18, fontname = 'times new roman')
    plt.xlabel(r'Communication Rounds $t$ ($\tau=6$)',fontsize=25)
    plt.ylabel(r'Errors $e(t)$',fontsize=25)
    
    
    
    
    plt.savefig(fname="compare2_all.jpg",format="jpg", bbox_inches='tight')
    
    plt.savefig(fname="compare2_all.pdf",format="pdf", bbox_inches='tight')
    plt.show() 
    
