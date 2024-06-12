

import pandas as pd               #numpy和pandas都是一个小工具，用来转换的，不是运算的
import pretreatment
import matplotlib.pyplot as plt



p = pretreatment.PPPPPretreatment()   #这个样子 引用了pre里的PPP。并化简为p 声明了以后就随便用了  或者你 import pretreatment.PPPPPretreatment as p，应该也是行的

file_path = '待处理.xlsx'


def dddraw(a,b):
    b = b.transpose()                         #重新进行画图，需要转置反射率
    plt.figure()
    plt.plot(a, b, linewidth=1)      #1的格式是(228,) （变为.values） 2的格式是[228 rows x 310 columns]  这样对应上的，228是横轴长度，都放到最前面了
    plt.xlabel('spectrum/nm')                        #1的格式也可以是(228,)   未变为.values
    plt.ylabel('reflectivity')
    plt.pause(5)                     #延时6s后继续执行下面的程序  ，用draw函数，就必须配这一行，这一行执行才会出那个图



data = pd.read_excel(file_path,header=None)

print("以下为光谱输入")
spectrum = data.iloc[0, 1:]  # 所有行左侧的冒号代表从第一到最后的所有行，即为光谱波段。后面的0代表是第0列的数据，就是列、、、若为1就是样本1的反射率数据了
print(spectrum)
print(spectrum.shape)

print("以下为光谱反射率全局输入")
reflectivity = data.iloc[1:, 1:]
reflectivity = round(reflectivity, 4)
reflectivity = reflectivity.values   #zhuanhuazhuanh  统一转化为np格式
print(reflectivity)                    #读取完毕了
dddraw(spectrum,reflectivity)    #生成一下图,



print('使用msc就解除注释即可，每次只用一种预处理')

# reflectivity=p.msc(reflectivity)    #msc就执行这行代码


print('使用sg就解除注释即可，每次只用一种预处理')

# reflectivity=p.SG(reflectivity)    #SG就执行这行代码    可以在pre的.py程序里设置具体参数


print('使用snv就解除注释即可，每次只用一种预处理')

# reflectivity=p.snv(reflectivity)    #就执行这行代码    可以在pre的.py程序里设置具体参数


print('使用标准化就解除注释即可，每次只用一种预处理')
# #
# reflectivity=reflectivity.transpose()                 #这个要改转置  转置的.后面要加括号 #执行标准化操作 标准化操作要进行一次反射率转置
# reflectivity=p.StandardScaler(reflectivity)           #标准化操作 来自 gpt   一直出错是因为 在定义def函数的时候，需要定义两个输入 一个self  一个 传参 没有定义self  所以一直报错   传入的数据是198*310
# reflectivity=reflectivity.transpose()                 #za处理完毕了 再转置回来



dddraw(spectrum,reflectivity)    #生成一下图
df = pd.DataFrame(reflectivity)
df.to_excel('预处理结果.xlsx', index=False)
print('\n\n处理后的光谱数据保存完毕')

