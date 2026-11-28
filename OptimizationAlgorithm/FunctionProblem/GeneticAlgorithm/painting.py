import matplotlib.pyplot as plt
import numpy as np

# 让输出的图表能显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘制遗传过程中适应度函数变化曲线
def fitness_line_chart(max,average):
    x=list(range(len(max)))
    y1=max
    y2=average

    # 绘制适应度值的最大值和平均值的变化曲线
    plt.plot(x,y1,label='max',marker='o')
    plt.plot(x,y2,label='average',marker='s')

    plt.title("适应度值变化曲线（最大和平均）")
    plt.xlabel("遗传次数")
    plt.ylabel("适应度值")

    plt.legend()
    plt.grid(True)
    plt.show()


# 绘制函数图像和种群的分布
def group_location(function,min,max,location,group):
    x=np.linspace(min,max,500)
    y=[]
    for i in range(len(x)):
        y.append(function(x[i]))

    plt.plot(x,y,label="函数图像",color="blue")
    plt.scatter(location,group,s=40,label="种群",color="red")

    plt.xlabel("x")
    plt.ylabel("y")

    plt.legend()
    plt.show()
