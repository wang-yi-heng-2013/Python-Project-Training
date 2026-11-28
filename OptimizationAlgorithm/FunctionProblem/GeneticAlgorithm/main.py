import math
from GeneticAlgorithm import geneticalgorithm
import painting

def function(x):
    return x+10*math.sin(5*x)+7*math.cos(4*x)


# 调用构建的模型，参数分别是（函数、定义域下限，定义域上限，划分刻度数量）
result = geneticalgorithm(function,0,10,2^20)
result.start()

# 获取所需结果和数据
x,max=result.result_max()
fitness_max=result.fitness_max
fitness_average=result.fitness_average

print("函数最大值坐标： 最大值：")
print(f"{x:<12}",max)

# 绘制适应度变化函数
painting.fitness_line_chart(fitness_max,fitness_average)

