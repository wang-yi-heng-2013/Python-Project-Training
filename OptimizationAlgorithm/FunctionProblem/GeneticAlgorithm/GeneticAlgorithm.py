import math
import random
import painting

# 遗传算法求解函数最大值问题
class geneticalgorithm:
    def __init__(self,function,min,max,function_result_num,solution_num=30,
                 times_generation=100,generation_probability=0.05,mutation_probability=0.002):
        # 函数基本数据
        self.function=function  # 函数
        self.function_boundary=[min,max]    # 函数定义域

        # 保存解和需要记录的数据
        self.solution_num=solution_num  # 种群的数量
        self.gene_length=int(math.log2(function_result_num)+1)   # 一个解的列表长度
        self.function_result_num=pow(2,self.gene_length)    # 划分定义域刻度
        self.generation_group=[ [0]*self.gene_length for _ in range(solution_num) ] # 保存每一代种群
        self.generation_best=[0]*self.gene_length    # 记录遍历过程中的最优解
        self.fitness_best=0 # 保存最优解的适应度值
        self.fitness_max=[]    # 保存每一代的最大适应度
        self.fitness_average=[]    # 保存每一代的平均适应度
        self.fitness_group=[ 0 for _ in range(solution_num) ] # 保存种群中每个个体的适应度值

        # 算法所需参数
        self.times_generation=times_generation  # 遗传代数
        self.generation_probability=generation_probability  # 交叉概率
        self.mutation_probability=mutation_probability  # 变异概率


    # 算法入口
    def start(self):
        # 调用初始化方法
        self.initialize()
        # 初始种群分布绘制
        self.group_location()

        # 开始迭代
        for times in range(self.times_generation):
            # 选择可以作为父代遗传给下一代的个体
            self.choose_group()
            # 调用获取下一代的方法，即遗传和变异
            self.group_genetic()
            # 更新适应度值，以及获取需要的解
            self.fitness_update()
            # 绘制当前种群分布图
            if times%20==0 :
                self.group_location()

        # 结束后的种群分布
        self.group_location()


    # 获取初始解
    def initialize(self):
        # 随机方法获得初始种群
        for i in range(self.solution_num):
            for j in range(self.gene_length):
                self.generation_group[i][j]=random.choice([0,1])

        # 更新适应度值，以及相应的解
        self.fitness_update()


    # 获取当前种群的适应度值，并更新相应的解
    def fitness_update(self):
        # 更新种群中每一个的适应度值
        for i in range(self.solution_num):
            # 将保存个体的列表拼接成二进制数
            solve_location_2=''.join(map(str,self.generation_group[i]))
            # 转化为十进制数
            solve_location_10=int(solve_location_2,2)
            # 调用计算适应度值的方法
            self.fitness_group[i]=self.fitness_calculation(solve_location_10)

        # 保存当前的最大适应度和平均适应度
        fitness_max_now=max(self.fitness_group)
        self.fitness_max.append(fitness_max_now)
        self.fitness_average.append( (sum(self.fitness_group))/self.solution_num )

        # 判断是否需要更新最优解
        if fitness_max_now>self.fitness_best :
            self.fitness_best=fitness_max_now
            id=self.fitness_group.index(fitness_max_now)
            self.generation_best=self.generation_group[id].copy()


    # 单独定义计算适应度值的方法，便于在求(最大/最小)值时能及时调整
    def fitness_calculation(self,location):
        result=(self.function_boundary[0]+
                (location/self.function_result_num)*(self.function_boundary[1]-
                                                     self.function_boundary[0]))
        # 将函数值作为适应度值进行计算
        fitness=self.function(result)
        return fitness


    # 选择父本种群
    def choose_group(self):
        # 临时保存父本
        generation_group_temp = [[0] * self.gene_length for _ in range(self.solution_num)]

        # 采用轮盘选择法，计算每个个体可以作为父本的概率
        fitness_group_probabilty=[ 0 for _ in range(self.solution_num) ]

        # 将所有个体的适应度值置为正数，且保持比例不变
        fitness_min=min(self.fitness_group)
        if fitness_min<=0 :
            for i in range(self.solution_num):
                self.fitness_group[i]=self.fitness_group[i]+abs(fitness_min)+1

        # 计算每个个体在轮盘中的位置和范围
        fitness_group_probabilty[0]=self.fitness_group[0]
        for i in range(1,self.solution_num):
            fitness_group_probabilty[i]=fitness_group_probabilty[i-1]+self.fitness_group[i]

        for i in range(self.solution_num):
            fitness_group_probabilty[i]=fitness_group_probabilty[i]/fitness_group_probabilty[-1]

        # 选择父代
        for i in range(self.solution_num):
            choose=random.random()
            count=0
            while choose>fitness_group_probabilty[count]:
                count=count+1
            generation_group_temp[i]=self.generation_group[count].copy()

        # 选择完成后，让父本取代当前种群
        self.generation_group=generation_group_temp.copy()


    # 定义遗传方法
    def group_genetic(self):
        # 遗传过程(相邻两个个体发生)
        for i in range(0,self.solution_num,2):
            for j in range(0,self.gene_length):
                if random.random()<self.generation_probability :
                    temp=self.generation_group[i][j]
                    self.generation_group[i][j]=self.generation_group[i+1][j]
                    self.generation_group[i+1][j]=temp

        # 变异过程(遍历每个个体)
        for i in range(0,self.solution_num):
            for j in range(0,self.gene_length):
                if random.random()<self.mutation_probability :
                    self.generation_group[i][j]^=1  # 通过异或实现变异


    # 返回最优解及其坐标
    def result_max(self):
        # 将最优结果转变为坐标值形式
        result_best_2 = ''.join(map(str, self.generation_best))
        result_best_10 = int(result_best_2, 2)
        result_best=(self.function_boundary[0]+
                     (result_best_10/self.function_result_num)*(self.function_boundary[1]-
                                                                self.function_boundary[0]))
        result=self.function(result_best)
        return result_best,result

    # 绘制种群分布
    def group_location(self):
        location_group=[ 0 for _ in range(self.solution_num) ]

        for i in range(self.solution_num):
            result_best_2 = ''.join(map(str, self.generation_group[i]))
            result_best_10 = int(result_best_2, 2)
            location_group[i]=(self.function_boundary[0]+
                               (result_best_10/self.function_result_num)*(self.function_boundary[1]-
                                                                          self.function_boundary[0]))

        painting.group_location(self.function,self.function_boundary[0],self.function_boundary[1],
                                location_group,self.fitness_group)
