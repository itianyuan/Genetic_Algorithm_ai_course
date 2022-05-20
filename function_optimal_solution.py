from distutils import file_util
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


POPULATION_SIZE = 1000   #种群大小
GENERATION_NUMBER = 100 #迭代次数

CROSS_RATE = 0.6    #交叉率
VARIATION_RATE = 0.1   #变异率

S = 2   #轮盘赌保留的最优个体数

X_RANGE = [-3, 12.1]    #X范围
X_SIZE = 18

Y_RANGE = [4.1, 5.8]    #Y范围
Y_SIZE = 15


# 问题函数
def problem_function(x, y):
    return 21.5 + x*np.sin(4*np.pi*x) + y*np.sin(20*np.pi*y)
    # return 20+x+y


# 初始化图
# @param ax 3D图像
def init_graph(ax):
    x_sequence = np.linspace(*X_RANGE, 100)  # 创建x等差数列
    y_sequence = np.linspace(*Y_RANGE, 100)  # 创建y等差数列
    x_matrix, y_matrix = np.meshgrid(x_sequence, y_sequence)  # 生成x和y的坐标矩阵
    z_matrix = problem_function(x_matrix, y_matrix)  # 生成z坐标矩阵
    # 创建曲面图,行跨度为1，列跨度为1，设置颜色映射
    ax.plot_surface(x_matrix, y_matrix, z_matrix, rstride=1, cstride=1, cmap=plt.get_cmap('coolwarm'))
    ax.set_zlim(0, 40)  # 自定义z轴范围
    ax.set_xlabel('x')  # 设置x坐标轴标题
    ax.set_ylabel('y')  # 设置y坐标轴标题
    ax.set_zlabel('z')  # 设置z坐标轴标题
    # plt.pause(3)  # 暂停3秒
    plt.show()  # 显示图


# 解码
def decoding_DNA(population_matrix):

    #分割为xy矩阵
    x_matrix = population_matrix[:, 0:X_SIZE]
    y_matrix = population_matrix[:, X_SIZE:X_SIZE+Y_SIZE]

    decoding_vector_x = 2 ** np.arange(X_SIZE)[::-1]
    decoding_vector_y = 2 ** np.arange(Y_SIZE)[::-1]

    #映射x向量
    population_x_vector = x_matrix.dot(decoding_vector_x) / (2 ** X_SIZE - 1) * (X_RANGE[1] - X_RANGE[0]) + X_RANGE[0]
    #映射y向量
    population_y_vector = y_matrix.dot(decoding_vector_y) / (2 ** Y_SIZE - 1) * (Y_RANGE[1] - Y_RANGE[0]) + Y_RANGE[0]

    return population_x_vector, population_y_vector

# 两点交叉
def cross(population_matrix):
    for i in range (0,int(POPULATION_SIZE/2)):
        if np.random.rand() < CROSS_RATE:
            #选取交叉位置
            first_position = np.random.randint(X_SIZE+Y_SIZE)
            second_position = np.random.randint(X_SIZE+Y_SIZE)
            #前小于后
            if (first_position>second_position):
                first_position,second_position=second_position,first_position
            #交叉
            # print(first_position)
            # print(second_position)
            # print(population_matrix)
            temp=population_matrix[2*i][first_position:second_position]
            population_matrix[2*i][first_position:second_position]=population_matrix[2*i+1][first_position:second_position]
            population_matrix[2*i+1][first_position:second_position]=temp


# 变异
def variation(child_DNA):
    if np.random.rand() < VARIATION_RATE:
        variation_position = np.random.randint(X_SIZE+Y_SIZE)  #选取变异位置
        child_DNA[variation_position] = child_DNA[variation_position] ^ 1


# 交叉和变异
def update_population(population_matrix):

    cross(population_matrix)  #交叉

    new_population_matrix = []  #声明新的空种群
    # 遍历种群所有个体
    for father_DNA in population_matrix:
        child_DNA = father_DNA  # 孩子先得到父亲的全部DNA
        variation(child_DNA)  # DNA变异
        new_population_matrix.append(child_DNA)  # 添加到新种群中
    new_population_matrix = np.array(new_population_matrix)  # 转化数组
    return new_population_matrix


# 计算适应度值
def get_fitness_vector(population_matrix):
    population_x_vector, population_y_vector = decoding_DNA(population_matrix)  # 获取种群x和y向量
    fitness_vector = problem_function(population_x_vector, population_y_vector)  # 获取适应度向量
    fitness_vector = fitness_vector - np.min(fitness_vector) + 1e-3  # 适应度修正，保证适应度大于0
    return fitness_vector


# 轮盘赌选择
def natural_selection(population_matrix, fitness_vector):

    best_selected=sorted(np.argsort(fitness_vector)[-S:])
    
    # print(best_selected)
    other_selected = np.random.choice(np.arange(POPULATION_SIZE),  # 被选取的索引数组
                                   size=POPULATION_SIZE-S,  # 选取数量
                                   replace=True,  # 允许重复选取
                                   p=fitness_vector / fitness_vector.sum())  # 数组每个元素的获取概率

    # print(other_selected)
    index_array=np.append(best_selected,other_selected)
    # print(index_array)
    return population_matrix[index_array]


# 打印结果
# @param population_matrix 种群矩阵
def print_result(population_matrix):
    fitness_vector = get_fitness_vector(population_matrix)  # 获取适应度向量
    optimal_fitness_index = np.argmax(fitness_vector)  # 获取最大适应度索引
    #print('最佳适应度为：', fitness_vector[optimal_fitness_index])
    print('最优个体：', population_matrix[optimal_fitness_index])
    population_x_vector, population_y_vector = decoding_DNA(population_matrix)  # 获取种群x和y向量
    print('(x,y)：(',format(population_x_vector[optimal_fitness_index], '.4f'),',', format(population_y_vector[optimal_fitness_index], '.4f'),')')
    print('value：',
          format(problem_function(population_x_vector[optimal_fitness_index], population_y_vector[optimal_fitness_index]), '.4f'))


if __name__ == '__main__':
    fig = plt.figure()  # 创建空图像
    ax = Axes3D(fig)  # 创建3D图像
    plt.ion()  # 切换到交互模式绘制动态图像
    # init_graph(ax)  # 初始化图

    #初始化种群
    population_matrix = np.random.randint(2, size=(POPULATION_SIZE, X_SIZE+Y_SIZE))  #生成01随机数

    # 迭代50世代
    for i in range(GENERATION_NUMBER):
        print("代数："+str(i))
        print_result(population_matrix)  # 打印结果
        population_x_vector, population_y_vector = decoding_DNA(population_matrix)  #解码

        # plt.cla()
        # init_graph(ax)  # 初始化图

        # 绘制散点图，设置颜色和标记风格
        ax.scatter(population_x_vector,
                   population_y_vector,
                   problem_function(population_x_vector, population_y_vector),
                   c='g',
                   marker='x')
        # plt.show()  # 显示图
        # plt.pause(0.05)  # 暂停0.1秒

        fitness_vector = get_fitness_vector(population_matrix)  #计算个体适应度值

        population_matrix = natural_selection(population_matrix, fitness_vector)  #轮盘赌选择

        population_matrix = update_population(population_matrix)  #交叉和变异
        

    print_result(population_matrix)  # 打印结果
    plt.ioff()  # 关闭交互模式
    # plt.show()  # 绘制结果