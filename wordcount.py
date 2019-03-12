import numpy as np
from matplotlib import pylab


# 利用最小二乘法求损失函数(loss function)
def compute_error(b, m, data):

    #second way
    x = data[:, 0]  # 二维数组的第一列
    y = data[:, 1]  # 二维数组第二列
    totalError = (y-m*x-b)**2
    totalError = np.sum(totalError, axis=0)  # 二维数组的列和

    return totalError/(2*float(len(data)))


def optimizer(data, starting_b, starting_m, learning_rate, num_iter):
    # 给直线一个初始的截距和斜率
    b = starting_b
    m = starting_m

    # gradient descent 利用梯度下降，按导数（偏导）绝对值下降的方向
    for i in range(num_iter):
        # update b and m with the new more accurate b and m by performing
        # the gradient step
        b, m = compute_gradient(b, m, data, learning_rate)
        if i % 100 == 0:
            # 字符串格式化
            print('iter {0}:error={1}'.format(i, compute_error(b, m, data)))
    return [b, m]


def compute_gradient(b_current, m_current, data, learning_rate):

    b_gradient = 0
    m_gradient = 0

    N = float(len(data))
    # Two ways to implement this

    # Vectorization implementation
    x = data[:, 0]
    y = data[:, 1]
    # 对b求偏导partial derivations
    b_gradient = -(1/N)*(y-m_current*x-b_current)
    b_gradient = np.sum(b_gradient, axis=0)

    # 对m求偏导partial derivations
    m_gradient = -(1/N)*x*(y-m_current*x-b_current)
    m_gradient = np.sum(m_gradient, axis=0)

    # update our b and m values using our partial derivations
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def plot_data(data, b, m):

    # plotting
    x = data[:, 0]
    y = data[:, 1]
    y_predict = m*x+b
    # 表示形状为圆形
    pylab.plot(x, y, '^')
    pylab.plot(x, y_predict, 'r')
    pylab.show()


def Linear_regression():
    # get train data
    data =np.loadtxt('data.csv', delimiter=',')

    # 步长
    learning_rate = 0.001
    initial_b = 0.0
    initial_m = 0.0
    # 迭代次数
    num_iter = 5000

    # train model
    # print b m error
    print('initial variables:\n initial_b = {0}\n intial_m = {1}\n error of begin = {2} \n'\
        .format(initial_b, initial_m, compute_error(initial_b, initial_m, data)))

    # optimizing b and m
    [b, m] = optimizer(data, initial_b, initial_m, learning_rate, num_iter)

    # print final b m error
    print('final formula parmaters:\n b = {1}\n m={2}\n error of end = {3} \n'.format(num_iter, b, m, compute_error(b, m, data)))

    # plot result
    plot_data(data, b, m)

if __name__ == '__main__':
    Linear_regression()


