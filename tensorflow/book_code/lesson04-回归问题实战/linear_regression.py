import numpy as np
import matplotlib.pyplot as pyplot
# y = wx + b
def compute_error_for_line_given_points(b, w, points):
    totalError = 0  
    for i in range(0, len(points)): 
        x = points[i, 0]  #行向量
        y = points[i, 1]  #列向量
        if(0==i):
            print("Starting gradient descent at x = {0}, y = {1}".format(x, y))
        # computer mean-squared-error
        totalError += (y - (w * x + b)) ** 2
    # average loss for each point
    return totalError / float(len(points))

def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # grad_b = 2(wx+b-y)
        b_gradient += (2/N) * ((w_current * x + b_current) - y)
        # grad_w = 2(wx+b-y)*x
        w_gradient += (2/N) * x * ((w_current * x + b_current) - y)
    # update w'
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]

def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    # update for several times
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]
def run():
    points = np.genfromtxt("data.csv", delimiter=",") #读取数据以逗号分隔数据
    #print(points)
    learning_rate = 0.0001 #学习率
    initial_b = 0 # 初始化截距
    initial_w = 0 # 初始化斜率
    num_iterations = 1000  # 迭代次数
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}".format(initial_b, initial_w,
                  compute_error_for_line_given_points(initial_b, initial_w, points)))
    print("Running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, w = {2}, error = {3}".format(num_iterations, b, w,
                 compute_error_for_line_given_points(b, w, points)))
    x = np.linspace(20, 80 , 5)
    y = w * x + b
    pyplot.plot(x, y)
    pyplot.scatter(points[:,0], points[:,1])
    pyplot.show()
if __name__ == '__main__':
    run()
#https://blog.csdn.net/m0_63062182/article/details/124011947
