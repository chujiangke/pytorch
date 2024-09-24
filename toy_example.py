import numpy as np 


def mse(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) **2
    return totalError / float(len(points))

def step_gradient(b_current, w_current, points, lr):
    b_gradient = 0
    w_gradient = 0
    M = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (2/M) * ((w_current * x + b_current) - y)
        w_gradient += (2/M) * ((w_current * x + b_current) - y)
    new_b = b_current - (lr * b_gradient)
    new_w = w_current - (lr * w_gradient)
    return [new_b, new_w]

def gradient_descent(points, starting_b, starting_w, lr,  num_iterations):
    b = starting_b
    w = starting_w

    for step in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), lr)
        loss = mse(b, w, points)
        if step%50 == 0:
            print(f"iteration:{step}, loss:{loss}, w:{w}, b:{b}")
    return [b, w]

if __name__== "__main__":
    data = []
    for i in range(100):
        x = np.random.uniform(-10, 10)
        eps = np.random.normal(0., 0.01)
        y = 1.477 *x + 0.089 + eps
        data.append([x, y])
    data = np.array(data)

    lr = 0.01
    initial_b = 0
    initial_w = 0

    num_iterations = 1000
    [b, w] = gradient_descent(data, initial_b, initial_w, lr, num_iterations)
    loss = mse(b, w, data)
    print(f"Final loss:{loss}, w:{w}, b:{b}")