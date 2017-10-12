import math
cur_x = 100 # The algorithm starts at x=100
eta = 0.01 # step size multiplier
precision = 0.00001
previous_step_size = cur_x

def df(x):
    return 2*x + math.exp(x) -6

while previous_step_size > precision:
    prev_x = cur_x
    cur_x += -eta * df(prev_x)
    previous_step_size = abs(cur_x - prev_x)

f = (cur_x - 3)**2 + math.exp(cur_x)
print("The local minimum is %f" % f)
