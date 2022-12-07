from PO_drafted import PO
import time
from statistics import mean
import random
import numpy as np

po_params = {
    'fun':lambda x: x[0]**2 + x[1]**2, #the function we want to optimize
    'n':3,                             #the number of contituencies, political parties and party members
    'tmax':100,                        #number of iterations
    'd':2 ,                            #number of dimensions, we don't want this to be a parameter later I think
    'lambdamax' : 1                    #upper limit of the party switching rate
}


def f1(x):
    return x[0]**2 + x[1]**2


def f2(x):
    return x[0] ** 4 + 2*x[1] ** 4 + random.uniform(0,1)


def f3(x):
    return np.abs(x[0])**2 + np.abs(x[1]) ** 3


def f4(x):
    return np.abs(x[0]) + np.abs(x[1])


def f5(x):
    return (x[0]+.5)**2 + (x[1]+.5)**2


def f6(x):
    return (x[0]**2)**(x[1]**2+1)+(x[1]**2)**(x[0]**2+1)


def f7(x):
    return x[0]**2 + x[1]**2 + x[1]**2


def f8(x):
    return np.abs(x[0]) + np.abs(x[1]) + np.abs(x[0]) * np.abs(x[1])


def f9(x):
    return x[0]**10 + x[1]**10

## Run model
all_fun = [f1,f2,f3,f4,f5,f6,f7,f8,f9]

for k in range(len(all_fun)):
    print(f"Function : {k}")
    minimas = []
    times = []
    po_params['fun'] = all_fun[k]
    for i in range(1000):
        start_time = time.time()
        po = PO(po_params)
        result = po._train__()
        times.append(time.time() - start_time)
        minimas.append(result[1])

    print("--- %s seconds ---" % (sum(times)))
    print("--- BEST MIN %s---" % ('%.2E' % min(minimas)))
    print("--- WORST MIN %s---" % ('%.2E' % max(minimas)))
    print("--- MEAN MIN %s---" % ('%.2E' % mean(minimas)))
