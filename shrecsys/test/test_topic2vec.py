# -*- coding:utf-8 -*-

import random
import matplotlib.pyplot as plt
if __name__=="__main__":
    x = [random.randint(1,100) for i in range(10000)]
    y = [2*x[i]+random.randint(1,5) for i in range(10000)]
    fig = plt.figure(1)
    fig.set_size_inches(w=20,h=20)
    fig1 = fig.add_subplot(1,2,1)

    fig2 = fig.add_subplot(1,2,2)
    fig3 = fig.add_subplot(2,1,1)
    fig4 = fig.add_subplot(2,2,2)
    fig.subplots_adjust()
    fig1.plot(x,y)
    fig2.scatter(x,y)
    fig3.plot(x)
    fig4.plot(y)
    plt.show()