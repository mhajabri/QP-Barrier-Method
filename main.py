"""
@author: Mhamed
"""

import numpy as np
from Class_BM import QP_BarrierMethod
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist


if __name__ == "__main__": 
    n=50
    d=200
    
    np.random.seed(17)
    X = np.random.rand(n,d)
    np.random.seed(20)
    w = np.random.rand(d,1)
    np.random.seed(42)
    y = np.dot(X,w)+np.random.rand(n,1)
    Lambda = 10
    
    Q = 1/2*np.eye(n)
    A = np.vstack((X.T , -X.T))
    p = y
    b = Lambda * np.ones((2*d,1))
    x0 = 0*np.ones((n,1))
    
    L=[]
    opti_values=[]
    opti_dual = []
    list_hist=[]
    mu_val = [2, 5, 10, 15, 20, 50, 100]
    obj_fct = lambda x, Q: (x.T).dot(Q).dot(x) + (p.T).dot(x)
    ###############################################################
    
    for mu in mu_val:
        model = QP_BarrierMethod(Q,A,p,b,x0, mu=mu)
        x_sol, xhist, newton_iterations = model.barr_method()
        optimal_value, hist_values = obj_fct(x_sol,Q), np.array([obj_fct(x,Q) for x in xhist])
        L.append(newton_iterations)
        opti_dual.append(x_sol)
        opti_values.append(optimal_value)
        list_hist.append(hist_values)
    
    # Plot f(v_t) - f*
    f, ax = plt.subplots(1, figsize=(10,6))
    
    ax.set_yscale("log")
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Duality gap")
    ax.grid()
    
    for i in range(len(mu_val)):
        optimal_value = opti_values[i][0]
        dgap = np.array([hist_value[0,0] -optimal_value for hist_value in list_hist[i] ])
        ax.step(np.arange(len(dgap)), dgap, label=" mu = {}".format(mu_val[i]))
        
    ax.legend()
    plt.show();
    
    # Impact of mu on w*
    w_star = np.array([np.dot(np.linalg.pinv(X),sol + y) for sol in opti_dual])[:,:,0]
    pdist(w_star)
    
    # Plot Newton iterations to chose mu
    plt.figure(figsize=(10,6))
    plt.plot(mu_val, L, '-ok');
    plt.xlabel('mu')
    plt.ylabel('Number of Newton iterations')
    plt.show();