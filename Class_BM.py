# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 02:59:31 2018

@author: Mhamed
"""

import numpy as np


class QP_BarrierMethod():
    """ 
    Class that solves the following quadratic optimization problem : 
    minimize phi(v) = v^TQv + p^Tv subject to Av <= b.
    Uses barrier method to solve.
    """
    
    def __init__(self, Q, A, p, b, v0, alpha=0.3, beta=0.7, t=1, mu=10, tol=1e-02):
        self.Q = Q
        self.A = A
        self.p = p
        self.b = b
        self.v0 = v0
        
        self.t = t
        self.alpha = alpha
        self.beta = beta
        self.tol = tol
        self.mu = mu
        
        
    def backtrack_linesearch(self, x, dx):
        """ 
        Takes in input : 
            x, ndarray of shape (A.shape[1],a)
            dx, ndarray of shape (A.shape[1],a)
        """
        
        t = 1
        #Make sure that the log exists before calculating the value of phi + inequality condition
        while ((np.min(self.b-self.A.dot(x+t*dx))<=0)  or (self._phi(x+t*dx) > self._phi(x) + self.alpha * t * np.dot(self._grad(x).T,dx)[0,0])) and t > 1e-2:
            t = self.beta * t
        return t
    
    
    def centering_step(self, x, max_iter=1000):
        xhist = list()
        i = 0
        lambda_squared = 1
        while i<max_iter:
            H = self._hess(x)
            J = self._grad(x)
            # Step 1 : compute newton step and decrement
            dx = -np.dot(np.linalg.inv(H),J)
            lambda_squared = -np.dot(J.T,dx)[0,0]
            xhist.append(x)
            # Step 2 : stopping criterion
            if lambda_squared < 2 * self.tol:
                break
            # Step 3 : Choose step size t by backtracking line search
            t = self.backtrack_linesearch(x, dx)
            # Step 4 : update
            x = x + t * dx
            i += 1
        
        return x, np.array(xhist)
    
    
    def barr_method(self):
        m = self.A.shape[0]
        xhist = [self.v0]
        x = self.v0
        i=1
        newton_iterations = 0
        while m / self.t >= self.tol:
            x, hist_newton = self.centering_step(x)
            xhist.append(x)
            self._set_param(self.mu * self.t)
            newton_iterations += len(hist_newton)
            print(i)
            i+=1
            
        x_sol = xhist[-1]
        return x_sol, np.array(xhist), newton_iterations
    
    
    
    def _phi(self, x):                                    
        tmp = self.t *  np.sum(x * np.dot(self.Q,x)) + x.T.dot(self.p)  - np.sum(np.log(self.b - self.A.dot(x)))
        return tmp[0,0]
                                             
    def _grad(self, x):
        tmp=list()
        for i in range(self.b.shape[0]):
            tmp.append( (1./(self.b[i]-self.A[i].dot(x))) * (self.A[i]))
        return self.t * (2*self.Q.dot(x) + self.p)+ sum(tmp).reshape(-1,1)

    def _hess(self,x):
        tmp = list()
        for i in range(self.b.shape[0]) : 
            tmp.append( ( (1./(self.b[i]-self.A[i].dot(x)))**2) * (np.dot(self.A[i].reshape(-1,1),self.A[i].reshape(-1,1).T) ) )
        hess_x = 2*self.t*self.Q + sum(tmp)
        return  hess_x
    
    def _set_param(self,t):
        self.t=t
        
        
