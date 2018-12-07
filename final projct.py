# -*- coding: utf-8 -*-
import statsmodels.api as sm
import pandas as pd
import numpy as np
import 
#%%
'''
Curveid 1Y	     2Y	  3Y	  4Y	  5Y  	10Y
Curve1	18.01	20.98	23.06	31.16	34.92	39.13
Curve2	52.06	78.08	77.89	104.70	119.97	125.96
Curve3	247.19	237.08	229.04	217.03	199.01	149.00
Curve4	59.32	75.13	83.00	105.45	116.00	126.00
'''
#%%
ratelist1 = [18.01,	20.98,	23.06,	31.16,	34.92,	39.13]
ratelist2 = [52.06,	78.08,	77.89,	104.70,119.97,125.96]
ratelist3 = [247.19,237.08,229.04,217.03,199.01,149.00]
ratelist4 = [59.32,75.13,83.00,105.45,116.00	,126.00]


ratelist = [ratelist1]
def get_point(ratelist):
    point = []
    for i in ratelist:
        for ret_index in range(1,len(i) - 1):
            point.append([i[ret_index - 1],i[ret_index]])
    return point

rate_points = get_point(ratelist)
#%%
def vasicek_model(rate_points):
    rate_points = np.array(rate_points)
    X = rate_points[:,0]
    Y = rate_points[:,1]
    
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    a = results.params[0]
    b = results.params[1]
    sigma_e = (np.mean((results.resid)**2))
    print(np.mean(results.resid))
    print(a)
    print(b)
    L = a/(b)
    k = -np.log(b)
    sigma_r = ((sigma_e*2*k/(1-np.exp(-2*k))))
   
    return L,k,sigma_r

L,k,sigma_r = vasicek_model(rate_points)

#%%

def get_R(Rt,T,t,rate_points):
    L,k,sigma_r = vasicek_model(rate_points)
    B = 1/k*(1 - np.exp(-k*(T - t)))
    print('B',B)
    print('L',L)
    print('qq,',(L - sigma_r/(2*k**2))*(B - T + t) - B**2*sigma_r/(4*k))
    A = np.exp((L - sigma_r/(2*k**2))*(B - T + t) - B**2*sigma_r/(4*k))
    P = A*np.exp(-B*Rt)
    print('**')
    print(A)
    R = 1/(T - t)*(np.log(A)+B*Rt)
    print(Rt*np.exp(-k*(T - t))+L*(1-np.exp(k*(T - t))))
    return R

#%%
get_R(18.01,2,1,rate_points)