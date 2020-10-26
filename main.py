import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from lmfit import minimize, Parameters, Parameter, report_fit

# read data from global database
df_all = pd.read_csv("./data/total_data.csv", delimiter=',')
df_all.drop(df_all[df_all.iso_code!="AUT"].index, inplace=True)
df = df_all[["date", "total_cases", "total_tests"]].copy()
df.dropna(inplace=True)  # drop rows with empty entries
day0 = df['date'].iat[0]
df['date'] = (pd.to_datetime(df['date'], dayfirst=True).sub(pd.to_datetime(df['date'], dayfirst=True).iat[0])).dt.days
data = df.to_numpy().transpose()

''' sensitivity and specificity from [1].
Error sensitivity in percent = ((88%-84%)/2)/(sqrt(2)*erfinv(0.95))
& error specificity in percent = ((97%-94%)/2)/(sqrt(2)*erfinv(0.95)) 
interesting values: np.max(np.diff(data[2])[1:-1])/8.859e6 & np.max(data[2])/8.859e6'''
new_actual_positives_per_day_per_test = 0
new_actual_positives_per_day_per_test += unp.uarray(0.86*np.diff(data[1])/(np.diff(data[2])*np.diff(data[0])),
                                                    0.01*np.diff(data[1])/(np.diff(data[2])*np.diff(data[0])))
new_actual_positives_per_day_per_test += unp.uarray(.04*np.diff(data[2]-data[1])/(np.diff(data[2])*np.diff(data[0])),
                                                    0.00765*np.diff(data[2]-data[1])/(np.diff(data[2])*np.diff(data[0])))

'''-------- fitting constant --------'''
n = 100
n_estimate = data[1][70]
def const(x, c): return c
popt, pcov = curve_fit(const, data[0][n:int(1.5*n)], unp.nominal_values(new_actual_positives_per_day_per_test)[n:int(1.5*n)])
offset_infected = popt[0]
print(offset_infected)

'''-------- optimizing solution of ode --------'''
def sir(gamma, t, params):
    try:
        c1 = params['c1'].value
        c2 = params['c2'].value
    except KeyError:
        c1, c2 = params
    return [-c1*gamma[0]*gamma[1], c1*gamma[0]*gamma[1]-c2*gamma[1], c2*gamma[1]]

def g(t, x0, params):
    return odeint(sir, x0, t, args=(params,))

def residual(params, t, data):
    x0 = params['N'].value-1, 1, 0
    model = g(t, x0, params)

    # you only have data for one of your variables
    x2_model = model[:, 1]
    return (x2_model - data).ravel()

y0 = [n_estimate-1, 1, 0]
t_measured, x2_measured = data[0][1:n], unp.nominal_values(new_actual_positives_per_day_per_test)[:n-1]*n_estimate
paras = Parameters()
paras.add('N', value=n_estimate,           vary=False)  # data[1][70] is number of positive cases when it was roughly constant
paras.add('c1', value=4.5429e-05,     vary=True)
paras.add('c2', value=0.20587243,          vary=True)

paras_down = Parameters()
paras_down.add('N', value=n_estimate,           vary=False)  # data[1][70] is number of positive cases when it was roughly constant
paras_down.add('c1', value=4.5429e-05-1.1877e-06,     vary=False)
paras_down.add('c2', value=0.20587243-0.01581181,          vary=False)

paras_up = Parameters()
paras_up.add('N', value=n_estimate,           vary=False)  # data[1][70] is number of positive cases when it was roughly constant
paras_up.add('c1', value=4.5429e-05+1.1877e-06,     vary=False)
paras_up.add('c2', value=0.20587243+0.01581181,          vary=False)

t_fit = np.linspace(t_measured[0]-0, t_measured[-1], 1000)
result = minimize(residual, paras, args=(t_measured, x2_measured), method='leastsq')
data_fitted = g(t_fit, y0, result.params)
data_fitted_down = g(t_fit, y0, paras_down)
data_fitted_up = g(t_fit, y0, paras_up)
report_fit(result)

plt.plot(t_fit, data_fitted[:, 1]/n_estimate+offset_infected, '-', lw=.6)
plt.plot(t_fit, data_fitted_down[:, 1]/n_estimate+offset_infected, '-', lw=.6)
plt.plot(t_fit, data_fitted_up[:, 1]/n_estimate+offset_infected, '-', lw=.6)
plt.plot(data[0][1:n], unp.nominal_values(new_actual_positives_per_day_per_test)[:n-1], 'bo', markersize=1)
plt.plot(data[0][n:], unp.nominal_values(new_actual_positives_per_day_per_test)[n-1:], 'ko', markersize=1)
plt.errorbar(data[0][1:], unp.nominal_values(new_actual_positives_per_day_per_test),
             yerr=unp.std_devs(new_actual_positives_per_day_per_test), fmt='ro',
             linewidth=0.8, capsize=2, capthick=0.6, markersize=0)
plt.xlabel("days since " + day0)
plt.ylabel("proportion of infected people")
plt.show()
t_0, t_max = data[0][1], data[0][30]

''' References:
[1] Revista da Associação Médica Brasileira https://doi.org/10.1590/1806-9282.66.7.880 
'''
