import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit

from matplotlib import rcParams
rcParams['font.family'] = 'serif'

##name functions and variables so easy to understand
def f(x, m, c):
    return m*x + c
# hello
def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color="orchid", alpha=0.6)

    return ax

##seperate file with function for loading data, make sure can load data from different sources
##file can also have function for processing data
weights = pd.read_csv("data\weight_data.csv")
weights = weights.drop([0, 1, 2, 11, 20]) ##dropping repeated measurements
weights["datetime"] = pd.to_datetime(weights["Time of Measurement"]) ## relying on data being in particular format


weights["Date"] = weights["datetime"].dt.date
weights["Time"] = weights["datetime"].dt.strftime('%H:%m:%S')
weights["Day"] = np.arange(0,len(weights["Date"]))
weight_labels = weights
print(weight_labels["Weight(kg)"])

weights = weights.drop([6, 8]) ##dropping post-breakfast measurements
weights["sigma"] = np.ones(len(weights["Day"]))*0.3
n = weights["Day"].size

print(weights)

coeff, err = curve_fit(f, weights["Day"], weights["Weight(kg)"])
coeff_w_std, err_w_std = curve_fit(f, weights["Day"], weights["Weight(kg)"], sigma=weights["sigma"], absolute_sigma=True)

m = coeff.size
dof = n - m
model_weights = f(weights["Day"], coeff[0],coeff[1])
resid = weights["Weight(kg)"] - model_weights
s_err = np.sqrt(np.sum(resid ** 2) / dof)  #

tinv = lambda p, df: abs(stats.t.ppf(p/2, df))
ts = tinv(0.05, len(weights["Day"])-2)
t = stats.t.ppf(0.95, n - m)
print(f"slope (95%): {coeff[0]*7:.6f} +/- {ts*np.sqrt(err[0][0])*7:.6f}")
print(f"slope (95%): {coeff_w_std[0]*7:.6f} +/- {ts*np.sqrt(err_w_std[0][0])*7:.6f}")

x2 = np.linspace(np.min(weights["Day"]), np.max(weights["Day"]), 100)
y2 = f(x2, coeff[0], coeff[1])

plusminus = u"\u00B1"

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(weights["Day"], weights["Weight(kg)"], "*", color="mediumvioletred", markersize=8)
ax.plot(weights["Day"], model_weights, "-", color="mediumvioletred", linewidth=1.5, label="Fit", alpha=0.7)
ax.errorbar(weights["Day"], weights["Weight(kg)"], yerr=0.3, ls="None", c="mediumvioletred")
plot_ci_manual(t, s_err, n, weights["Day"], x2, y2, ax=ax)
ax.set_ylabel("Weight (kg)", fontname='serif', fontsize=12, weight='bold')
ax.set_xticks(weight_labels["Day"], labels=weight_labels["Date"], rotation="vertical", fontname='serif', fontsize=9)
#ax.set_yticklabels(weights["Weight(kg)"], fontsize=9, fontname='serif')
plt.text(weight_labels["Day"][4], weight_labels["Weight(kg)"][15],
         f"Weight gain: \n {coeff_w_std[0]*7:.3f}{plusminus}{ts*np.sqrt(err_w_std[0][0])*7:.3f} kg/week",
          fontsize=11, fontname='serif')
plt.tight_layout()
plt.savefig("weight.png")
plt.show()