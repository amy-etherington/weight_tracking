import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

weights=[]
weights = pd.read_csv("data\weight_data.csv")

weights = weights.drop([0, 1, 2])
weights["datetime"] = pd.to_datetime(weights["Time of Measurement"])
weights["Date"]=weights["datetime"].dt.date
weights["Time"]=weights["datetime"].dt.strftime('%H:%m:%S')
weights["Day"]=np.arange(0,len(weights["Date"]))
weights = weights.drop([6, 8])


def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color="orchid", alpha=0.5)

    return ax


def equation(a, b):
    """Return a 1D polynomial."""
    return np.polyval(a, b)


x = weights["Day"]
y = weights["Weight(kg)"]
p, cov = np.polyfit(x, y, 1, cov=True)
print(p[0]*7, np.sqrt(cov[0,0])*7)

# parameters and covariance from of the fit of 1-D polynom.
y_model = equation(p, x)  # model using the fit parameters; NOTE: parameters here are coefficients

# Statistics
n = weights["Day"].size  # number of observations
m = p.size  # number of parameters
dof = n - m  # degrees of freedom
t = stats.t.ppf(0.95, n - m)  # t-statistic; used for CI and PI bands

tinv = lambda p, df: abs(stats.t.ppf(p/2, df))
ts = tinv(0.05, len(weights["Day"])-2)
print(f"slope (95%): {p[0]*7:.6f} +/- {ts*np.sqrt(cov[0][0])*7:.6f}")

# Estimates of Error in Data/Model
resid = y - y_model  # residuals; diff. actual data from predicted values
chi2 = np.sum((resid / y_model) ** 2)  # chi-squared; estimates error in data
chi2_red = chi2 / dof  # reduced chi-squared; measures goodness of fit
s_err = np.sqrt(np.sum(resid ** 2) / dof)  # standard deviation of the error

# Plotting --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Data
ax.plot(
    x, y, "*", color="dodgerblue", markersize=8
)

# Fit
ax.plot(x, y_model, "-", color="dodgerblue", linewidth=1.5, label="Fit", alpha=0.7)
ax.errorbar(weights["Day"], weights["Weight(kg)"], yerr=0.3, ls="None", c="dodgerblue")

x2 = np.linspace(np.min(x), np.max(x), 100)
y2 = equation(p, x2)

# Confidence Interval (select one)
plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax)
# plot_ci_bootstrap(x, y, resid, ax=ax)
ax.set_ylabel("Weight (kg)")
ax.set_xlabel("Day")
ax.set_xticklabels(weights["Date"].astype(str),rotation=90)
plt.savefig("weight.png")
plt.tight_layout()
plt.show()


#plt.figure(figsize=(6,5))
#plt.scatter(weights["Day"], weights["Weight(kg)"], marker="*", c="dodgerblue")
#plt.errorbar(weights["Day"], weights["Weight(kg)"], yerr=0.3, ls="None", c="dodgerblue")
#plt.plot(weights["Day"], lin_regress.intercept + lin_regress.slope*weights["Day"], label='fitted line', c="darkorchid")
#plt.ylabel("Weight (kg)")
#plt.xlabel("Day")
#plt.tight_layout()
#plt.show()

