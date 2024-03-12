import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import t

weights=[]
weights = pd.read_csv("data\weight_data.csv")

weights = weights.drop([0, 1, 2])
weights["datetime"] = pd.to_datetime(weights["Time of Measurement"])
weights["Date"]=weights["datetime"].dt.date
weights["Time"]=weights["datetime"].dt.strftime('%H:%m:%S')
weights["Day"]=np.arange(0,len(weights["Date"]))

pre_breakkie_weights = weights.drop([6, 8])
weights = weights.set_index("datetime")
morning_weights = weights.between_time("00:00:00", "11:59:00")

lin_regress = stats.linregress(weights["Day"], weights["Weight(kg)"])
lin_regress_morning = stats.linregress(morning_weights["Day"], morning_weights["Weight(kg)"])
lin_regress_pre_breakkie = stats.linregress(pre_breakkie_weights["Day"], pre_breakkie_weights["Weight(kg)"])

tinv = lambda p, df: abs(t.ppf(p/2, df))
ts = tinv(0.05, len(weights["Day"])-2)
print(f"slope (95%): {lin_regress.slope*7:.6f} +/- {ts*lin_regress.stderr*7:.6f}")

ts = tinv(0.05, len(morning_weights["Day"])-2)
print(f"slope (95%): {lin_regress_morning.slope*7:.6f} +/- {ts*lin_regress_morning.stderr*7:.6f}")

ts = tinv(0.05, len(pre_breakkie_weights["Day"])-2)
print(f"slope (95%): {lin_regress_pre_breakkie.slope*7:.6f} +/- {ts*lin_regress_pre_breakkie.stderr*7:.6f}")


plt.figure(figsize=(6,5))
plt.scatter(weights["Date"].astype(str), weights["Weight(kg)"], marker="*", c="dodgerblue")
plt.errorbar(weights["Date"].astype(str), weights["Weight(kg)"], yerr=0.3, ls="None", c="dodgerblue")
plt.plot(weights["Day"], lin_regress.intercept + lin_regress.slope*weights["Day"], label='fitted line', c="darkorchid")
plt.ylabel("Weight (Stone)")
plt.xticks(rotation="vertical")
plt.xlabel("Date")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
plt.scatter(morning_weights["Day"], morning_weights["Weight(kg)"], marker="*", c="dodgerblue")
plt.errorbar(morning_weights["Day"], morning_weights["Weight(kg)"], yerr=0.3, ls="None", c="dodgerblue")
plt.plot(morning_weights["Day"], lin_regress_morning.intercept + lin_regress_morning.slope*morning_weights["Day"], label='fitted line', c="darkorchid")
plt.ylabel("Weight (kg)")
plt.xlabel("Day")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
plt.scatter(pre_breakkie_weights["Day"], pre_breakkie_weights["Weight(kg)"], marker="*", c="dodgerblue")
plt.errorbar(pre_breakkie_weights["Day"], pre_breakkie_weights["Weight(kg)"], yerr=0.3, ls="None", c="dodgerblue")
plt.plot(pre_breakkie_weights["Day"], lin_regress_pre_breakkie.intercept + lin_regress_pre_breakkie.slope*pre_breakkie_weights["Day"], label='fitted line', c="darkorchid")
plt.ylabel("Weight (kg)")
plt.xlabel("Day")
plt.tight_layout()
plt.show()

print(pre_breakkie_weights["Day"])