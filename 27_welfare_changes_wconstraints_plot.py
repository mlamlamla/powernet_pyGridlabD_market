import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt
from sklearn.linear_model import LinearRegression

# Default
run = 'Diss'
ind_b = 90

# Find relevant runs
df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])
inds = [127,130,131,132,133,134,135,136,137]
inds = [128,138,139,140,141,142,143,144,145]
inds = [129,146,147,148,149,150,151,152,153]
ind_15 = inds[-1] # most congested system
df_settings = df_settings.loc[inds]

df_results = pd.read_csv(run + '/' + 'welfare_changes_by_C_'+str(inds[0])+'.csv',index_col=0)

df_results['net_welfare_change_fixed'] = 0.0 # change to fixed retail rate + 1.5 MW constraint
df_results['net_welfare_change_LEM'] = 0.0 # change to LEM + 1.5 MW constraint

cong_costs = 20.

for ind in df_results.index:
	welfare_LEM = df_results['sum_LEM_comfort'].loc[ind] - df_results['supply_cost_LEM'].loc[ind] - df_results['C>=_LEM'].loc[ind]*cong_costs
	welfare_fixed_15 = df_results['sum_fixed_comfort'].loc[ind_15] - df_results['supply_cost_fixed'].loc[ind_15] - df_results['C>=_fixed'].loc[ind_15]*cong_costs
	df_results.at[ind,'net_welfare_change_fixed'] = welfare_LEM - welfare_fixed_15
	welfare_fixed_15 = df_results['sum_LEM_comfort'].loc[ind_15] - df_results['supply_cost_LEM'].loc[ind_15] - df_results['C>=_LEM'].loc[ind_15]*cong_costs
	df_results.at[ind,'net_welfare_change_LEM'] = welfare_LEM - welfare_fixed_15

df_results['MVinv_fixed'] = 0.0 # change to fixed retail rate + 1.5 MW constraint
df_results['MVinv_LEM'] = 0.0 # change to LEM + 1.5 MW constraint

# Marginal value of investment

df_results.sort_values(by='C',ascending=True,inplace=True)

ind0 = df_results.index[0]
for ind in df_results.index[1:]:
	df_results.at[ind,'MVinv_fixed'] = df_results['net_welfare_change_fixed'].loc[ind] - df_results['net_welfare_change_fixed'].loc[ind0]
	df_results.at[ind,'MVinv_LEM'] = df_results['net_welfare_change_LEM'].loc[ind] - df_results['net_welfare_change_LEM'].loc[ind0]
	ind0 = ind

print(df_results)
df_results.to_csv(run + '/' + 'welfare_changes_by_C_'+str(ind_15)+'.csv')

df_results_137 = pd.read_csv(run + '/' + 'welfare_changes_by_C_'+str(137)+'.csv',index_col=['C'])
df_results_145 = pd.read_csv(run + '/' + 'welfare_changes_by_C_'+str(145)+'.csv',index_col=['C'])
df_results_153 = pd.read_csv(run + '/' + 'welfare_changes_by_C_'+str(153)+'.csv',index_col=['C'])
df_results_sum = df_results_137 + df_results_145 + df_results_153

import pdb; pdb.set_trace()