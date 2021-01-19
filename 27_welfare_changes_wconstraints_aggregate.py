import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt
from sklearn.linear_model import LinearRegression

# This is actually 27_welfare_changes_wconstraints_plot.py

# Default
run = 'Diss'
ind_b = 90
folder_b = run + '/Diss_'+"{:04d}".format(ind_b) #+'_5min'

# Relevant runs for constraint evaluation

df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])
inds1 = [127,130,131,132,133,134,135,136,137]
inds2 = [128,138,139,140,141,142,143,144,145]
inds3 = [129,146,147,148,149,150,151,152,153]
inds4 = [206,209,210,211,212,213,214,215,216]
inds4 = [292,295,296,297,298,299,300,301,302]

ind_Cs = []

for cong_costs in [40.0,50.0,60.0]:

	print(cong_costs)

	results = []
	for inds in [inds4]:

		ind_15 = inds[-1] # most congested system

		df_results = pd.read_csv(run + '/' + 'welfare_changes_by_C_'+str(inds[0])+'.csv',index_col=0)

		df_results['net_welfare_change_fixed'] = 0.0 # change to fixed retail rate + 1.5 MW constraint
		df_results['net_welfare_change_LEM'] = 0.0 # change to LEM + 1.5 MW constraint

		for ind in df_results.index:
			welfare_LEM = df_results['sum_LEM_comfort'].loc[ind] - df_results['supply_cost_LEM'].loc[ind] - df_results['C>=_LEM'].loc[ind]*cong_costs
			welfare_fixed_15 = df_results['sum_fixed_comfort'].loc[ind_15] - df_results['supply_cost_fixed'].loc[ind_15] - df_results['C>=_fixed'].loc[ind_15]*cong_costs
			df_results.at[ind,'net_welfare_change_fixed'] = welfare_LEM - welfare_fixed_15
			welfare_fixed_15 = df_results['sum_LEM_comfort'].loc[ind_15] - df_results['supply_cost_LEM'].loc[ind_15] - df_results['C>=_LEM'].loc[ind_15]*cong_costs
			df_results.at[ind,'net_welfare_change_LEM'] = welfare_LEM - welfare_fixed_15

		df_results.at[inds[0],'C'] = 2400
		df_results.sort_values(by='C',inplace=True)
		df_results.set_index('C',inplace=True)	
		results += [df_results]
		print(df_results['net_welfare_change_fixed'])

#import pdb; pdb.set_trace()

df_results.index = df_results.index/1000.

cong_costs = 50.

# Comparee LEM to fixed retail rate under given constraint

df_results['wchange_LEM_40'] = df_results['sum_LEM_comfort'] - df_results['sum_fixed_comfort'] + df_results['C>=_fixed']*40. - df_results['C>=_LEM']*40. + df_results['supply_cost_fixed'] - df_results['supply_cost_LEM']
df_results['wchange_LEM_50'] = df_results['sum_LEM_comfort'] - df_results['sum_fixed_comfort'] + df_results['C>=_fixed']*50. - df_results['C>=_LEM']*50. + df_results['supply_cost_fixed'] - df_results['supply_cost_LEM']
df_results['wchange_LEM_60'] = df_results['sum_LEM_comfort'] - df_results['sum_fixed_comfort'] + df_results['C>=_fixed']*60. - df_results['C>=_LEM']*60. + df_results['supply_cost_fixed'] - df_results['supply_cost_LEM']

print(df_results['C>=_fixed']*cong_costs)

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns1 = ppt.plot(df_results['wchange_LEM_40'],label='Congestion costs 40 USD/MWh',marker='x',color='0.25')
lns2 = ppt.plot(df_results['wchange_LEM_50'],label='Congestion costs 50 USD/MWh',marker='x',color='0.5')
lns3 = ppt.plot(df_results['wchange_LEM_60'],label='Congestion costs 60 USD/MWh',marker='x',color='0.75')
ax.set_xlim(1.45,2.45)
ax.set_ylim(ymin=0.0)
ax.set_xlabel('Capacity constraint [MW]')
ppt.xticks(df_results.index,df_results.index[:-1].tolist() + ['$\\infty$'])
ax.set_ylabel('Welfare gains [USD]')
ax.legend()
ppt.savefig(run + '/27_wchange_compFixed_bycC'+'_'+str(inds4[0])+'.png', bbox_inches='tight')
ppt.savefig(run + '/27_wchange_compFixed_bycC'+'_'+str(inds4[0])+'.pdf', bbox_inches='tight')


# Investment perspective: compare LEM + grid expansion to fixed retail rate at 1.5 MW

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns1 = ppt.plot(df_results['sum_LEM_comfort'] - df_results['sum_fixed_comfort'],label='comfort change',marker='x',color='0.25')
lns2 = ppt.plot(df_results['supply_cost_fixed'] - df_results['supply_cost_LEM'],label='procurement cost savings',marker='x',color='0.5')
lns3 = ppt.plot(df_results['C>=_fixed'].loc[1.5]*cong_costs - df_results['C>=_LEM']*cong_costs,label='congestion cost savings',marker='x',color='0.75')
ax.set_xlabel('Capacity constraint [MW]')
ppt.xticks(df_results.index,df_results.index[:-1].tolist() + ['$\\infty$'])
ax.set_ylabel('Welfare gains [USD]')
lns = lns1 + lns2 + lns3
ax.legend()
ppt.savefig(run + '/27_wchange_compFixed15_bycC_disagg'+'_'+str(inds4[0])+'.png', bbox_inches='tight')
ppt.savefig(run + '/27_wchange_compFixed15_bycC_disagg'+'_'+str(inds4[0])+'.pdf', bbox_inches='tight')

# Calculate marginal value of grid investment

cong_costs = 50.

df_results['welfare_fixed'] = df_results['sum_fixed_comfort'] - df_results['supply_cost_fixed'] - df_results['C>=_fixed']*cong_costs
df_results['welfare_LEM'] = df_results['sum_LEM_comfort'] - df_results['supply_cost_LEM'] - df_results['C>=_LEM']*cong_costs

ind0 = df_results.index[0]
for ind in df_results.index[1:]:
	df_results.at[ind,'MVinv_fixed'] = df_results['welfare_fixed'].loc[ind] - df_results['welfare_fixed'].loc[ind0]
	df_results.at[ind,'MVinv_LEM'] = df_results['welfare_LEM'].loc[ind] - df_results['welfare_LEM'].loc[ind0]
	ind0 = ind

df_results_MV = df_results.copy()
df_results_MV.index = df_results_MV.index - 0.1
df_results_MV = df_results_MV.iloc[1:]

#import pdb; pdb.set_trace()

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns1 = ppt.plot(df_results_MV['MVinv_fixed'],label='Under fixed retail rate',marker='x',color='0.25')
lns2 = ppt.plot(df_results_MV['MVinv_LEM'],label='Under LEM',marker='x',color='0.5')
ax.set_xlabel('Capacity constraint [MW]')
xticks = [round(a,1) for a in df_results_MV.index[:-1].tolist()]
ppt.xticks(df_results_MV.index,xticks + ['$\\infty$'])
ax.set_ylabel('Marginal value of grid investment [USD]')
ax.legend()
ppt.savefig(run + '/27_MVinv_bycC_'+str(cong_costs)+'_'+str(inds4[0])+'.png', bbox_inches='tight')
ppt.savefig(run + '/27_MVinv_bycC_'+str(cong_costs)+'_'+str(inds4[0])+'.pdf', bbox_inches='tight')


import pdb; pdb.set_trace()
