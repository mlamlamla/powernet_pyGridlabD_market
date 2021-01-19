import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt
from sklearn.linear_model import LinearRegression
from numpy.polynomial.polynomial import polyfit

# Builds on 25b_weeklywelfarechanges_compile.py

# Default
run = 'Diss'
ind_b = 90
folder_WS = run
no_houses = 437
inds0 = '203'

df_results = pd.read_csv(run + '/' + 'weekly_welfare_changes_b_'+inds0+'.csv',index_col=[0]) # from 25b_householdsavings_compile.py

print(len(df_results))
print('Summary')
print(df_results[['sum_fixed_comfort','sum_LEM_comfort','supply_cost_fixed','supply_cost_LEM','agg_welfare_change']].sum())
print('Max welfare change')
print(df_results['agg_welfare_change'].max())
print('Number of weeks with neg welfare change')
print(len(df_results.loc[df_results['agg_welfare_change'] < 0.0]))
print((df_results.loc[df_results['agg_welfare_change'] < 0.0]).mean()['agg_welfare_change'])
print('Correlation welfare changes <> Mean p')
print(df_results['weighted_mean_p'].corr(df_results['agg_welfare_change']))
print('Correlation welfare changes <> Max p')
print(df_results['max_p'].corr(df_results['agg_welfare_change']))
print('Correlation welfare changes <> Weighted std')
print(np.sqrt(df_results['weighted_var_p']).corr(df_results['agg_welfare_change']))
print('Correlation welfare changes <> Variance')
print(df_results['var_p'].corr(df_results['agg_welfare_change']))
print('Correlation welfare changes <> Weighted variance')
print(df_results['weighted_var_p'].corr(df_results['agg_welfare_change']))

#Histogram welfare change

max_y = 16
bw = 100
fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
#lns = ppt.hist(df_results['av_uchange'],bins=20,color='0.75',edgecolor='0.5')
lns = ppt.hist(df_results['agg_welfare_change'],bins=np.arange(-100,round(df_results['agg_welfare_change'].max()*1.2,0),bw),color='0.75',edgecolor='0.5')
#ax.set_ylim(0,75)
if df_results['agg_welfare_change'].min() > 0.0:
	ax.set_xlim(0,df_results['agg_welfare_change'].max()*1.05)
else:
	ax.vlines(0,0,max_y,'k',lw=1)
	#ax.set_xlim(xmin=(df_results['agg_welfare_change'].min()-2*bw),xmax=(df_results['agg_welfare_change'].max()+2*bw))
ax.set_xlabel('Welfare change [USD]')
if max_y > 0.0:
	ax.set_ylim(0,max_y)
ax.set_ylabel('Number of weeks')
ppt.savefig(folder_WS+'/25b_hist_wchange_year_'+str(inds0)+'.png', bbox_inches='tight')
ppt.savefig(folder_WS+'/25b_hist_wchange_year_'+str(inds0)+'.pdf', bbox_inches='tight')
#import pdb; pdb.set_trace()

# Dependence on system characteristics

# Mean p

reg = LinearRegression()
reg.fit(df_results['weighted_mean_p'].to_numpy().reshape(len(df_results),1),(df_results['agg_welfare_change']).to_numpy().reshape(len(df_results),1))
reg.coef_

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(df_results['weighted_mean_p'],df_results['agg_welfare_change'],marker='x',color='0.6')
lns2 = ppt.plot(df_results['weighted_mean_p'],reg.intercept_[0] + reg.coef_[0][0]*df_results['weighted_mean_p'],'-',color='0.25')
ax.set_xlabel('Weighted WS price [USD/MWh]')
ax.set_ylabel('Aggregate welfare change in a week [USD]')
ppt.savefig(folder_WS + '/25b_wchange_meanp_'+str(inds0)+'.png', bbox_inches='tight')
ppt.savefig(folder_WS + '/25b_wchange_meanp_'+str(inds0)+'.pdf', bbox_inches='tight')

print('Welfare change <> Weighted mean price')
print(str(reg.intercept_[0]) + ' + ' + str(reg.coef_[0][0]) + '* weighted_mean_p')

# Std p

reg = LinearRegression()
reg.fit(np.sqrt(df_results['weighted_var_p'].to_numpy().reshape(len(df_results),1)),(df_results['agg_welfare_change']).to_numpy().reshape(len(df_results),1))
reg.coef_

scale = 1.05

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(np.sqrt(df_results['weighted_var_p'].to_numpy()),df_results['agg_welfare_change'],marker='x',color='0.6')
lns2 = ppt.plot(np.sqrt(df_results['weighted_var_p'].to_numpy()),reg.intercept_[0] + reg.coef_[0][0]*np.sqrt(df_results['weighted_var_p'].to_numpy()),'-',color='0.25')
ax.set_xlabel('Weighted WS price standard variation [USD/MWh]')
ax.set_xlim(xmin=0.0,xmax=np.sqrt(df_results['weighted_var_p'].to_numpy()).max()*scale)
ax.set_ylabel('Aggregate welfare change in a week [USD]')
ax.hlines(0.,0.,np.sqrt(df_results['weighted_var_p'].to_numpy()).max()*scale,'k',lw=1)
#ax.set_ylim(ymin=0.0)
ppt.savefig(folder_WS + '/25b_wchange_stdp_'+str(inds0)+'.png', bbox_inches='tight')
ppt.savefig(folder_WS + '/25b_wchange_stdp_'+str(inds0)+'.pdf', bbox_inches='tight')

print('Welfare change <> Weighted std price')
print(str(reg.intercept_[0]) + ' + ' + str(reg.coef_[0][0]) + '* weighted_var_p')

# Var p

reg = LinearRegression()
reg.fit(df_results['weighted_var_p'].to_numpy().reshape(len(df_results),1),(df_results['agg_welfare_change']).to_numpy().reshape(len(df_results),1))
reg.coef_

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(df_results['weighted_var_p'],df_results['agg_welfare_change'],marker='x',color='0.6')
lns2 = ppt.plot(df_results['weighted_var_p'],reg.intercept_[0] + reg.coef_[0][0]*df_results['weighted_var_p'],'-',color='0.25')
ax.set_xlabel('Weighted WS price variance [USD2/MWh2]')
ax.set_ylabel('Aggregate welfare change in a week [USD]')
ppt.savefig(folder_WS + '/25b_wchange_varp_'+str(inds0)+'.png', bbox_inches='tight')
ppt.savefig(folder_WS + '/25b_wchange_varp_'+str(inds0)+'.pdf', bbox_inches='tight')

print('Welfare change <> Weighted var price')
print(str(reg.intercept_[0]) + ' + ' + str(reg.coef_[0][0]) + '* weighted_var_p')

import pdb; pdb.set_trace()
