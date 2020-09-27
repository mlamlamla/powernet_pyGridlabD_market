#This file compares the WS market participation with stationary price bids to fixed price scenarie

#Includes the impact of the LEM under a constrained system on customers

import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt
import matplotlib.dates as mdates

ind_b = 46 #year-long simulation without LEM or constraint

ind_Cs = [57,56,55,54,58,59,60,61,62]
ind_unconstrained = 64
ind_constrained = ind_Cs[0]

if ind_constrained in [47,63]:
	start = pd.Timestamp(2016,8,1)
	end = pd.Timestamp(2016,8,8)
	retail_kWh = 0.0704575196993475 #first week of august
	df_HVAC = pd.read_csv('Diss/HVAC_settings_2016-08-01_2016-08-08_ext.csv',index_col=[0])
elif ind_constrained in [49,50,51,52,53,54,55,56,57,58,59,60,61,62,64]:
	start = pd.Timestamp(2016,12,19)
	end = pd.Timestamp(2016,12,26)
	retail_kWh = 0.02254690804746962 #mid-Dec
	df_HVAC = pd.read_csv('Diss/HVAC_settings_2016-12-19_2016-12-26_ext.csv',index_col=[0])

df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])

retail_MWh = retail_kWh*1000.

df_summary = pd.DataFrame(index=ind_Cs+[ind_unconstrained],columns=['line_capacity','mean_u','mean_p','comf_T','mean_T','mean_T_95','mean_T_5','min_T','min_T_95','min_T_5','var_T','var_T_95','var_T_5'],data=0)

for ind in ind_Cs + [ind_unconstrained]:
	folder = 'Diss/Diss_00'+str(ind)
	C = df_settings['line_capacity'].loc[ind]

	df_welfare = pd.read_csv(folder + '/df_welfare_withparameters.csv',index_col=[0],parse_dates=True)
	df_welfare = df_welfare.loc[df_welfare['heating_system'] != 'GAS']
	df_welfare['u_change'] = (df_welfare['LEM_u'] - df_welfare['LEM_cost']) - (df_welfare['fixed_u'] - df_welfare['fixed_cost'])
	df_summary['line_capacity'].loc[ind] = C
	df_summary['mean_u'].loc[ind] = df_welfare['u_change'].mean()
	df_summary['mean_T'].loc[ind] = (df_welfare['LEM_T_mean']/df_welfare['comf_temperature']).mean()
	df_summary['mean_T_95'].loc[ind] = (df_welfare['LEM_T_mean'].loc[df_welfare['alpha'] >= df_welfare['alpha'].quantile(q=0.95)]/df_welfare['comf_temperature'].loc[df_welfare['alpha'] >= df_welfare['alpha'].quantile(q=0.95)]).mean()
	df_summary['mean_T_5'].loc[ind] = (df_welfare['LEM_T_mean'].loc[df_welfare['alpha'] <= df_welfare['alpha'].quantile(q=0.05)]/df_welfare['comf_temperature'].loc[df_welfare['alpha'] <= df_welfare['alpha'].quantile(q=0.05)]).mean()
	df_summary['min_T'].loc[ind] = ((df_welfare['LEM_T_min']/df_welfare['comf_temperature'])).mean()
	df_summary['min_T_95'].loc[ind] = ((df_welfare['LEM_T_min']/df_welfare['comf_temperature'])).loc[df_welfare['alpha'] >= df_welfare['alpha'].quantile(q=0.9)].mean()
	df_summary['min_T_5'].loc[ind] = ((df_welfare['LEM_T_min']/df_welfare['comf_temperature'])).loc[df_welfare['alpha'] <= df_welfare['alpha'].quantile(q=0.1)].mean()
	#import pdb; pdb.set_trace()
	df_summary['var_T'].loc[ind] = (df_welfare['LEM_T_var']/df_welfare['fixed_T_var']).mean()
	df_summary['var_T_95'].loc[ind] = (df_welfare['LEM_T_var'].loc[df_welfare['alpha'] >= df_welfare['alpha'].quantile(q=0.95)]/df_welfare['fixed_T_var'].loc[df_welfare['alpha'] >= df_welfare['alpha'].quantile(q=0.95)]).mean()
	df_summary['var_T_5'].loc[ind] = (df_welfare['LEM_T_var'].loc[df_welfare['alpha'] <= df_welfare['alpha'].quantile(q=0.05)]/df_welfare['fixed_T_var'].loc[df_welfare['alpha'] <= df_welfare['alpha'].quantile(q=0.05)]).mean()
	df_summary['comf_T'].loc[ind] = df_welfare['comf_temperature'].mean()

	df_prices = pd.read_csv(folder+'/df_prices.csv',index_col=[0],parse_dates=True).loc[start:end]
	df_summary['mean_p'].loc[ind] = df_prices['clearing_price'].mean()
	print(C)
	print(df_welfare['LEM_T_min'].mean())

print(df_summary)

#df_summary = df_summary.iloc[1:]
df_summary['line_capacity'].loc[ind_unconstrained] = 2400
df_summary['diff_u'] = (df_summary['mean_u'] - df_summary['mean_u'].loc[ind_unconstrained])
df_summary.sort_values('line_capacity',inplace=True)
df_summary.set_index('line_capacity',inplace=True)
df_summary.index = df_summary.index/1000.

#import pdb; pdb.set_trace()

###############
#
# u and p in dep. of C
#
#################

#end = pd.Timestamp(2016,12,20)
fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = []
lns += ax.plot(df_summary['diff_u'],color='0.15',marker='x',label='Utility change')
ax.set_xlabel('Capacity constraint')
ppt.xticks(df_summary.index,df_summary.index[:-1].tolist() + ['$\\infty$'])
ax.set_ylabel('Utility change compared to\n an unconstrained system [USD]')

ax.set_ylim(-15.,0.)
#labs = [l.get_label() for l in lns]
#L = ax.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
ppt.savefig('Diss/constraint_u_p.png', bbox_inches='tight')
ppt.savefig('Diss/constraint_u_p.pdf', bbox_inches='tight')

print(df_summary[['mean_u','diff_u','mean_p']])
#import pdb; pdb.set_trace()

###############
#
# T range in dep. of C
#
#################

#end = pd.Timestamp(2016,12,20)
fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = []
lns += ax.plot(df_summary['min_T'],color='0.15',marker='x',label='All customers')
lns += ax.plot(df_summary['min_T_95'],color='0.15',marker='x',ls='--',label='Customers with $\\alpha \geq \\alpha^{90}$')
lns += ax.plot(df_summary['min_T_5'],color='0.15',marker='x',ls=':',label='Customers with $\\alpha \leq \\alpha^{10}$')
ax.set_xlabel('Capacity constraint [MW]')
ppt.xticks(df_summary.index,df_summary.index[:-1].tolist() + ['$\\infty$'])
ax.set_ylabel('$\\theta^{min}/\\theta^{comf}$')
ax2 = ax.twinx()
lns += ax2.plot(df_summary['mean_p'],color='0.5',marker='x',label='Mean LEM price')
ax2.set_ylabel('Price [USD/MWh]')
#ax2.set_ylim(0.0,100.)
labs = [l.get_label() for l in lns]
#L = ax.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
L = ax.legend(lns, labs, loc='center right', ncol=1)
ppt.savefig('Diss/constraint_Trange.png', bbox_inches='tight')
ppt.savefig('Diss/constraint_Trange.pdf', bbox_inches='tight')

print(df_summary[['min_T_95','min_T','min_T_5','mean_p']])
print('Mean comfort temperature:')
print(df_welfare['comf_temperature'].mean())
print('Mean heating setpoint:')
print(df_welfare['heating_setpoint'].mean())
print('Min temperature drop')
print((df_summary['min_T']*df_summary['comf_T']).iloc[0] - (df_summary['min_T']*df_summary['comf_T']).iloc[-1])
import pdb; pdb.set_trace()