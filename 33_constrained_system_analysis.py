#This file compares the WS market participation with stationary price bids to fixed price scenarie

#Includes the impact of the LEM under a constrained system on customers

import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt
import matplotlib.dates as mdates

run = 'Diss'
ind_b = 90 #year-long simulation without LEM or constraint
df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])

ind_Cs = [130,131,132,133,134,135,136,137]
ind_unconstrained = 127
ind_constrained = ind_Cs[0]

ind_WS = ind_unconstrained

start = pd.to_datetime(df_settings['start_time'].loc[ind_WS]) + pd.Timedelta(days=1)
end = pd.to_datetime(df_settings['end_time'].loc[ind_WS])
df_HVAC = pd.read_csv(run+'/HVAC_settings/'+df_settings['settings_file'].loc[ind_WS],index_col=[0])
city = df_settings['city'].loc[ind_WS]

folder_b = run + '/Diss_'+"{:04d}".format(ind_b) #+'_5min'
folder_WS = run + '/Diss_'+"{:04d}".format(ind_unconstrained)

# Calculate RR in benchmark case

folder = folder_b
market_file = df_settings['market_data'].loc[ind_WS]
market_file = 'Ercot_LZ_SOUTH.csv'

df_slack = pd.read_csv(folder+'/load_node_149.csv',skiprows=range(8))
df_slack['# timestamp'] = df_slack['# timestamp'].map(lambda x: str(x)[:-4])
df_slack = df_slack.iloc[:-1]
df_slack['# timestamp'] = pd.to_datetime(df_slack['# timestamp'])
df_slack.set_index('# timestamp',inplace=True)
df_slack = df_slack.loc[start:end]
df_slack = df_slack/1000 #kW

df_WS = pd.read_csv('glm_generation_'+city+'/'+market_file,parse_dates=[0])
df_WS.rename(columns={'Unnamed: 0':'timestamp'},inplace=True)
df_WS.set_index('timestamp',inplace=True)
df_WS = df_WS.loc[start:end]

df_WS['system_load'] = df_slack['measured_real_power']
supply_wlosses = (df_WS['system_load']/1000./12.).sum() # MWh
df_WS['supply_cost'] = df_WS['system_load']/1000.*df_WS['RT']/12.
supply_cost_wlosses = df_WS['supply_cost'].sum()

df_total_load = pd.read_csv(folder+'/total_load_all.csv',skiprows=range(8)) #in kW
df_total_load['# timestamp'] = df_total_load['# timestamp'].map(lambda x: str(x)[:-4])
df_total_load = df_total_load.iloc[:-1]
df_total_load['# timestamp'] = pd.to_datetime(df_total_load['# timestamp'])
df_total_load.set_index('# timestamp',inplace=True)
df_total_load = df_total_load.loc[start:end]
total_load = (df_total_load.sum(axis=1)/12.).sum() #kWh

df_WS['res_load'] = df_total_load.sum(axis=1)
supply_wolosses = (df_WS['res_load']/1000./12.).sum() # only residential load, not what is measured at trafo
df_WS['res_cost'] = df_WS['res_load']/1000.*df_WS['RT']/12.
supply_cost_wolosses = df_WS['res_cost'].sum()

try:
	df_inv_load = pd.read_csv(folder+'/total_P_Out.csv',skiprows=range(8)) #in W
	df_inv_load['# timestamp'] = df_inv_load['# timestamp'].map(lambda x: str(x)[:-4])
	df_inv_load = df_inv_load.iloc[:-1]
	df_inv_load['# timestamp'] = pd.to_datetime(df_inv_load['# timestamp'])
	df_inv_load.set_index('# timestamp',inplace=True)  
	df_PV_b = df_inv_load.loc[start:end]
	PV_supply = (df_inv_load.sum(axis=1)/1000./12.).sum() #in kWh
except:
	PV_supply = 0.0

print('RR with net-metering (w/o losses')

net_demand  = total_load - PV_supply
retail_kWh = supply_cost_wlosses/net_demand
retail_kWh_wolosses = supply_cost_wolosses/net_demand

retail_MWh = retail_kWh*1000.

###############
#
# Specific day
#
#################

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = []

#Load
#Benchmark / no LEM
lns += ax.plot(df_slack_b['measured_real_power']/1000000,color='0.75',label='Load benchmark')
#Constrained case
lns += ax.plot(df_slack_WS['measured_real_power']/1000000,color='0.25',label='Load LEM, C = '+str(df_settings['line_capacity'].loc[ind_constrained]/1000)+' MW')
ax.hlines(df_settings['line_capacity'].loc[ind_constrained]/1000,start,end,'0.25',':')

#Price
ax2 = ax.twinx()
lns += ax2.plot(df_prices_1min_b['clearing_price'],color='0.75',ls='--',label='WS price')
lns += ax2.plot(df_prices_1min_LEM['clearing_price'],color='0.25',ls='--',label='LEM price')

ax.set_xlabel('Time')
ax.set_xlim(start_peakday,end_peakday)
ax.set_ylim(0.0,2.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_ylabel('Aggregate system load [MW]')
ax2.set_ylabel('Price [USD/MWh]')
ax2.set_ylim(0.0,150.)
labs = [l.get_label() for l in lns]
L = ax.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
ppt.savefig('Diss/'++'_aggload_p_'+str(ind_constrained)+'.png', bbox_inches='tight')
ppt.savefig('Diss/'++'_aggload_p_'+str(ind_constrained)+'.pdf', bbox_inches='tight')

import pdb; pdb.set_trace()

# For different constraints

df_summary = pd.DataFrame(index=ind_Cs+[ind_unconstrained],columns=['line_capacity','mean_u','mean_p','comf_T','mean_T','mean_T_95','mean_T_5','min_T','min_T_95','min_T_5','var_T','var_T_95','var_T_5'],data=0)

for ind in [ind_unconstrained] + ind_Cs:
	print(ind)
	folder = 'Diss/Diss_'+"{:04d}".format(ind)
	C = df_settings['line_capacity'].loc[ind]

	df_welfare = pd.read_csv(folder + '/df_welfare_withparameters.csv',index_col=[0],parse_dates=True)
	#import pdb; pdb.set_trace()

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

ax.set_ylim(-8.,0.)
#labs = [l.get_label() for l in lns]
#L = ax.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
ppt.savefig('Diss/constraint_u_p_'+str(ind_unconstrained)+'.png', bbox_inches='tight')
ppt.savefig('Diss/constraint_u_p_'+str(ind_unconstrained)+'.pdf', bbox_inches='tight')

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
ppt.savefig('Diss/constraint_Trange_'+str(ind_unconstrained)+'.png', bbox_inches='tight')
ppt.savefig('Diss/constraint_Trange_'+str(ind_unconstrained)+'.pdf', bbox_inches='tight')

print(df_summary[['min_T_95','min_T','min_T_5','mean_p']])
print('Mean comfort temperature:')
print(df_welfare['comf_temperature'].mean())
print('Mean heating setpoint:')
print(df_welfare['heating_setpoint'].mean())
print('Min temperature drop')
print((df_summary['min_T']*df_summary['comf_T']).iloc[0] - (df_summary['min_T']*df_summary['comf_T']).iloc[-1])
import pdb; pdb.set_trace()