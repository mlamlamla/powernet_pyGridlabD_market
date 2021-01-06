#This file compiles the year-long savings per house

import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt
from sklearn.linear_model import LinearRegression

# Default
run = 'Diss'
ind_b = 90
folder_b = run + '/Diss_'+"{:04d}".format(ind_b) #+'_5min'

# Find relevant runs
df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])
inds = [157,159,129,160,161,162,163,164,165,166,167,168,156,169,170,171]
inds += [125,124,126,128,127]
inds += [*range(172,202)]

df_settings = df_settings.loc[inds]

##################
#
# Evaluate comfort
#
##################

df_T = pd.read_csv(folder_b+'/T_all.csv',skiprows=range(8))
df_comfort = pd.DataFrame(index=df_T.columns[1:],columns=inds,data=0.0)

print('Welfare changes')

for ind_WS in df_settings.index:
	
	df_HVAC = pd.read_csv(run+'/HVAC_settings/'+df_settings['settings_file'].loc[ind_WS],index_col=[0])
	start = pd.to_datetime(df_settings['start_time'].loc[ind_WS]) + pd.Timedelta(days=1)
	end = pd.to_datetime(df_settings['end_time'].loc[ind_WS])
	print(ind_WS)
	print(start)
	print(end)

	folder_WS = run + '/Diss_'+"{:04d}".format(ind_WS)

	# In TS case

	df_T = pd.read_csv(folder_WS+'/T_all.csv',skiprows=range(8))
	df_T['# timestamp'] = df_T['# timestamp'].map(lambda x: str(x)[:-4])
	df_T = df_T.iloc[:-1]
	df_T['# timestamp'] = pd.to_datetime(df_T['# timestamp'])
	df_T.set_index('# timestamp',inplace=True)
	df_T = df_T.loc[start:end]
	end = df_T.index[-1]

	# In benchmark case

	df_T_b = pd.read_csv(folder_b+'/T_all.csv',skiprows=range(8))
	df_T_b['# timestamp'] = df_T_b['# timestamp'].map(lambda x: str(x)[:-4])
	df_T_b = df_T_b.iloc[:-1]
	df_T_b['# timestamp'] = pd.to_datetime(df_T_b['# timestamp'])
	df_T_b.set_index('# timestamp',inplace=True)
	df_T_b = df_T_b.loc[start:end]

	# Calculate comfort component of utility for all houses

	df_u = df_T.copy()
	df_u_b = df_T_b.copy()
	
	df_welfare = pd.DataFrame(index=df_u.columns,columns=['LEM_comfort','fixed_comfort','change_comfort'])
	for col in df_u.columns:

		alpha = df_HVAC['alpha'].loc[col]
		T_com = df_HVAC['comf_temperature'].loc[col]
		
		df_u[col] = (df_u[col] - T_com)
		df_u[col] = -alpha*df_u[col].pow(2)
		df_welfare['LEM_comfort'].loc[col] = df_u[col].sum()
		
		df_u_b[col] = (df_u_b[col] - T_com)
		df_u_b[col] = -alpha*df_u_b[col].pow(2)
		df_welfare['fixed_comfort'].loc[col] = df_u_b[col].sum()

	# Save comfort change by each house

	df_welfare['change_comfort'] = df_welfare['LEM_comfort'] - df_welfare['fixed_comfort']
	df_comfort[ind_WS] = df_welfare['change_comfort'] # Time-wise aggregated comfort changes per house

df_comfort.to_csv(run + '/' + '26b_housewise_comfort_changes.csv')

##################
#
# Evaluate energy procurement cost
#
##################

print('Energy procurement cost')

df_savings = df_comfort.copy()

for ind_WS in df_settings.index:
	
	df_HVAC = pd.read_csv(run+'/HVAC_settings/'+df_settings['settings_file'].loc[ind_WS],index_col=[0])
	start = pd.to_datetime(df_settings['start_time'].loc[ind_WS]) + pd.Timedelta(days=1)
	end = pd.to_datetime(df_settings['end_time'].loc[ind_WS])
	print(ind_WS)
	print(start)
	print(end)

	folder_WS = run + '/Diss_'+"{:04d}".format(ind_WS)

	# WS market cost

	city = 'Austin'
	market_file = 'Ercot_LZ_SOUTH.csv'

	df_WS = pd.read_csv('glm_generation_'+city+'/'+market_file,parse_dates=[0])
	df_WS.rename(columns={'Unnamed: 0':'timestamp'},inplace=True)
	df_WS.set_index('timestamp',inplace=True)
	df_WS = df_WS.loc[start:end]

	# In TS case

	df_slack = pd.read_csv(folder_WS + '/load_node_149.csv',skiprows=range(8))
	df_slack['# timestamp'] = df_slack['# timestamp'].map(lambda x: str(x)[:-4])
	df_slack = df_slack.iloc[:-1]
	df_slack['# timestamp'] = pd.to_datetime(df_slack['# timestamp'])
	df_slack.set_index('# timestamp',inplace=True)
	df_slack = df_slack.loc[start:end]
	df_slack = df_slack/1000 #kW
	end = df_slack.index[-1]

	df_total_load = pd.read_csv(folder_WS + '/total_load_all.csv',skiprows=range(8)) #in kW
	df_total_load['# timestamp'] = df_total_load['# timestamp'].map(lambda x: str(x)[:-4])
	df_total_load = df_total_load.iloc[:-1]
	df_total_load['# timestamp'] = pd.to_datetime(df_total_load['# timestamp'])
	df_total_load.set_index('# timestamp',inplace=True)
	df_total_load = df_total_load.loc[start:end]
	total_load = (df_total_load.sum(axis=1)/12.).sum() #kWh

	df_cost_byhouse = pd.DataFrame(index=df_total_load.index,columns=df_total_load.columns,data=0.0)

	for ind in df_cost_byhouse.index:
		try:
			df_cost_byhouse.loc[ind] = df_total_load.div(df_total_load.sum(axis=1),axis=0).loc[ind]*df_slack['measured_real_power'].loc[ind]/1000./12.*df_WS['RT'].loc[ind]
		except:
			df_cost_byhouse.loc[ind] = df_total_load.div(df_total_load.sum(axis=1),axis=0).loc[ind]*df_slack['measured_real_power'].loc[ind]/1000./12.*df_WS['RT'].loc[ind].iloc[-1]

	# In benchmark case

	df_slack = pd.read_csv(folder_b + '/load_node_149.csv',skiprows=range(8))
	df_slack['# timestamp'] = df_slack['# timestamp'].map(lambda x: str(x)[:-4])
	df_slack = df_slack.iloc[:-1]
	df_slack['# timestamp'] = pd.to_datetime(df_slack['# timestamp'])
	df_slack.set_index('# timestamp',inplace=True)
	df_slack = df_slack.loc[start:end]
	df_slack = df_slack/1000 #kW
	df_slack = df_slack.loc[start:end]

	df_total_load = pd.read_csv(folder_b + '/total_load_all.csv',skiprows=range(8)) #in kW
	df_total_load['# timestamp'] = df_total_load['# timestamp'].map(lambda x: str(x)[:-4])
	df_total_load = df_total_load.iloc[:-1]
	df_total_load['# timestamp'] = pd.to_datetime(df_total_load['# timestamp'])
	df_total_load.set_index('# timestamp',inplace=True)
	df_total_load = df_total_load.loc[start:end]
	total_load = (df_total_load.sum(axis=1)/12.).sum() #kWh

	df_cost_byhouse_b = pd.DataFrame(index=df_total_load.index,columns=df_total_load.columns,data=0.0)
	for ind in df_cost_byhouse_b.index:
		try:
			df_cost_byhouse_b.loc[ind] = df_total_load.div(df_total_load.sum(axis=1),axis=0).loc[ind]*df_slack['measured_real_power'].loc[ind]/1000./12.*df_WS['RT'].loc[ind]
		except:
			df_cost_byhouse_b.loc[ind] = df_total_load.div(df_total_load.sum(axis=1),axis=0).loc[ind]*df_slack['measured_real_power'].loc[ind]/1000./12.*df_WS['RT'].loc[ind].iloc[-1]
	
	df_savings[ind_WS] = (df_cost_byhouse_b - df_cost_byhouse).sum(axis=0)

df_savings.to_csv(run + '/' + '26b_housewise_cost_changes.csv')

# Aggregate

df_welfare = df_comfort + df_savings
df_welfare['agg_welfare_changes'] = df_welfare.sum(axis=1)
df_welfare.sort_values(by='agg_welfare_changes',ascending=False,inplace=True)

df_welfare.to_csv(run + '/' + '26b_housewise_welfare_changes.csv')

df_welfare['cum_welfare_change_year'] = 0.0
for house in df_welfare.index:
	df_welfare.at[house,'cum_welfare_change_year'] = df_welfare['agg_welfare_changes'].loc[:house].sum()

df_welfare['cum_welfare_change_year_rel'] = 100*df_welfare['cum_welfare_change_year']/df_welfare['cum_welfare_change_year'].max()
df_welfare['house_no_abs'] = range(1,len(df_welfare)+1)
df_welfare['house_no'] = range(1,len(df_welfare)+1)
df_welfare['house_no'] = 100.*df_welfare['house_no']/len(df_welfare)

df_welfare.to_csv(run + '/' + '26b_housewise_welfare_changes.csv')

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.plot(df_welfare['house_no'],df_welfare['cum_welfare_change_year_rel'],color='0.6')
#import pdb; pdb.set_trace()
ax.set_xlabel('Share of houses in LEM [%]')
#ax.set_xticks(range(0,110,10))
ax.set_xlim(0.0,100.)
ax.set_ylim(0.0,100.)
#ax.set_ylim(bottom=0.0)
ax.set_ylabel('Realized welfare gains [%]')
ppt.savefig(run + '/26b_welfare_cdf.png', bbox_inches='tight')
ppt.savefig(run + '/26b_welfare_cdf.pdf', bbox_inches='tight')

print('50% most valuable houses realize [%] welfare gains')
print(df_welfare.loc[df_welfare['house_no'] < 50.].iloc[-1]['cum_welfare_change_year_rel'])

print('Max welfare gain at:')
print(df_welfare.loc[df_welfare['cum_welfare_change_year_rel'] == df_welfare['cum_welfare_change_year_rel'].max()])
print(df_welfare['house_no_abs'].loc[df_welfare['cum_welfare_change_year_rel'] == df_welfare['cum_welfare_change_year_rel'].max()])

import pdb; pdb.set_trace()

reg = LinearRegression()
reg.fit(df_results['RR'].to_numpy().reshape(len(df_results),1),(df_results['av_uchange']).to_numpy().reshape(len(df_results),1))
reg.coef_

reg2 = LinearRegression()
reg2.fit(df_results['max_p'].to_numpy().reshape(len(df_results),1),(df_results['av_uchange']).to_numpy().reshape(len(df_results),1))
reg2.coef_

reg3 = LinearRegression()
reg3.fit(df_results['var_p'].to_numpy().reshape(len(df_results),1),(df_results['av_uchange']).to_numpy().reshape(len(df_results),1))
reg3.coef_