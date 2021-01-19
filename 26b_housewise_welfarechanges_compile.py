#This file compares welfare change between benchmark to LEM/TS and attributes them by house

import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt
from sklearn.linear_model import LinearRegression

# Default
run = 'Diss'
ind_b = 364 #90

# Find relevant runs
df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])

# With wrong retail tariff calculation (duplicates in df_WS)
inds = [157,159,129,160,161,162,163,164,165,166,167,168,156,169,170,171]
inds += [125,124,126,128,127]
inds += [*range(172,202)]

inds = [*range(203,209)]
inds += [236,237]
inds += [*range(238,281)]

inds = [286,267]
inds = [288,203]
inds = [350,267]
inds = [292,206]
inds = [290,204]
inds = [293,207]

inds = [289,290,291,292,293,319,320,321,322,323,324,350]

df_settings = df_settings.loc[inds]

folder_b = run + '/Diss_'+"{:04d}".format(ind_b) #+'_5min'
city = df_settings['city'].loc[inds[0]] #'Austin'
market_file = df_settings['market_data'].loc[inds[0]]

df_T_b_year = pd.read_csv(folder_b+'/T_all.csv',skiprows=range(8))
df_T_b_year['# timestamp'] = df_T_b_year['# timestamp'].map(lambda x: str(x)[:-4])
df_T_b_year = df_T_b_year.iloc[:-1]
df_T_b_year['# timestamp'] = pd.to_datetime(df_T_b_year['# timestamp'])
df_T_b_year.set_index('# timestamp',inplace=True)

df_hvac_load_b_year = pd.read_csv(folder_b+'/hvac_load_all.csv',skiprows=range(8))
df_hvac_load_b_year['# timestamp'] = df_hvac_load_b_year['# timestamp'].map(lambda x: str(x)[:-4])
df_hvac_load_b_year = df_hvac_load_b_year.iloc[:-1]
df_hvac_load_b_year['# timestamp'] = pd.to_datetime(df_hvac_load_b_year['# timestamp'])
df_hvac_load_b_year.set_index('# timestamp',inplace=True) 

df_slack_b_year = pd.read_csv(folder_b+'/load_node_149.csv',skiprows=range(8))
df_slack_b_year['# timestamp'] = df_slack_b_year['# timestamp'].map(lambda x: str(x)[:-4])
df_slack_b_year = df_slack_b_year.iloc[:-1]
df_slack_b_year['# timestamp'] = pd.to_datetime(df_slack_b_year['# timestamp'])
df_slack_b_year.set_index('# timestamp',inplace=True)
df_slack_b_year = df_slack_b_year/1000 #kW

df_total_load_b_year = pd.read_csv(folder_b + '/total_load_all.csv',skiprows=range(8)) #in kW
df_total_load_b_year['# timestamp'] = df_total_load_b_year['# timestamp'].map(lambda x: str(x)[:-4])
df_total_load_b_year = df_total_load_b_year.iloc[:-1]
df_total_load_b_year['# timestamp'] = pd.to_datetime(df_total_load_b_year['# timestamp'])
df_total_load_b_year.set_index('# timestamp',inplace=True)

df_WS_year = pd.read_csv('glm_generation_'+city+'/'+market_file,parse_dates=[0])
df_WS_year.rename(columns={'Unnamed: 0':'timestamp'},inplace=True)
df_WS_year.drop_duplicates(subset='timestamp',keep='last',inplace=True)
df_WS_year.set_index('timestamp',inplace=True)

import pdb; pdb.set_trace()

#df_results = pd.DataFrame(index=df_settings.index,columns=['max_p','var_p','weighted_mean_p','weighted_var_p','sum_fixed_comfort','sum_LEM_comfort','supply_cost_fixed','supply_cost_LEM'],data=0.0)
# try:
# 	df_comfort = pd.read_csv(run + '/' + '26b_housewise_comfort_changes_'+str(inds[0])+'.csv',index_col=[0])
# 	df_savings = pd.read_csv(run + '/' + '26b_housewise_cost_changes_'+str(inds[0])+'.csv',index_col=[0])
# except:
# 	df_comfort = pd.DataFrame(index=df_T_b_year.columns,columns=inds,data=0.0)
# 	df_savings = df_comfort.copy()

df_comfort = pd.DataFrame(index=df_T_b_year.columns,columns=inds,data=0.0)
df_savings = df_comfort.copy()

for ind_WS in df_settings.index:

	#import pdb; pdb.set_trace()
	#if (ind_WS in df_comfort.columns) or (str(ind_WS) in df_comfort.columns):
	#	continue
	
	df_HVAC = pd.read_csv(run+'/HVAC_settings/'+df_settings['settings_file'].loc[ind_WS],index_col=[0])
	start = pd.to_datetime(df_settings['start_time'].loc[ind_WS]) + pd.Timedelta(days=1)
	end = pd.to_datetime(df_settings['end_time'].loc[ind_WS])
	print(str(ind_WS) + ': ' + str(start) + ' - ' + str(end))

	recread_data = True 
	recalculate_df_welfare = True

	house_no = 0

	folder_WS = run + '/Diss_'+"{:04d}".format(ind_WS)

	##################
	#
	# Evaluate comfort
	#
	##################

	# In TS case

	df_T = pd.read_csv(folder_WS+'/T_all.csv',skiprows=range(8))
	df_T['# timestamp'] = df_T['# timestamp'].map(lambda x: str(x)[:-4])
	df_T = df_T.iloc[:-1]
	df_T['# timestamp'] = pd.to_datetime(df_T['# timestamp'])
	df_T.set_index('# timestamp',inplace=True)
	df_T = df_T.loc[start:end]
	end = df_T.index[-1]

	df_hvac_load = pd.read_csv(folder_WS+'/hvac_load_all.csv',skiprows=range(8))
	df_hvac_load['# timestamp'] = df_hvac_load['# timestamp'].map(lambda x: str(x)[:-4])
	df_hvac_load = df_hvac_load.iloc[:-1]
	df_hvac_load['# timestamp'] = pd.to_datetime(df_hvac_load['# timestamp'])
	df_hvac_load.set_index('# timestamp',inplace=True) 
	df_hvac_load = df_hvac_load.loc[start:end]

	# In benchmark case

	df_T_b = df_T_b_year.loc[start:end]

	df_hvac_load_b = df_hvac_load_b_year.loc[start:end]

	# Calculate comfort component of utility for all houses

	if recalculate_df_welfare:
		df_u = df_T.copy()
		df_u_b = df_T_b.copy()
		#df_welfare = pd.DataFrame(index=df_u.columns,columns=['fixed_u','fixed_cost','fixed_T_mean','fixed_T_var','fixed_av_retail','LEM_u','LEM_cost','LEM_T_mean','LEM_T_var','LEM_av_retail'])
		#df_welfare = pd.DataFrame(index=df_u.columns,columns=['fixed_u','fixed_cost','fixed_T_mean','fixed_T_min','fixed_T_max','fixed_T_min5','fixed_T_max95','fixed_T_var','fixed_av_retail','LEM_u','LEM_cost','LEM_T_mean','LEM_T_min','LEM_T_max','LEM_T_min5','LEM_T_max95','LEM_T_var','LEM_av_retail'])
		df_welfare = pd.DataFrame(index=df_u.columns,columns=['LEM_comfort','fixed_comfort'])
		for col in df_u.columns:

			alpha = df_HVAC['alpha'].loc[col]
			T_com = df_HVAC['comf_temperature'].loc[col]
			
			df_u[col] = (df_u[col] - T_com)
			df_u[col] = -alpha*df_u[col].pow(2)
			df_welfare['LEM_comfort'].loc[col] = df_u[col].sum()
			
			df_u_b[col] = (df_u_b[col] - T_com)
			df_u_b[col] = -alpha*df_u_b[col].pow(2)
			df_welfare['fixed_comfort'].loc[col] = df_u_b[col].sum()

	df_welfare['change_comfort'] = df_welfare['LEM_comfort'] - df_welfare['fixed_comfort']
	assert len(df_comfort) == len(df_welfare)
	assert set(df_comfort.index) == set(df_welfare.index)
	df_comfort[ind_WS] = df_welfare['change_comfort'] # Time-wise aggregated comfort changes per house
	df_comfort.to_csv(run + '/' + '26b_housewise_comfort_changes_'+str(inds[0])+'.csv')

	##################
	#
	# Calculate supply cost in fixed price scenario / no TS
	#
	##################

	df_slack_b = df_slack_b_year.loc[start:end]
	df_total_load_b = df_total_load_b_year.loc[start:end]

	df_WS = df_WS_year.loc[start:end]

	assert len(df_WS) == len(df_slack_b)
	assert set(df_WS.index) == set(df_slack_b.index)
	assert len(df_WS) == len(df_total_load_b)
	assert set(df_WS.index) == set(df_total_load_b.index)

	df_cost_byhouse_b = pd.DataFrame(index=df_total_load_b.index,columns=df_total_load_b.columns,data=0.0)

	for ind in df_cost_byhouse_b.index:
		try:
			df_cost_byhouse_b.loc[ind] = df_total_load_b.div(df_total_load_b.sum(axis=1),axis=0).loc[ind]*df_slack_b['measured_real_power'].loc[ind]/1000./12.*df_WS['RT'].loc[ind]
		except:
			df_cost_byhouse_b.loc[ind] = df_total_load_b.div(df_total_load_b.sum(axis=1),axis=0).loc[ind]*df_slack_b['measured_real_power'].loc[ind]/1000./12.*df_WS['RT'].loc[ind].iloc[-1]

	df_slack_WS = pd.read_csv(folder_WS+'/load_node_149.csv',skiprows=range(8))
	df_slack_WS['# timestamp'] = df_slack_WS['# timestamp'].map(lambda x: str(x)[:-4])
	df_slack_WS = df_slack_WS.iloc[:-1]
	df_slack_WS['# timestamp'] = pd.to_datetime(df_slack_WS['# timestamp'])
	df_slack_WS.set_index('# timestamp',inplace=True)
	df_slack_WS = df_slack_WS.loc[start:end]
	df_slack_WS = df_slack_WS/1000 #kW

	df_total_load_WS = pd.read_csv(folder_WS + '/total_load_all.csv',skiprows=range(8)) #in kW
	df_total_load_WS['# timestamp'] = df_total_load_WS['# timestamp'].map(lambda x: str(x)[:-4])
	df_total_load_WS = df_total_load_WS.iloc[:-1]
	df_total_load_WS['# timestamp'] = pd.to_datetime(df_total_load_WS['# timestamp'])
	df_total_load_WS.set_index('# timestamp',inplace=True)
	df_total_load_WS = df_total_load_WS.loc[start:end]

	assert len(df_WS) == len(df_slack_WS)
	assert set(df_WS.index) == set(df_slack_WS.index)
	assert len(df_WS) == len(df_total_load_WS)
	assert set(df_WS.index) == set(df_total_load_WS.index)

	df_cost_byhouse_WS = pd.DataFrame(index=df_total_load_WS.index,columns=df_total_load_WS.columns,data=0.0)

	for ind in df_cost_byhouse_WS.index:
		try:
			df_cost_byhouse_WS.loc[ind] = df_total_load_WS.div(df_total_load_WS.sum(axis=1),axis=0).loc[ind]*df_slack_WS['measured_real_power'].loc[ind]/1000./12.*df_WS['RT'].loc[ind]
		except:
			import pdb; pdb.set_trace()

	df_savings[ind_WS] = (df_cost_byhouse_b - df_cost_byhouse_WS).sum(axis=0)
	import pdb; pdb.set_trace()
	df_savings.to_csv(run + '/' + '26b_housewise_cost_changes_'+str(inds[0])+'.csv')

# Aggregate

df_welfare = df_comfort + df_savings
df_welfare['agg_welfare_changes'] = df_welfare.sum(axis=1)
df_welfare.sort_values(by='agg_welfare_changes',ascending=False,inplace=True)

df_welfare.to_csv(run + '/' + '26b_housewise_welfare_changes_'+str(inds[0])+'.csv')

df_welfare['cum_welfare_change_year'] = 0.0
for house in df_welfare.index:
	df_welfare.at[house,'cum_welfare_change_year'] = df_welfare['agg_welfare_changes'].loc[:house].sum()

df_welfare['cum_welfare_change_year_rel'] = 100*df_welfare['cum_welfare_change_year']/df_welfare['cum_welfare_change_year'].max()
df_welfare['house_no_abs'] = range(1,len(df_welfare)+1)
df_welfare['house_no'] = range(1,len(df_welfare)+1)
df_welfare['house_no'] = 100.*df_welfare['house_no']/len(df_welfare)

df_welfare.to_csv(run + '/' + '26b_housewise_welfare_changes_'+str(inds[0])+'.csv')

df_welfare['alpha'] = df_HVAC['alpha']
df_welfare['R2'] = df_HVAC['R2']
df_welfare['P_cool'] = df_HVAC['P_cool']
df_welfare['P_heat'] = df_HVAC['P_heat']
df_welfare['gamma_cool'] = df_HVAC['gamma_cool']
df_welfare['gamma_heat'] = df_HVAC['gamma_heat']
df_welfare['cooling_setpoint'] = df_HVAC['cooling_setpoint']
df_welfare['heating_setpoint'] = df_HVAC['heating_setpoint']

import pdb; pdb.set_trace()

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
ppt.savefig(run + '/26b_welfare_cdf_'+str(inds[0])+'.png', bbox_inches='tight')
ppt.savefig(run + '/26b_welfare_cdf_'+str(inds[0])+'.pdf', bbox_inches='tight')

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
