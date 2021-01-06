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
df_settings = df_settings.loc[inds]

df_results = pd.DataFrame(index=df_settings.index,columns=['C','sum_fixed_comfort','sum_LEM_comfort','supply_cost_fixed','supply_cost_LEM','C>=_fixed','C>=_LEM'],data=0.0)

for ind_WS in df_settings.index:
	
	df_HVAC = pd.read_csv(run+'/HVAC_settings/'+df_settings['settings_file'].loc[ind_WS],index_col=[0])
	start = pd.to_datetime(df_settings['start_time'].loc[ind_WS]) + pd.Timedelta(days=1)
	end = pd.to_datetime(df_settings['end_time'].loc[ind_WS])
	C = float(df_settings['line_capacity'].loc[ind_WS])
	print(ind_WS)
	print(C)

	df_results.at[ind_WS,'C'] = C

	folder_b = run + '/Diss_'+"{:04d}".format(ind_b) #+'_5min'
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

	#import pdb; pdb.set_trace()
	df_results.at[ind_WS,'sum_LEM_comfort'] = df_welfare['LEM_comfort'].sum()
	df_results.at[ind_WS,'sum_fixed_comfort'] = df_welfare['fixed_comfort'].sum()

	##################
	#
	# Calculate supply cost in fixed price scenario / no TS
	#
	##################

	folder = folder_b
	city = 'Austin'
	market_file = 'Ercot_HBSouth.csv'
	market_file = 'Ercot_LZ_SOUTH.csv'
	market_file = df_settings['market_data'].loc[ind_WS]
	market_file = 'Ercot_LZ_SOUTH.csv'

	df_slack = pd.read_csv(folder+'/load_node_149.csv',skiprows=range(8))
	df_slack['# timestamp'] = df_slack['# timestamp'].map(lambda x: str(x)[:-4])
	df_slack = df_slack.iloc[:-1]
	df_slack['# timestamp'] = pd.to_datetime(df_slack['# timestamp'])
	df_slack.set_index('# timestamp',inplace=True)
	df_slack = df_slack.loc[start:end]
	df_slack = df_slack/1000 #kW

	df_slack_C = df_slack.loc[df_slack['measured_real_power'] >= C]
	df_results.at[ind_WS,'C>=_fixed'] = (df_slack_C['measured_real_power']/12./1000.).sum() # in MWh

	df_WS = pd.read_csv('glm_generation_'+city+'/'+market_file,parse_dates=[0])
	df_WS.rename(columns={'Unnamed: 0':'timestamp'},inplace=True)
	df_WS.set_index('timestamp',inplace=True)
	df_WS = df_WS.loc[start:end]

	df_WS['system_load_fixed'] = df_slack['measured_real_power']
	df_WS['supply_cost_fixed'] = df_WS['system_load_fixed']/1000.*df_WS['RT']/12.

	df_results.at[ind_WS,'supply_cost_fixed'] = df_WS['supply_cost_fixed'].sum()

	df_slack = pd.read_csv(folder_WS+'/load_node_149.csv',skiprows=range(8))
	df_slack['# timestamp'] = df_slack['# timestamp'].map(lambda x: str(x)[:-4])
	df_slack = df_slack.iloc[:-1]
	df_slack['# timestamp'] = pd.to_datetime(df_slack['# timestamp'])
	df_slack.set_index('# timestamp',inplace=True)
	df_slack = df_slack.loc[start:end]
	df_slack = df_slack/1000 #kW

	df_slack_C = df_slack.loc[df_slack['measured_real_power'] >= C]
	df_results.at[ind_WS,'C>=_LEM'] = (df_slack_C['measured_real_power']/12./1000.).sum() # in MWh

	df_WS['system_load_LEM'] = df_slack['measured_real_power']
	df_WS['supply_cost_LEM'] = df_WS['system_load_LEM']/1000.*df_WS['RT']/12.

	df_results.at[ind_WS,'supply_cost_LEM'] = df_WS['supply_cost_LEM'].sum()

df_results.to_csv(run + '/' + 'welfare_changes_by_C_'+str(inds[0])+'.csv')

df_results['increase_comfort'] = df_results['sum_LEM_comfort'] - df_results['sum_fixed_comfort']
df_results['savings_supply_cost'] = df_results['supply_cost_fixed'] - df_results['supply_cost_LEM']
df_results['savings_cong_cost'] = (df_results['C>=_fixed'] - df_results['C>=_LEM'])*20
df_results['net_welfare_change'] = df_results['increase_comfort'] + df_results['savings_supply_cost'] + df_results['savings_cong_cost']

import pdb; pdb.set_trace()