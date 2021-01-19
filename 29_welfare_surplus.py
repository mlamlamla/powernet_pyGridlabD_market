#This file compiles the year-long customer surplus and welfare gain per house

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

inds = [124,203]

df_settings = df_settings.loc[inds]

for ind_WS in df_settings.index:

	df_HVAC = pd.read_csv(run+'/HVAC_settings/'+df_settings['settings_file'].loc[ind_WS],index_col=[0])
	start = pd.to_datetime(df_settings['start_time'].loc[ind_WS]) + pd.Timedelta(days=1)
	end = pd.to_datetime(df_settings['end_time'].loc[ind_WS])
	print(ind_WS)
	print(start)
	print(end)

	folder_WS = run + '/Diss_'+"{:04d}".format(ind_WS)

	# To determine start and end

	df_T = pd.read_csv(folder_WS+'/T_all.csv',skiprows=range(8))
	df_T['# timestamp'] = df_T['# timestamp'].map(lambda x: str(x)[:-4])
	df_T = df_T.iloc[:-1]
	df_T['# timestamp'] = pd.to_datetime(df_T['# timestamp'])
	df_T.set_index('# timestamp',inplace=True)
	df_T = df_T.loc[start:end]
	end = df_T.index[-1]

	###################
	#
	# Welfare change
	#
	###################

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
	sum_comfort_change = df_welfare['change_comfort'].sum()
	print('Welfare change: ' + str(sum_comfort_change))

	# Calculate procurement cost

	city = 'Austin'
	market_file = df_settings['market_data'].loc[ind_WS]
	market_file = 'Ercot_LZ_SOUTH.csv'

	df_WS = pd.read_csv('glm_generation_'+city+'/'+market_file,parse_dates=[0])
	df_WS.rename(columns={'Unnamed: 0':'timestamp'},inplace=True)
	df_WS.drop_duplicates(subset='timestamp',keep='last',inplace=True)
	df_WS.set_index('timestamp',inplace=True)
	df_WS = df_WS.loc[start:end]

	df_slack_b = pd.read_csv(folder_b + '/load_node_149.csv',skiprows=range(8))
	df_slack_b['# timestamp'] = df_slack_b['# timestamp'].map(lambda x: str(x)[:-4])
	df_slack_b = df_slack_b.iloc[:-1]
	df_slack_b['# timestamp'] = pd.to_datetime(df_slack_b['# timestamp'])
	df_slack_b.set_index('# timestamp',inplace=True)
	df_slack_b = df_slack_b.loc[start:end]
	df_slack_b = df_slack_b/1000 #kW

	df_slack_WS = pd.read_csv(folder_WS + '/load_node_149.csv',skiprows=range(8))
	df_slack_WS['# timestamp'] = df_slack_WS['# timestamp'].map(lambda x: str(x)[:-4])
	df_slack_WS = df_slack_WS.iloc[:-1]
	df_slack_WS['# timestamp'] = pd.to_datetime(df_slack_WS['# timestamp'])
	df_slack_WS.set_index('# timestamp',inplace=True)
	df_slack_WS = df_slack_WS.loc[start:end]
	df_slack_WS = df_slack_WS/1000 #kW

	assert len(df_WS) == len(df_slack_b)
	assert len(df_WS) == len(df_slack_WS)

	df_WS['system_load_b'] = df_slack_b['measured_real_power']
	supply_wlosses_b = (df_WS['system_load_b']/1000./12.).sum() # MWh
	df_WS['supply_cost_b'] = df_WS['system_load_b']/1000.*df_WS['RT']/12.
	supply_cost_wlosses_b = df_WS['supply_cost_b'].sum()

	df_WS['system_load_WS'] = df_slack_WS['measured_real_power']
	supply_wlosses_WS = (df_WS['system_load_WS']/1000./12.).sum() # MWh
	df_WS['supply_cost_WS'] = df_WS['system_load_WS']/1000.*df_WS['RT']/12.
	supply_cost_wlosses_WS = df_WS['supply_cost_WS'].sum()

	# Evaluate system welfare

	print('Supply cost savings: ' + str(supply_cost_wlosses_b - supply_cost_wlosses_WS))
	print('Welfare change: ' + str(sum_comfort_change + supply_cost_wlosses_b - supply_cost_wlosses_WS))

	# Assign system welfare changes to houses

	df_total_load_b = pd.read_csv(folder_b + '/total_load_all.csv',skiprows=range(8)) #in kW
	df_total_load_b['# timestamp'] = df_total_load_b['# timestamp'].map(lambda x: str(x)[:-4])
	df_total_load_b = df_total_load_b.iloc[:-1]
	df_total_load_b['# timestamp'] = pd.to_datetime(df_total_load_b['# timestamp'])
	df_total_load_b.set_index('# timestamp',inplace=True)
	df_total_load_b = df_total_load_b.loc[start:end]
	total_load_b = (df_total_load_b.sum(axis=1)/12.).sum() #kWh

	df_cost_byhouse_b = pd.DataFrame(index=df_total_load_b.index,columns=df_total_load_b.columns,data=0.0)
	for ind in df_cost_byhouse_b.index:
		try:
			df_cost_byhouse_b.loc[ind] = df_total_load_b.div(df_total_load_b.sum(axis=1),axis=0).loc[ind]*df_slack_b['measured_real_power'].loc[ind]/1000./12.*df_WS['RT'].loc[ind]
		except:
			df_cost_byhouse_b.loc[ind] = df_total_load_b.div(df_total_load_b.sum(axis=1),axis=0).loc[ind]*df_slack_b['measured_real_power'].loc[ind]/1000./12.*df_WS['RT'].loc[ind].iloc[-1]

	df_total_load_WS = pd.read_csv(folder_WS + '/total_load_all.csv',skiprows=range(8)) #in kW
	df_total_load_WS['# timestamp'] = df_total_load_WS['# timestamp'].map(lambda x: str(x)[:-4])
	df_total_load_WS = df_total_load_WS.iloc[:-1]
	df_total_load_WS['# timestamp'] = pd.to_datetime(df_total_load_WS['# timestamp'])
	df_total_load_WS.set_index('# timestamp',inplace=True)
	df_total_load_WS = df_total_load_WS.loc[start:end]
	total_load_WS = (df_total_load_WS.sum(axis=1)/12.).sum() #kWh	

	df_cost_byhouse_WS = pd.DataFrame(index=df_total_load_WS.index,columns=df_total_load_WS.columns,data=0.0)
	for ind in df_cost_byhouse_WS.index:
		try:
			df_cost_byhouse_WS.loc[ind] = df_total_load_WS.div(df_total_load_WS.sum(axis=1),axis=0).loc[ind]*df_slack_WS['measured_real_power'].loc[ind]/1000./12.*df_WS['RT'].loc[ind]
		except:
			#df_cost_byhouse_WS.loc[ind] = df_total_load_WS.div(df_total_load_WS.sum(axis=1),axis=0).loc[ind]*df_slack_WS['measured_real_power'].loc[ind]/1000./12.*df_WS['RT'].loc[ind].iloc[-1]
			import pdb; pdb.set_trace()

	assert round(df_cost_byhouse_b.sum().sum() - df_cost_byhouse_WS.sum().sum(),0) == round(supply_cost_wlosses_b - supply_cost_wlosses_WS,0)
	#df_savings[ind_WS] = (df_cost_byhouse_b - df_cost_byhouse_WS).sum(axis=0)

	import pdb; pdb.set_trace()

	# ##################
	# #
	# # Calculate retail rate in fixed price scenario / no TS
	# #
	# ##################

	# df_total_load_b = pd.read_csv(folder_b + '/total_load_all.csv',skiprows=range(8)) #in kW
	# df_total_load_b['# timestamp'] = df_total_load_b['# timestamp'].map(lambda x: str(x)[:-4])
	# df_total_load_b = df_total_load_b.iloc[:-1]
	# df_total_load_b['# timestamp'] = pd.to_datetime(df_total_load_b['# timestamp'])
	# df_total_load_b.set_index('# timestamp',inplace=True)
	# df_total_load_b = df_total_load_b.loc[start:end]
	# total_load_b = (df_total_load_b.sum(axis=1)/12.).sum() #kWh

	# df_WS['res_load'] = df_total_load_b.sum(axis=1)
	# #supply_res_wolosses = (df_WS['res_load']/1000./12.).sum() # only residential load, not what is measured at trafo
	# df_WS['res_cost'] = df_WS['res_load']/1000.*df_WS['RT']/12.
	# #supply_res_cost_wolosses = df_WS['res_cost'].sum()

	# try:
	# 	df_inv_load = pd.read_csv(folder_b + '/total_P_Out.csv',skiprows=range(8)) #in W
	# 	df_inv_load['# timestamp'] = df_inv_load['# timestamp'].map(lambda x: str(x)[:-4])
	# 	df_inv_load = df_inv_load.iloc[:-1]
	# 	df_inv_load['# timestamp'] = pd.to_datetime(df_inv_load['# timestamp'])
	# 	df_inv_load.set_index('# timestamp',inplace=True)  
	# 	df_inv_load = df_inv_load.loc[start:end]
	# 	PV_supply = (df_inv_load.sum(axis=1)/1000./12.).sum() #in kWh
	# except:
	# 	PV_supply = 0.0

	# net_demand_b  = total_load_b - PV_supply
	# retail_kWh = supply_cost_wlosses/net_demand_b
	# #retail_kWh_wolosses = supply_cost_wolosses/net_demand

	# retail_MWh = retail_kWh*1000.
	# print('Retail rate with df_WS with duplicates :' + str(retail_MWh))

	# df_WS['time'] = df_WS.index
	# df_WS.drop_duplicates(subset='time',keep='last',inplace=True)

	# list_duplicates = []
	# for dt in df_WS.index:
	# 	list_times = df_WS.index.to_list()
	# 	if (list_times.count(dt) > 1):
	# 		list_duplicates += [dt]
	# print(list_duplicates)
	
	# supply_wlosses = (df_WS['system_load']/1000./12.).sum() # MWh
	# supply_cost_wlosses = df_WS['supply_cost'].sum()
	# retail_kWh = supply_cost_wlosses/net_demand_b
	# retail_MWh = retail_kWh*1000.

	# print('Retail rate with correct df_WS: ' + str(retail_MWh))


