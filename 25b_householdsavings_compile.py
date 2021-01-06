#This file compares the WS market participation with stationary price bids to fixed price scenarie

# Difference to 25: Doesn't calculate the new retail rate but directly calculates welfare change from utility comfort change and energy procurement cost 
# (i.e. includes unresponsive load but this should cancel out - so same result)

import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt
from sklearn.linear_model import LinearRegression

# Default
run = 'Diss'
ind_b = 90

# Find relevant runs
df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])
inds = [157,159,129,160,161,162,163,164,165,166,167,168,156,169,170,171]
inds += [125,124,126,128,127]
inds += [*range(172,202)]
df_settings = df_settings.loc[inds]

try:
	df_results = pd.read_csv(run + '/' + 'weekly_welfare_changes_b.csv',index_col=[0])
except:
	df_results = pd.DataFrame(index=df_settings.index,columns=['max_p','var_p','weighted_mean_p','weighted_var_p','sum_fixed_comfort','sum_LEM_comfort','supply_cost_fixed','supply_cost_LEM'],data=0.0)

	for ind_WS in df_settings.index:
		
		df_HVAC = pd.read_csv(run+'/HVAC_settings/'+df_settings['settings_file'].loc[ind_WS],index_col=[0])
		start = pd.to_datetime(df_settings['start_time'].loc[ind_WS]) + pd.Timedelta(days=1)
		end = pd.to_datetime(df_settings['end_time'].loc[ind_WS])
		print(ind_WS)
		print(start)
		print(end)

		recread_data = True 
		recalculate_df_welfare = True

		house_no = 0

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

		df_hvac_load = pd.read_csv(folder_WS+'/hvac_load_all.csv',skiprows=range(8))
		df_hvac_load['# timestamp'] = df_hvac_load['# timestamp'].map(lambda x: str(x)[:-4])
		df_hvac_load = df_hvac_load.iloc[:-1]
		df_hvac_load['# timestamp'] = pd.to_datetime(df_hvac_load['# timestamp'])
		df_hvac_load.set_index('# timestamp',inplace=True) 
		df_hvac_load = df_hvac_load.loc[start:end]

		# In benchmark case

		df_T_b = pd.read_csv(folder_b+'/T_all.csv',skiprows=range(8))
		df_T_b['# timestamp'] = df_T_b['# timestamp'].map(lambda x: str(x)[:-4])
		df_T_b = df_T_b.iloc[:-1]
		df_T_b['# timestamp'] = pd.to_datetime(df_T_b['# timestamp'])
		df_T_b.set_index('# timestamp',inplace=True)
		df_T_b = df_T_b.loc[start:end]

		df_hvac_load_b = pd.read_csv(folder_b+'/hvac_load_all.csv',skiprows=range(8))
		df_hvac_load_b['# timestamp'] = df_hvac_load_b['# timestamp'].map(lambda x: str(x)[:-4])
		df_hvac_load_b = df_hvac_load_b.iloc[:-1]
		df_hvac_load_b['# timestamp'] = pd.to_datetime(df_hvac_load_b['# timestamp'])
		df_hvac_load_b.set_index('# timestamp',inplace=True) 
		df_hvac_load_b = df_hvac_load_b.loc[start:end]

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

		df_WS = pd.read_csv('glm_generation_'+city+'/'+market_file,parse_dates=[0])
		df_WS.rename(columns={'Unnamed: 0':'timestamp'},inplace=True)
		df_WS.set_index('timestamp',inplace=True)
		df_WS = df_WS.loc[start:end]

		df_WS['system_load_fixed'] = df_slack['measured_real_power']
		df_WS['supply_cost_fixed'] = df_WS['system_load_fixed']/1000.*df_WS['RT']/12.

		df_results.at[ind_WS,'max_p'] = df_WS['RT'].max()
		df_results.at[ind_WS,'var_p'] = df_WS['RT'].var()
		df_results.at[ind_WS,'weighted_mean_p'] = (df_WS['system_load_fixed']/df_WS['system_load_fixed'].sum()*df_WS['RT']).sum()
		df_results.at[ind_WS,'weighted_var_p'] = (df_WS['system_load_fixed']/df_WS['system_load_fixed'].sum()*(df_WS['RT'] - df_WS['RT'].mean()).pow(2)).sum()
		
		df_results.at[ind_WS,'supply_cost_fixed'] = df_WS['supply_cost_fixed'].sum()

		df_slack = pd.read_csv(folder_WS+'/load_node_149.csv',skiprows=range(8))
		df_slack['# timestamp'] = df_slack['# timestamp'].map(lambda x: str(x)[:-4])
		df_slack = df_slack.iloc[:-1]
		df_slack['# timestamp'] = pd.to_datetime(df_slack['# timestamp'])
		df_slack.set_index('# timestamp',inplace=True)
		df_slack = df_slack.loc[start:end]
		df_slack = df_slack/1000 #kW

		df_WS['system_load_LEM'] = df_slack['measured_real_power']
		df_WS['supply_cost_LEM'] = df_WS['system_load_LEM']/1000.*df_WS['RT']/12.

		df_results.at[ind_WS,'supply_cost_LEM'] = df_WS['supply_cost_LEM'].sum()

	df_results['agg_welfare_change'] = (df_results['sum_LEM_comfort'] - df_results['supply_cost_LEM']) - (df_results['sum_fixed_comfort'] - df_results['supply_cost_fixed'])
	df_results.to_csv(run + '/' + 'weekly_welfare_changes_b.csv')

print('Summary')
print(df_results.sum())
print('Correlation welfare changes <> Mean p')
print(df_results['weighted_mean_p'].corr(df_results['agg_welfare_change']))
print('Correlation welfare changes <> Max p')
print(df_results['max_p'].corr(df_results['agg_welfare_change']))
print('Correlation welfare changes <> Variance')
print(df_results['var_p'].corr(df_results['agg_welfare_change']))
print('Correlation welfare changes <> Weighted variance')
print(df_results['weighted_var_p'].corr(df_results['agg_welfare_change']))
