#This file compares the WS market participation with stationary price bids to fixed price scenarie

import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt
from sklearn.linear_model import LinearRegression
import pdb; pdb.set_trace()

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
	df_results = pd.read_csv(run + '/' + 'weekly_welfare_changes.csv',index_col=[0])
except:
	df_results = pd.DataFrame(index=df_settings.index,columns=['RR','RR_wolosses','max_p','var_p','weighted_var_p','u_change','av_uchange'],data=0.0)

for ind_WS in df_settings.index:
	if not ind_WS in df_results.index:
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
		# Find relevant end date for specific simulation 
		# 
		##################

		df_T = pd.read_csv(folder_WS+'/T_all.csv',skiprows=range(8))
		df_T['# timestamp'] = df_T['# timestamp'].map(lambda x: str(x)[:-4])
		df_T = df_T.iloc[:-1]
		df_T['# timestamp'] = pd.to_datetime(df_T['# timestamp'])
		df_T.set_index('# timestamp',inplace=True)
		df_T = df_T.loc[start:end]
		#import pdb; pdb.set_trace()
		end = df_T.index[-1]

		##################
		#
		# Calculate retail rate in fixed price scenario / no TS
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
			df_inv_load = df_inv_load.loc[start:end]
			PV_supply = (df_inv_load.sum(axis=1)/1000./12.).sum() #in kWh
		except:
			PV_supply = 0.0

		print('RR with net-metering (w/o losses')

		net_demand  = total_load - PV_supply
		retail_kWh = supply_cost_wlosses/net_demand
		retail_kWh_wolosses = supply_cost_wolosses/net_demand

		retail_MWh = retail_kWh*1000.
		df_results.at[ind_WS,'RR'] = retail_MWh
		retail_MWh_wolosses = retail_kWh_wolosses*1000.
		df_results.at[ind_WS,'RR_wolosses'] = retail_MWh_wolosses
		df_results.at[ind_WS,'max_p'] = df_WS['RT'].max()
		df_results.at[ind_WS,'var_p'] = df_WS['RT'].var()
		df_results.at[ind_WS,'weighted_var_p'] = (df_WS['system_load']/df_WS['system_load'].sum()*(df_WS['RT'] - df_WS['RT'].mean()).pow(2)).sum()

		##################
		#
		# Evaluate utility
		#
		##################

		#Data
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

		df_hvac_load = pd.read_csv(folder_WS+'/hvac_load_all.csv',skiprows=range(8))
		df_hvac_load['# timestamp'] = df_hvac_load['# timestamp'].map(lambda x: str(x)[:-4])
		df_hvac_load = df_hvac_load.iloc[:-1]
		df_hvac_load['# timestamp'] = pd.to_datetime(df_hvac_load['# timestamp'])
		df_hvac_load.set_index('# timestamp',inplace=True) 
		df_hvac_load = df_hvac_load.loc[start:end]

		df_prices = pd.read_csv(folder_WS+'/df_prices.csv',index_col=[0],parse_dates=True)
		df_prices_1min = df_prices.copy()
		df_prices_1min = df_prices_1min.loc[start:end]

		#Benchmark
		if recalculate_df_welfare:
			df_u = df_T.copy()
			df_u_b = df_T_b.copy()
			#df_welfare = pd.DataFrame(index=df_u.columns,columns=['fixed_u','fixed_cost','fixed_T_mean','fixed_T_var','fixed_av_retail','LEM_u','LEM_cost','LEM_T_mean','LEM_T_var','LEM_av_retail'])
			df_welfare = pd.DataFrame(index=df_u.columns,columns=['fixed_u','fixed_cost','fixed_T_mean','fixed_T_min','fixed_T_max','fixed_T_min5','fixed_T_max95','fixed_T_var','fixed_av_retail','LEM_u','LEM_cost','LEM_T_mean','LEM_T_min','LEM_T_max','LEM_T_min5','LEM_T_max95','LEM_T_var','LEM_av_retail'])
			for col in df_u.columns:
				#print(col)
				#import pdb; pdb.set_trace()

				alpha = df_HVAC['alpha'].loc[col]
				T_com = df_HVAC['comf_temperature'].loc[col]
				#import pdb; pdb.set_trace()
				df_u[col] = (df_u[col] - T_com)
				df_u[col] = -alpha*df_u[col].pow(2)
				sum_u = (df_u[col] - df_hvac_load[col]/12.*df_prices_1min['clearing_price']/1000.).sum() # This includes the loss component
				df_welfare['LEM_u'].loc[col] = df_u[col].sum()
				df_welfare['LEM_cost'].loc[col] = (df_hvac_load[col]/12.*df_prices_1min['clearing_price']/1000.).sum()
				df_welfare['LEM_T_mean'].loc[col] = df_T[col].mean()
				df_welfare['LEM_T_var'].loc[col] = df_T[col].var()
				df_welfare['LEM_av_retail'].loc[col] = 1000*(df_hvac_load[col]/12.*df_prices_1min['clearing_price']/1000.).sum()/(df_hvac_load[col].sum()/12.)

				df_u_b[col] = (df_u_b[col] - T_com)
				df_u_b[col] = -alpha*df_u_b[col].pow(2)
				sum_u_b = (df_u_b[col] - df_hvac_load_b[col]/12.*retail_MWh/1000.).sum()
				df_welfare['fixed_u'].loc[col] = df_u_b[col].sum()
				df_welfare['fixed_cost'].loc[col] = (df_hvac_load_b[col]/12.*retail_MWh/1000.).sum()
				df_welfare['fixed_T_mean'].loc[col] = df_T_b[col].mean()
				df_welfare['fixed_T_var'].loc[col] = df_T_b[col].var()
				df_welfare['fixed_av_retail'].loc[col] = retail_MWh

				# Added for temperature variance analysis

				df_welfare['LEM_T_min'].loc[col] = df_T[col].min()
				df_welfare['LEM_T_max'].loc[col] = df_T[col].max()
				df_welfare['LEM_T_min5'].loc[col] = df_T[col].quantile(q=0.05)
				df_welfare['LEM_T_max95'].loc[col] = df_T[col].quantile(q=0.95)

				df_welfare['fixed_T_min'].loc[col] = df_T_b[col].min()
				df_welfare['fixed_T_max'].loc[col] = df_T_b[col].max()
				df_welfare['fixed_T_min5'].loc[col] = df_T_b[col].quantile(q=0.05)
				df_welfare['fixed_T_max95'].loc[col] = df_T_b[col].quantile(q=0.95)
				
				#
				
				df_welfare.to_csv(folder_WS + '/df_welfare_i.csv')
				df = df_HVAC.join(df_welfare)
				df.to_csv(folder_WS + '/df_welfare_withparameters_i.csv')
				#import pdb; pdb.set_trace()

		df_welfare = pd.read_csv(folder_WS + '/df_welfare_withparameters_i.csv',index_col=[0],parse_dates=True)
		df_welfare['u_change'] = (df_welfare['LEM_u'] - df_welfare['LEM_cost']) - (df_welfare['fixed_u'] - df_welfare['fixed_cost'])

		df_results.at[ind_WS,'u_change'] = df_welfare['u_change'].sum()
		df_results.at[ind_WS,'av_uchange'] = df_welfare['u_change'].mean()

df_results.to_csv(run + '/' + 'weekly_welfare_changes.csv')
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