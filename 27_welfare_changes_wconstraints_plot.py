import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt
from sklearn.linear_model import LinearRegression

# Default
run = 'Diss'
ind_b = 90
folder_b = run + '/Diss_'+"{:04d}".format(ind_b) #+'_5min'

# All relevant market runs for 2016

df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])
inds = [157,159,129,160,161,162,163,164,165,166,167,168,156,169,170,171]
inds += [125,124,126,128,127]
inds += [*range(172,202)]

df_settings = df_settings.loc[inds]

for ind in inds:

	start = pd.to_datetime(df_settings['start_time'].loc[ind]) + pd.Timedelta(days=1)
	end = pd.to_datetime(df_settings['end_time'].loc[ind])

	# load df_slack

	folder_WS = run + '/Diss_'+"{:04d}".format(ind)
	df_slack = pd.read_csv(folder_WS+'/load_node_149.csv',skiprows=range(8))
	df_slack['# timestamp'] = df_slack['# timestamp'].map(lambda x: str(x)[:-4])
	df_slack = df_slack.iloc[:-1]
	df_slack['# timestamp'] = pd.to_datetime(df_slack['# timestamp'])
	df_slack.set_index('# timestamp',inplace=True)
	df_slack = df_slack.loc[start:end]
	df_slack = df_slack/1000 #kW

	# max

	print(str(start) + ', max load: ' + str(df_slack['measured_real_power'].max()))
	print(str(start) + ', energy consumed: ' + str(df_slack['measured_real_power'].sum()/12./1000.))

import pdb; pdb.set_trace()

# Relevant runs for constraint evaluation

df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])
inds1 = [127,130,131,132,133,134,135,136,137]
inds2 = [128,138,139,140,141,142,143,144,145]
inds3 = [129,146,147,148,149,150,151,152,153]

for cong_costs in [40.0,50.0,60.0]:
	results = []
	for inds in [inds1,inds2,inds3]:

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

		df_results.set_index('C',inplace=True)
		results += [df_results]

	#import pdb; pdb.set_trace()
	for df in results[:-1]:
		df_results += df

	print(cong_costs)
	print(df_results['net_welfare_change_fixed'])

