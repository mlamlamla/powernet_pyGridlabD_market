#This file compares the WS market participation with stationary price bids to fixed price scenarie

#Unconstrained system for utlity: energy procurement cost

import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt
import matplotlib.dates as mdates

run = 'Diss'
ind_b = 90 #year-long simulation without LEM or constraint
ind_WS = 124
df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])
city = df_settings['city'].loc[ind_WS]

df_summary_all = pd.DataFrame()

# For RR calculation

folder_b = 'Diss/Diss_00'+str(ind_b) #+'_5min' #No LEM
df_slack = pd.read_csv(folder_b+'/load_node_149.csv',skiprows=range(8))
df_slack['# timestamp'] = df_slack['# timestamp'].map(lambda x: str(x)[:-4])
df_slack = df_slack.iloc[:-1]
df_slack['# timestamp'] = pd.to_datetime(df_slack['# timestamp'])
df_slack.set_index('# timestamp',inplace=True)
df_slack = df_slack/1000. # in kWh

df_total_load = pd.read_csv(folder_b+'/total_load_all.csv',skiprows=range(8))
df_total_load['# timestamp'] = df_total_load['# timestamp'].map(lambda x: str(x)[:-4])
df_total_load['# timestamp'] = pd.to_datetime(df_total_load['# timestamp'])
df_total_load.set_index('# timestamp',inplace=True)  # in kWh

market_file = df_settings['market_data'].loc[ind_WS]
market_file = 'Ercot_LZ_SOUTH.csv'
df_WS_year = pd.read_csv('glm_generation_'+city+'/'+market_file,parse_dates=[0])
df_WS_year.rename(columns={'Unnamed: 0':'timestamp'},inplace=True)
df_WS_year.set_index('timestamp',inplace=True) # in USD/MWh

for ind_unconstrained in [ind_WS]:

	start = pd.to_datetime(df_settings['start_time'].loc[ind_WS]) + pd.Timedelta(days=1)
	end = pd.to_datetime(df_settings['end_time'].loc[ind_WS])
	df_HVAC = pd.read_csv(run+'/HVAC_settings/'+df_settings['settings_file'].loc[ind_WS],index_col=[0])

	# Benchmark / no LEM for Retail Rate calculation
	df_slack_b = df_slack.loc[start:end]
	df_total_load_b = df_total_load.loc[start:end]
	df_total_load_b_agg = pd.DataFrame(index=df_total_load_b.index,columns=['total_load_houses'],data=df_total_load_b.sum(axis=1))
	total_load = (df_total_load_b.sum(axis=1)/12.).sum() #kWh

	df_WS = df_WS_year.loc[start:end]

	df_WS['system_load'] = df_slack_b['measured_real_power']/1000.
	supply_wlosses = (df_WS['system_load']/12.).sum() # MWh
	df_WS['supply_cost'] = df_WS['system_load']/12.*df_WS['RT']
	supply_cost_wlosses = df_WS['supply_cost'].sum()

	df_WS['res_load'] = df_total_load_b.sum(axis=1)
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
		df_PV_b = pd.DataFrame(index=df_total_load_b.index,columns=df_total_load_b.columns,data=0.0)
		PV_supply = 0.0

	print('Average procurement cost')

	net_demand  = total_load - PV_supply # kWh
	retail_kWh = supply_cost_wlosses/net_demand
	retail_MWh = retail_kWh*1000.
	print(retail_MWh)

	#LEM
	folder = 'Diss/Diss_'+"{:04d}".format(ind_unconstrained) #Prices from unconstrained LEM (== WS price)
	df_slack_LEM = pd.read_csv(folder+'/load_node_149.csv',skiprows=range(8))
	df_slack_LEM['# timestamp'] = df_slack_LEM['# timestamp'].map(lambda x: str(x)[:-4])
	df_slack_LEM = df_slack_LEM.iloc[:-1]
	df_slack_LEM['# timestamp'] = pd.to_datetime(df_slack_LEM['# timestamp'])
	df_slack_LEM.set_index('# timestamp',inplace=True)
	df_slack_LEM = df_slack_LEM.loc[start:end]
	df_slack_LEM = df_slack_LEM/1000. # in kWh

	df_total_load_LEM = pd.read_csv(folder+'/total_load_all.csv',skiprows=range(8))
	df_total_load_LEM['# timestamp'] = df_total_load_LEM['# timestamp'].map(lambda x: str(x)[:-4])
	df_total_load_LEM['# timestamp'] = pd.to_datetime(df_total_load_LEM['# timestamp'])
	df_total_load_LEM.set_index('# timestamp',inplace=True)
	df_total_load_LEM = df_total_load_LEM.loc[start:end]
	df_total_load_LEM_agg = pd.DataFrame(index=df_total_load_LEM.index,columns=['total_load_houses'],data=df_total_load_LEM.sum(axis=1))
	df_total_load_LEM = df_total_load_LEM.loc[start:end] # in kWh

	try:
		df_PV_LEM = pd.read_csv(folder+'/total_P_Out.csv',skiprows=range(8))
		df_PV_LEM['# timestamp'] = df_PV_LEM['# timestamp'].map(lambda x: str(x)[:-4])
		df_PV_LEM['# timestamp'] = pd.to_datetime(df_PV_LEM['# timestamp'])
		df_PV_LEM.set_index('# timestamp',inplace=True)
		df_PV_LEM = df_PV_LEM.loc[start:end]
		PV_supply = (df_PV_LEM.sum(axis=1)/1000./12.).sum() #in kWh
	except:
		df_PV_LEM = pd.DataFrame(index=df_total_load_b.index,columns=df_total_load_b.columns,data=0.0)
		PV_supply = 0.0

	df_hvac_load_LEM = pd.read_csv(folder+'/hvac_load_all.csv',skiprows=range(8))
	df_hvac_load_LEM['# timestamp'] = df_hvac_load_LEM['# timestamp'].map(lambda x: str(x)[:-4])
	df_hvac_load_LEM['# timestamp'] = pd.to_datetime(df_hvac_load_LEM['# timestamp'])
	df_hvac_load_LEM.set_index('# timestamp',inplace=True)
	df_hvac_load_LEM = df_hvac_load_LEM.loc[start:end] # in kWh
	df_hvac_load_LEM[df_hvac_load_LEM < 0.5] = 0. #Get out gas-fired heating

	df_inflex_load_LEM = df_total_load_LEM - df_hvac_load_LEM

	df_prices = pd.read_csv(folder+'/df_prices.csv',index_col=[0],parse_dates=True)
	df_prices = df_prices.loc[start:end]

	#Summary
	df_summary = pd.DataFrame(columns=['No LEM','LEM'])

	#Average procurement price
	#import pdb; pdb.set_trace()
	peak_b = (df_slack_b['measured_real_power']/1000).max()
	peak_LEM = (df_slack_LEM['measured_real_power']/1000).max()

	MWh_b = (df_slack_b['measured_real_power']/12./1000.).sum()
	MWh_LEM = (df_slack_LEM['measured_real_power']/12./1000.).sum()

	consumption_total_b = df_total_load_b.sum().sum()/12./1000. #in MWh
	consumption_flexhvac_b = 0.0
	share_flex_b = 0.0

	consumption_total_LEM = df_total_load_LEM.sum().sum()/12./1000. #in MWh
	consumption_flexhvac_LEM = df_hvac_load_LEM.sum().sum()/12./1000.
	share_flex_LEM = consumption_flexhvac_LEM/consumption_total_LEM*100.

	PV_gen_b = df_PV_b.sum().sum()/12./1000.  #in MWh
	PV_gen_LEM = df_PV_LEM.sum().sum()/12./1000.  #in MWh

	#proc_cost_b = (df_slack_b['measured_real_power']/12./1000.*df_prices['clearing_price']).sum()
	proc_cost_b = (df_slack_b['measured_real_power']/12./1000.*df_WS['RT']).sum()
	MWh_price_b = proc_cost_b/MWh_b # retail rate

	#proc_cost_LEM = (df_slack_LEM['measured_real_power']/12./1000000*df_prices['clearing_price']).sum()
	proc_cost_LEM = (df_slack_LEM['measured_real_power']/12./1000.*df_WS['RT']).sum()
	MWh_price_LEM = proc_cost_LEM/MWh_LEM

	#proc_cost_inflex_b = ((df_total_load_b.sum(axis=1) - df_PV_b.sum(axis=1)/1000.)/12./1000*df_prices['clearing_price']).sum()
	proc_cost_load_b = ((df_total_load_b.sum(axis=1) - df_PV_b.sum(axis=1)/1000.)/12./1000*df_WS['RT']).sum()
	proc_cost_losses_b = proc_cost_b - proc_cost_load_b
	grid_tariff_b = proc_cost_losses_b/((df_total_load_b.sum().sum() - df_PV_b.sum().sum()/1000.)/12.) #per kWh

	#proc_cost_inflex_LEM = ((df_inflex_load_LEM.sum(axis=1) - df_PV_LEM.sum(axis=1)/1000.)/12./1000*df_prices['clearing_price']).sum()
	proc_cost_load_LEM = ((df_total_load_LEM.sum(axis=1) - df_PV_LEM.sum(axis=1)/1000.)/12./1000*df_WS['RT']).sum()
	proc_cost_flex_LEM = (df_hvac_load_LEM.sum(axis=1)/12./1000*df_WS['RT']).sum()
	proc_cost_losses_LEM = proc_cost_LEM - proc_cost_load_LEM #- proc_cost_flex_LEM
	grid_tariff_LEM = proc_cost_losses_LEM/((df_total_load_LEM.sum().sum() - df_PV_LEM.sum().sum()/1000.)/12.) #per kWh
	share_flex_cost_LEM = proc_cost_flex_LEM/proc_cost_load_LEM*100

	retail_rate_b = proc_cost_b/(consumption_total_b - PV_gen_b)/1000.

	LEM_income = (df_hvac_load_LEM.sum(axis=1)/12./1000*df_prices['clearing_price']).sum()
	cost_inflex = proc_cost_b - LEM_income
	#retail_rate_LEM = proc_cost_inflex_LEM/(df_inflex_load_LEM.sum().sum()/12. - df_PV_LEM.sum().sum()/12./1000.) #kWh
	load_inflex = df_total_load_LEM.sum().sum()/12. - df_hvac_load_LEM.sum().sum()/12. # in kWh
	retail_rate_LEM = cost_inflex / load_inflex

	df_summary = df_summary.append(pd.DataFrame(index=['Total peak load [MW]'],columns=['No LEM','LEM'],data=[[peak_b,peak_LEM]]))
	df_summary = df_summary.append(pd.DataFrame(index=['Energy procured [MWh]'],columns=['No LEM','LEM'],data=[[MWh_b,MWh_LEM]]))
	df_summary = df_summary.append(pd.DataFrame(index=['PV generation [MWh]'],columns=['No LEM','LEM'],data=[[PV_gen_b,PV_gen_LEM]]))
	df_summary = df_summary.append(pd.DataFrame(index=['Share of flexible load [\%]'],columns=['No LEM','LEM'],data=[[share_flex_b,share_flex_LEM]]))
	df_summary = df_summary.append(pd.DataFrame(index=['Procurement cost [USD]'],columns=['No LEM','LEM'],data=[[proc_cost_b,proc_cost_LEM]]))
	df_summary = df_summary.append(pd.DataFrame(index=['Share of flexible procurement cost [\%]'],columns=['No LEM','LEM'],data=[[0.0,share_flex_cost_LEM]]))
	df_summary = df_summary.append(pd.DataFrame(index=['Average procurement price [USD/MWh]'],columns=['No LEM','LEM'],data=[[MWh_price_b,MWh_price_LEM]]))
	df_summary = df_summary.append(pd.DataFrame(index=['Fixed retail rate [USD/kWh]'],columns=['No LEM','LEM'],data=[[retail_rate_b,retail_rate_LEM]]))
	df_summary = df_summary.append(pd.DataFrame(index=['Grid tariff losses [USD/kWh]'],columns=['No LEM','LEM'],data=[[grid_tariff_b,grid_tariff_LEM]]))
	
	if len(df_summary_all) == 0:
		df_summary_all = df_summary
	else:
		df_summary_all = df_summary_all.merge(df_summary,left_index=True,right_index=True)
	
df_summary_round = df_summary_all.round(3)
print(df_summary_round)
table = open('Diss/df_summary_utility.tex','w')
table.write(df_summary_round.to_latex(index=True,escape=False,na_rep=''))
table.close()

import pdb; pdb.set_trace()