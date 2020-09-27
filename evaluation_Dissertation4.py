#This file compares the WS market participation with stationary price bids to fixed price scenarie

#Unconstrained system for utlity: energy procurement cost

import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt
import matplotlib.dates as mdates

ind_b = 46 #year-long simulation without LEM or constraint
df_summary_all = pd.DataFrame()

for ind_unconstrained in [63,64]:

	#Get start and end date
	if ind_unconstrained in [47,63]:
		#ind_unconstrained = 47 #update with +1d 63
		start = pd.Timestamp(2016,8,1)
		end = pd.Timestamp(2016,8,8)
		retail_kWh = 0.0704575196993475 #first week of august
		df_HVAC = pd.read_csv('Diss/HVAC_settings_2016-08-01_2016-08-08_ext.csv',index_col=[0])
	elif ind_unconstrained in [49,50,51,52,53,54,55,56,57,58,59,60,61,62,64]:
		#ind_unconstrained = 49 #update with +1d 64
		start = pd.Timestamp(2016,12,19)
		end = pd.Timestamp(2016,12,26)
		retail_kWh = 0.02254690804746962 #mid-Dec
		df_HVAC = pd.read_csv('Diss/HVAC_settings_2016-12-19_2016-12-26_ext.csv',index_col=[0])

	retail_MWh = retail_kWh*1000.

	#Data
	df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])

	#Benchmark / no LEM
	folder_b = 'Diss/Diss_00'+str(ind_b) #+'_5min' #No LEM
	df_slack_b = pd.read_csv(folder_b+'/load_node_149.csv',skiprows=range(8))
	df_slack_b['# timestamp'] = df_slack_b['# timestamp'].map(lambda x: str(x)[:-4])
	df_slack_b = df_slack_b.iloc[:-1]
	df_slack_b['# timestamp'] = pd.to_datetime(df_slack_b['# timestamp'])
	df_slack_b.set_index('# timestamp',inplace=True)
	df_slack_b = df_slack_b.loc[start:end]

	df_total_load_b = pd.read_csv(folder_b+'/total_load_all.csv',skiprows=range(8))
	df_total_load_b['# timestamp'] = df_total_load_b['# timestamp'].map(lambda x: str(x)[:-4])
	df_total_load_b['# timestamp'] = pd.to_datetime(df_total_load_b['# timestamp'])
	df_total_load_b.set_index('# timestamp',inplace=True)
	df_total_load_b = df_total_load_b.loc[start:end]
	df_total_load_b_agg = pd.DataFrame(index=df_total_load_b.index,columns=['total_load_houses'],data=df_total_load_b.sum(axis=1))

	df_PV_b = pd.read_csv(folder_b+'/total_P_Out.csv',skiprows=range(8))
	df_PV_b['# timestamp'] = df_PV_b['# timestamp'].map(lambda x: str(x)[:-4])
	df_PV_b['# timestamp'] = pd.to_datetime(df_PV_b['# timestamp'])
	df_PV_b.set_index('# timestamp',inplace=True)           
	df_PV_b = df_PV_b.loc[start:end]

	#LEM
	folder = 'Diss/Diss_00'+str(ind_unconstrained) #Prices from unconstrained LEM (== WS price)
	df_slack_LEM = pd.read_csv(folder+'/load_node_149.csv',skiprows=range(8))
	df_slack_LEM['# timestamp'] = df_slack_LEM['# timestamp'].map(lambda x: str(x)[:-4])
	df_slack_LEM = df_slack_LEM.iloc[:-1]
	df_slack_LEM['# timestamp'] = pd.to_datetime(df_slack_LEM['# timestamp'])
	df_slack_LEM.set_index('# timestamp',inplace=True)
	df_slack_LEM = df_slack_LEM.loc[start:end]

	df_total_load_LEM = pd.read_csv(folder+'/total_load_all.csv',skiprows=range(8))
	df_total_load_LEM['# timestamp'] = df_total_load_LEM['# timestamp'].map(lambda x: str(x)[:-4])
	df_total_load_LEM['# timestamp'] = pd.to_datetime(df_total_load_LEM['# timestamp'])
	df_total_load_LEM.set_index('# timestamp',inplace=True)
	df_total_load_LEM = df_total_load_LEM.loc[start:end]
	df_total_load_LEM_agg = pd.DataFrame(index=df_total_load_LEM.index,columns=['total_load_houses'],data=df_total_load_LEM.sum(axis=1))
	df_total_load_LEM = df_total_load_LEM.loc[start:end]

	df_PV_LEM = pd.read_csv(folder+'/total_P_Out.csv',skiprows=range(8))
	df_PV_LEM['# timestamp'] = df_PV_LEM['# timestamp'].map(lambda x: str(x)[:-4])
	df_PV_LEM['# timestamp'] = pd.to_datetime(df_PV_LEM['# timestamp'])
	df_PV_LEM.set_index('# timestamp',inplace=True)
	df_PV_LEM = df_PV_LEM.loc[start:end]

	df_hvac_load_LEM = pd.read_csv(folder+'/hvac_load_all.csv',skiprows=range(8))
	df_hvac_load_LEM['# timestamp'] = df_hvac_load_LEM['# timestamp'].map(lambda x: str(x)[:-4])
	df_hvac_load_LEM['# timestamp'] = pd.to_datetime(df_hvac_load_LEM['# timestamp'])
	df_hvac_load_LEM.set_index('# timestamp',inplace=True)
	df_hvac_load_LEM = df_hvac_load_LEM.loc[start:end] 
	df_hvac_load_LEM[df_hvac_load_LEM < 0.5] = 0. #Get out gas-fired heating

	df_inflex_load_LEM = df_total_load_LEM - df_hvac_load_LEM

	df_prices = pd.read_csv(folder+'/df_prices.csv',index_col=[0],parse_dates=True)
	df_prices = df_prices.loc[start:end]

	#Summary
	df_summary = pd.DataFrame(columns=['No LEM','LEM'])

	#Average procurement price
	peak_b = (df_slack_b['measured_real_power']/1000000).max()
	peak_LEM = (df_slack_LEM['measured_real_power']/1000000).max()

	MWh_b = (df_slack_b['measured_real_power']/12./1000000).sum()
	MWh_LEM = (df_slack_LEM['measured_real_power']/12./1000000).sum()

	consumption_total_b = df_total_load_b.sum().sum()/12./1000 #in MWh
	consumption_flexhvac_b = 0.0
	share_flex_b = 0.0

	consumption_total_LEM = df_total_load_LEM.sum().sum()/12./1000 #in MWh
	consumption_flexhvac_LEM = df_hvac_load_LEM.sum().sum()/12./1000.
	share_flex_LEM = consumption_flexhvac_LEM/consumption_total_LEM*100.

	PV_gen_b = df_PV_b.sum().sum()/12./1000000  #in MWh
	PV_gen_LEM = df_PV_LEM.sum().sum()/12./1000000  #in MWh

	proc_cost_b = (df_slack_b['measured_real_power']/12./1000000*df_prices['clearing_price']).sum()
	MWh_price_b = proc_cost_b/MWh_b

	proc_cost_LEM = (df_slack_LEM['measured_real_power']/12./1000000*df_prices['clearing_price']).sum()
	MWh_price_LEM = proc_cost_LEM/MWh_LEM

	proc_cost_inflex_b = ((df_total_load_b.sum(axis=1) - df_PV_b.sum(axis=1)/1000.)/12./1000*df_prices['clearing_price']).sum()
	proc_losses_b = proc_cost_b - proc_cost_inflex_b
	grid_tariff_b = proc_losses_b/((df_total_load_b.sum().sum() - df_PV_b.sum().sum()/1000.)/12.) #per kWh

	proc_cost_inflex_LEM = ((df_inflex_load_LEM.sum(axis=1) - df_PV_LEM.sum(axis=1)/1000.)/12./1000*df_prices['clearing_price']).sum()
	proc_cost_flex_LEM = (df_hvac_load_LEM.sum(axis=1)/12./1000*df_prices['clearing_price']).sum()
	proc_losses_LEM = proc_cost_LEM - proc_cost_inflex_LEM - proc_cost_flex_LEM
	grid_tariff_LEM = proc_losses_LEM/((df_total_load_LEM.sum().sum() - df_PV_LEM.sum().sum()/1000.)/12.) #per kWh
	share_flex_cost_LEM = proc_cost_flex_LEM/proc_cost_LEM*100

	retail_rate_b = proc_cost_b/(consumption_total_b - PV_gen_b)/1000.
	retail_rate_LEM = proc_cost_inflex_LEM/(df_inflex_load_LEM.sum().sum()/12. - df_PV_LEM.sum().sum()/12./1000.) #kWh

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