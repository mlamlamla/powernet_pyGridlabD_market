#This file compares the WS market participation with stationary price bids to fixed price scenarie

import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt

max_y = 145

# Energy procurement cost scenario
run = 'Diss'
ind_WS = 124
ind_b = 90

df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])
#df_HVAC = pd.read_csv(run+'/HVAC_settings_'+str(start).split(' ')[0]+'_'+str(end).split(' ')[0]+'_OLS4.csv',index_col=[0])
#df_HVAC = pd.read_csv(run+'/HVAC_settings_inclmidnight/'+df_settings['settings_file'].loc[ind_WS],index_col=[0])
df_HVAC = pd.read_csv(run+'/HVAC_settings/'+df_settings['settings_file'].loc[ind_WS],index_col=[0])
#df_HVAC = pd.read_csv('Diss/old/HVAC_settings_2016-08-01_2016-08-08_ext.csv',index_col=[0])
start = pd.to_datetime(df_settings['start_time'].loc[ind_WS]) + pd.Timedelta(days=1)
end = pd.to_datetime(df_settings['end_time'].loc[ind_WS])

recread_data = True 
recalculate_df_welfare = True

house_no = 0

folder_b = run + '/Diss_'+"{:04d}".format(ind_b) #+'_5min'
folder_WS = run + '/Diss_'+"{:04d}".format(ind_WS)


# retail_kWh = 0.02391749988554048 #USD/kWh
# retail_kWh = 0.03245935410676796 # (july) # 0.02391749988554048 (year) #USD/kWh
# retail_kWh = 0.026003645505416464 #first week July
# retail_kWh = 0.0704575196993475 #first week of august
#retail_kWh = 0.02254690804746962 #mid-Dec

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
print(end)

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

# df_WS['system_load'] = df_slack['measured_real_power']
# df_WS['supply_cost'] = df_WS['system_load']/1000.*df_WS['RT']/12.
# supply_cost = df_WS['supply_cost'].sum()

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
print(retail_MWh)
retail_MWh_wolosses = retail_kWh_wolosses*1000.
print(retail_MWh_wolosses)
#import pdb; pdb.set_trace()

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
		
		df_welfare.to_csv(folder_WS + '/df_welfare.csv')
		df = df_HVAC.join(df_welfare)
		df.to_csv(folder_WS + '/df_welfare_withparameters.csv')
		#import pdb; pdb.set_trace()

df_welfare = pd.read_csv(folder_WS + '/df_welfare_withparameters.csv',index_col=[0],parse_dates=True)
df_welfare['u_change'] = (df_welfare['LEM_u'] - df_welfare['LEM_cost']) - (df_welfare['fixed_u'] - df_welfare['fixed_cost'])
#import pdb; pdb.set_trace()
print('Average utility change')
print(df_welfare['u_change'].mean())
print('Total utility change')
print(df_welfare['u_change'].sum())


#Histogram utility change
fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(df_welfare['beta'],df_welfare['u_change'])
#ax.set_ylim(0,75)
# if df_welfare['u_change'].min() > 0.0:
# 	ax.set_xlim(0,df_welfare['u_change'].max()*1.05)
# else:
# 	ax.vlines(0,0,max_y,'k',lw=1)
# ax.set_xlabel('Utility change [USD]')
# if max_y > 0.0:
# 	ax.set_ylim(0,max_y)
# ax.set_ylabel('Number of houses')
ppt.savefig(folder_WS+'/beta_uchange_'+str(ind_WS)+'.png', bbox_inches='tight')
ppt.savefig(folder_WS+'/beta_uchange_'+str(ind_WS)+'.pdf', bbox_inches='tight')
import pdb; pdb.set_trace()
