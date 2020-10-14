#This file compares if positive investment incentives exist for pv if under fixed RR or in LEM

import pandas as pd
import matplotlib.pyplot as ppt

run = 'Diss'

# LEM

ind_WS_noPV = 88 # or 89?
ind_WS_PV = 81

# No LEM

ind_b_noPV = 90
ind_b_PV = 46

# Basecase
ind_b = ind_b_noPV
ind_WS = ind_b_PV

# Input data

df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])
df_HVAC = pd.read_csv(run+'/HVAC_settings/'+df_settings['settings_file'].loc[ind_WS],index_col=[0])
start = pd.to_datetime(df_settings['start_time'].loc[ind_WS]) + pd.Timedelta(days=1)
end = pd.to_datetime(df_settings['end_time'].loc[ind_WS])

recread_data = True 
recalculate_df_welfare = True

folder_b = run + '/Diss_00'+str(ind_b) #+'_5min'
folder_WS = run + '/Diss_00'+str(ind_WS)

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
df_WS['supply_cost'] = df_WS['system_load']/1000.*df_WS['RT']/12.
supply_cost = df_WS['supply_cost'].sum()

df_total_load = pd.read_csv(folder+'/total_load_all.csv',skiprows=range(8)) #in kW
df_total_load['# timestamp'] = df_total_load['# timestamp'].map(lambda x: str(x)[:-4])
df_total_load = df_total_load.iloc[:-1]
df_total_load['# timestamp'] = pd.to_datetime(df_total_load['# timestamp'])
df_total_load.set_index('# timestamp',inplace=True)
df_total_load = df_total_load.loc[start:end]
total_load = (df_total_load.sum(axis=1)/12.).sum() #kWh

df_inv_load_b = pd.read_csv(folder+'/total_P_Out.csv',skiprows=range(8)) #in W
df_inv_load_b['# timestamp'] = df_inv_load_b['# timestamp'].map(lambda x: str(x)[:-4])
df_inv_load_b = df_inv_load_b.iloc[:-1]
df_inv_load_b['# timestamp'] = pd.to_datetime(df_inv_load_b['# timestamp'])
df_inv_load_b.set_index('# timestamp',inplace=True)  
df_inv_load_b = df_inv_load_b.loc[start:end]
PV_supply = (df_inv_load_b.sum(axis=1)/1000./12.).sum() #in kWh

print('Base case RR with net-metering')
net_demand  = total_load - PV_supply
retail_kWh = supply_cost/net_demand
retail_MWh = retail_kWh*1000.
print(retail_kWh)

##################
#
# Evaluate utility in basecase
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

# df_inv_load_b has already been read in when calculating RR

# df_T has been read in at the beginning of this file already

df_hvac_load = pd.read_csv(folder_WS+'/hvac_load_all.csv',skiprows=range(8))
df_hvac_load['# timestamp'] = df_hvac_load['# timestamp'].map(lambda x: str(x)[:-4])
df_hvac_load = df_hvac_load.iloc[:-1]
df_hvac_load['# timestamp'] = pd.to_datetime(df_hvac_load['# timestamp'])
df_hvac_load.set_index('# timestamp',inplace=True) 
df_hvac_load = df_hvac_load.loc[start:end]

df_inv_load_WS = pd.read_csv(folder_WS+'/total_P_Out.csv',skiprows=range(8)) #in W
df_inv_load_WS['# timestamp'] = df_inv_load_WS['# timestamp'].map(lambda x: str(x)[:-4])
df_inv_load_WS = df_inv_load_WS.iloc[:-1]
df_inv_load_WS['# timestamp'] = pd.to_datetime(df_inv_load_WS['# timestamp'])
df_inv_load_WS.set_index('# timestamp',inplace=True)  
df_inv_load_WS = df_inv_load_WS.loc[start:end]

df_prices = pd.read_csv(folder_WS+'/df_prices.csv',index_col=[0],parse_dates=True)
df_prices_1min = df_prices.copy()
df_prices_1min = df_prices_1min.loc[start:end]

#Benchmark
has_PV = False
if recalculate_df_welfare:
	df_u = df_T.copy()
	df_u_b = df_T_b.copy()
	df_welfare = pd.DataFrame(index=df_u.columns,columns=['fixed_u','fixed_cost','fixed_income','fixed_T_mean','fixed_T_var','fixed_av_retail','LEM_u','LEM_cost','LEM_income','LEM_T_mean','LEM_T_var','LEM_av_retail'])
	for col in df_u.columns:
		#print(col)
		#import pdb; pdb.set_trace()

		alpha = df_HVAC['alpha'].loc[col]
		T_com = df_HVAC['comf_temperature'].loc[col]
		#import pdb; pdb.set_trace()
		df_u[col] = (df_u[col] - T_com)
		df_u[col] = -alpha*df_u[col].pow(2)
		sum_u = (df_u[col] - df_hvac_load[col]/12.*df_prices_1min['clearing_price']/1000.).sum()
		df_welfare['LEM_u'].loc[col] = df_u[col].sum()
		df_welfare['LEM_cost'].loc[col] = (df_hvac_load[col]/12.*df_prices_1min['clearing_price']/1000.).sum()
		if 'PV_inverter'+col[3:] in df_inv_load_WS.columns:
			df_welfare['LEM_income'].loc[col] = (df_inv_load_WS['PV_inverter'+col[3:]]/12./1000.*df_prices_1min['clearing_price']/1000.).sum()
			has_PV = True
		else:
			df_welfare['LEM_income'].loc[col] = 0.0
			has_PV = False
		df_welfare['LEM_T_mean'].loc[col] = df_T[col].mean()
		df_welfare['LEM_T_var'].loc[col] = df_T[col].var()
		df_welfare['LEM_av_retail'].loc[col] = 1000*(df_hvac_load[col]/12.*df_prices_1min['clearing_price']/1000.).sum()/(df_hvac_load[col].sum()/12.)

		df_u_b[col] = (df_u_b[col] - T_com)
		df_u_b[col] = -alpha*df_u_b[col].pow(2)
		sum_u_b = (df_u_b[col] - df_hvac_load_b[col]/12.*retail_MWh/1000.).sum()
		df_welfare['fixed_u'].loc[col] = df_u_b[col].sum()
		df_welfare['fixed_cost'].loc[col] = (df_hvac_load_b[col]/12.*retail_MWh/1000.).sum()
		if has_PV:
			df_welfare['fixed_income'].loc[col] = (df_inv_load_b['PV_inverter'+col[3:]]/12./1000.*retail_kWh).sum()
			has_PV = False
		else:
			df_welfare['fixed_income'].loc[col] = 0.0
			has_PV = False
		df_welfare['fixed_T_mean'].loc[col] = df_T_b[col].mean()
		df_welfare['fixed_T_var'].loc[col] = df_T_b[col].var()
		df_welfare['fixed_av_retail'].loc[col] = retail_MWh
		
		df_welfare.to_csv(folder_WS + '/df_welfare.csv')
		df = df_HVAC.join(df_welfare)
		df.to_csv(folder_WS + '/df_welfare_withparameters.csv')

#import pdb; pdb.set_trace()
df_welfare = pd.read_csv(folder_WS + '/df_welfare_withparameters.csv',index_col=[0],parse_dates=True)
df_welfare['u_change'] = (df_welfare['LEM_u'] - df_welfare['LEM_cost'] + df_welfare['LEM_income']) - (df_welfare['fixed_u'] - df_welfare['fixed_cost'] + df_welfare['fixed_income'])

print('Average utility change')
print(df_welfare['u_change'].mean())
print('Total utility change')
print(df_welfare['u_change'].sum())

print('u change for households with PV')
print(df_welfare.loc[df_welfare['LEM_income'] > 0.0]['u_change'].mean())
print('u change for households without PV')
print(df_welfare.loc[df_welfare['LEM_income'] == 0.0]['u_change'].mean())
print('Correlation between price and PV generation')
print(df_inv_load_b.sum(axis=1).corr(df_prices_1min['clearing_price']))

# #Temperature of a single house vs. price: Dow does temperature depend on price?
# fig = ppt.figure(figsize=(8,4),dpi=150)   
# ppt.ioff()
# ax = fig.add_subplot(111)
# lns = []
# lns += ppt.plot(df_T[df_T.columns[house_no]].loc[start:end],label='House '+str(house_no))
# ax.set_xlabel('Time')
# ax.set_ylabel('Temperature')
# ax2 = ax.twinx()
# lns += ax2.plot(df_prices['clearing_price'].loc[start:end],'r',label='WS price')
# labs = [l.get_label() for l in lns]
# L = ax.legend(lns, labs, loc='lower left', ncol=1)
# ppt.savefig(folder_WS+'/temperature_vs_price_byhouse.png', bbox_inches='tight')

# #HVAC load vs. price: Does aggregate HVAC load get reduced if price increases?
# fig = ppt.figure(figsize=(8,4),dpi=150)   
# ppt.ioff()
# ax = fig.add_subplot(111)
# lns = []
# lns += ppt.plot(df_hvac_load.sum(axis=1).loc[start:end],label='Total HVAC load')
# ax.set_xlabel('Time')
# ax.set_ylabel('HVAC load')
# ax2 = ax.twinx()
# lns += ax2.plot(df_prices['clearing_price'].loc[start:end],'r',label='WS price')
# labs = [l.get_label() for l in lns]
# L = ax.legend(lns, labs, loc='lower left', ncol=1)
# ppt.savefig(folder_WS+'/hvacload_vs_price_byhouse.png', bbox_inches='tight')

# import pdb; pdb.set_trace()

# #Histogram utility change
# fig = ppt.figure(figsize=(8,4),dpi=150)   
# ppt.ioff()
# ax = fig.add_subplot(111)
# lns = ppt.hist(df_welfare['u_change'],bins=20,color='0.75',edgecolor='0.5')
# ax.vlines(0,0,50)
# ax.set_ylim(0,50)
# ax.set_xlabel('Utility change')
# ax.set_ylabel('Number of houses')
# ppt.savefig(folder_WS+'/hist_uchange_'+str(ind_WS)+'.png', bbox_inches='tight')






