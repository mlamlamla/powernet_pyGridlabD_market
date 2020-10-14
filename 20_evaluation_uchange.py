#This file compares the WS market participation with stationary price bids to fixed price scenarie

import pandas as pd
import matplotlib.pyplot as ppt

# Energy procurement cost scenario
run = 'Diss'
ind_WS = 88

df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])
#df_HVAC = pd.read_csv(run+'/HVAC_settings_'+str(start).split(' ')[0]+'_'+str(end).split(' ')[0]+'_OLS4.csv',index_col=[0])
df_HVAC = pd.read_csv(run+'/'+df_settings['settings_file'].loc[ind_WS],index_col=[0])
start = pd.to_datetime(df_settings['start_time'].loc[ind_WS]) + pd.Timedelta(days=1)
end = pd.to_datetime(df_settings['end_time'].loc[ind_WS])
#import pdb; pdb.set_trace()

# ind_WS = 65
# start = pd.Timestamp(2016,7,18)
# end = pd.Timestamp(2016,7,25)

# ind_WS = 66
# start = pd.Timestamp(2016,9,12)
# end = pd.Timestamp(2016,9,19)

# Capacity scenario
# ind_WS = 64
# start = pd.Timestamp(2016,12,19)
# end = pd.Timestamp(2016,12,26)

recread_data = True 
recalculate_df_welfare = True

ind_b = 46
house_no = 0

folder_b = run + '/Diss_00'+str(ind_b) #+'_5min'
folder_WS = run + '/Diss_00'+str(ind_WS)
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

df_inv_load = pd.read_csv(folder+'/total_P_Out.csv',skiprows=range(8)) #in W
df_inv_load['# timestamp'] = df_inv_load['# timestamp'].map(lambda x: str(x)[:-4])
df_inv_load = df_inv_load.iloc[:-1]
df_inv_load['# timestamp'] = pd.to_datetime(df_inv_load['# timestamp'])
df_inv_load.set_index('# timestamp',inplace=True)  
df_inv_load = df_inv_load.loc[start:end]
PV_supply = (df_inv_load.sum(axis=1)/1000./12.).sum() #in kWh

print('RR with net-metering')
net_demand  = total_load - PV_supply

retail_kWh = supply_cost/net_demand
retail_MWh = retail_kWh*1000.
print(retail_kWh)

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
	df_welfare = pd.DataFrame(index=df_u.columns,columns=['fixed_u','fixed_cost','fixed_T_mean','fixed_T_var','fixed_av_retail','LEM_u','LEM_cost','LEM_T_mean','LEM_T_var','LEM_av_retail'])
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
		
		df_welfare.to_csv(folder_WS + '/df_welfare.csv')
		df = df_HVAC.join(df_welfare)
		df.to_csv(folder_WS + '/df_welfare_withparameters.csv')
		#import pdb; pdb.set_trace()

df_welfare = pd.read_csv(folder_WS + '/df_welfare_withparameters.csv',index_col=[0],parse_dates=True)
df_welfare['u_change'] = (df_welfare['LEM_u'] - df_welfare['LEM_cost']) - (df_welfare['fixed_u'] - df_welfare['fixed_cost'])
print('Average utility change')
print(df_welfare['u_change'].mean())
print('Total utility change')
print(df_welfare['u_change'].sum())

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

#Histogram utility change
fig = ppt.figure(figsize=(8,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.hist(df_welfare['u_change'],bins=20,color='0.75',edgecolor='0.5')
ax.vlines(0,0,50)
ax.set_ylim(0,50)
ax.set_xlabel('Utility change')
ax.set_ylabel('Number of houses')
ppt.savefig(folder_WS+'/hist_uchange_'+str(ind_WS)+'.png', bbox_inches='tight')






