#This file evaluates the outcome for a single house in terms of temperature and bill for RR and LEM

import pandas as pd
import matplotlib.pyplot as ppt
import os

# Energy procurement cost scenario
run = 'Diss'
ind_WS = 124
ind_b = 90
house_no = 0

df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])
df_HVAC = pd.read_csv(run+'/HVAC_settings/'+df_settings['settings_file'].loc[ind_WS],index_col=[0])
start = pd.to_datetime(df_settings['start_time'].loc[ind_WS]) + pd.Timedelta(days=1)
end = pd.to_datetime(df_settings['end_time'].loc[ind_WS])

folder_b = run + '/Diss_00'+str(ind_b) #+'_5min'
folder_WS = run + '/Diss_00'+str(ind_WS)

##################
#
# Find relevant end date for specific simulation 
# 
##################

print(start)
df_T = pd.read_csv(folder_WS+'/T_all.csv',skiprows=range(8))
df_T['# timestamp'] = df_T['# timestamp'].map(lambda x: str(x)[:-4])
df_T = df_T.iloc[:-1]
df_T['# timestamp'] = pd.to_datetime(df_T['# timestamp'])
df_T.set_index('# timestamp',inplace=True)
df_T = df_T.loc[start:end]
end = df_T.index[-1]
print(end)

##################
#
# Calculate retail rate in fixed price scenario / no TS
#
##################

folder = folder_b
city = 'Austin'
market_file = df_settings['market_data'].loc[ind_WS]

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

print('RR with net-metering')
net_demand  = total_load - PV_supply

retail_kWh = supply_cost/net_demand
retail_MWh = retail_kWh*1000.
print(retail_kWh)

##################
#
# Read-in data
#
##################

#Data
df_T_b = pd.read_csv(folder_b+'/T_all.csv',skiprows=range(8))
df_T_b['# timestamp'] = df_T_b['# timestamp'].map(lambda x: str(x)[:-4])
df_T_b = df_T_b.iloc[:-1]
df_T_b['# timestamp'] = pd.to_datetime(df_T_b['# timestamp'])
df_T_b.set_index('# timestamp',inplace=True)
df_T_b = df_T_b.loc[start:end]

# df_hvac_load_b = pd.read_csv(folder_b+'/hvac_load_all.csv',skiprows=range(8))
# df_hvac_load_b['# timestamp'] = df_hvac_load_b['# timestamp'].map(lambda x: str(x)[:-4])
# df_hvac_load_b = df_hvac_load_b.iloc[:-1]
# df_hvac_load_b['# timestamp'] = pd.to_datetime(df_hvac_load_b['# timestamp'])
# df_hvac_load_b.set_index('# timestamp',inplace=True) 
# df_hvac_load_b = df_hvac_load_b.loc[start:end]

# df_hvac_load = pd.read_csv(folder_WS+'/hvac_load_all.csv',skiprows=range(8))
# df_hvac_load['# timestamp'] = df_hvac_load['# timestamp'].map(lambda x: str(x)[:-4])
# df_hvac_load = df_hvac_load.iloc[:-1]
# df_hvac_load['# timestamp'] = pd.to_datetime(df_hvac_load['# timestamp'])
# df_hvac_load.set_index('# timestamp',inplace=True) 
# df_hvac_load = df_hvac_load.loc[start:end]

df_prices = pd.read_csv(folder_WS+'/df_prices.csv',index_col=[0],parse_dates=True)
df_prices_1min = df_prices.copy()
df_prices_1min = df_prices_1min.loc[start:end]

df_awarded_bids = pd.DataFrame()
files_bids = os.listdir(folder_WS)
for file in files_bids:
	if 'awarded' in file:
		df = pd.read_csv(folder_WS+'/'+file,index_col=[0],parse_dates=True)
		if len(df_awarded_bids) == 0:
			df_awarded_bids = df.copy()
		else:
			df_awarded_bids = df_awarded_bids.append(df)

house = df_T_b.columns[house_no]
T_cool = df_HVAC['cooling_setpoint'].loc[house]
T_heat = df_HVAC['heating_setpoint'].loc[house]
T_comf = df_HVAC['comf_temperature'].loc[house]

df_awarded_bids.sort_values(by='timestamp',inplace=True)
df_awarded_bids_house = df_awarded_bids.loc[df_awarded_bids['appliance_name'] == house]
df_awarded_bids_house.set_index('timestamp',inplace=True)
df_awarded_bids_house.index = pd.to_datetime(df_awarded_bids_house.index)

##################
#
# Plot
#
##################

import matplotlib.pyplot as ppt
fig = ppt.figure(figsize=(12,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.plot(df_T_b[house],lw=1,color='0.75',label='RR')
lns2 = ppt.plot(df_T[house],lw=1,color='0.25',label='LEM')
ax.hlines(T_cool,start,end,'r','--')
ax.hlines(T_heat,start,end,'r','--')
ax.hlines(T_comf,start,end,'k','--')
ax.set_xlabel('Time')
ax.set_ylabel('Temperature')
ax.set_xlim(start,end)
ax.legend()
ppt.savefig(folder_WS+'/T_byhouse_'+house+'.png', bbox_inches='tight')

fig = ppt.figure(figsize=(12,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.plot(df_awarded_bids_house['bid_price'],marker='x',lw=1,color='0.75',label='bid_price')
lns = ppt.plot(df_prices_1min['clearing_price'],lw=1,color='0.25',label='clear_price')
ax.hlines(retail_kWh*1000,start,end,'r','--')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.set_xlim(start,end)
ax.legend()
ppt.savefig(folder_WS+'/Bids_byhouse_'+house+'.png', bbox_inches='tight')
import pdb; pdb.set_trace()