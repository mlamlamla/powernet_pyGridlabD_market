#This file compares the WS market participation with stationary price bids to fixed price scenarie

import pandas as pd
import matplotlib.pyplot as ppt

# Energy procurement cost scenario

run = 'Diss'
ind_WS = 124
ind_b = 90

df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])
df_HVAC = pd.read_csv(run+'/HVAC_settings/'+df_settings['settings_file'].loc[ind_WS],index_col=[0])
start = pd.to_datetime(df_settings['start_time'].loc[ind_WS]) + pd.Timedelta(days=1)
end = pd.to_datetime(df_settings['end_time'].loc[ind_WS])

recread_data = True 
recalculate_df_welfare = True

house_no = 0

folder_b = run + '/Diss_00'+str(ind_b) #+'_5min'
folder_WS = run + '/Diss_0'+str(ind_WS)

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
# System variables
#
##################

city = 'Austin'
market_file = 'Ercot_LZ_SOUTH.csv'

# Benchmark / no TS

df_slack_b = pd.read_csv(folder_b+'/load_node_149.csv',skiprows=range(8))
df_slack_b['# timestamp'] = df_slack_b['# timestamp'].map(lambda x: str(x)[:-4])
df_slack_b = df_slack_b.iloc[:-1]
df_slack_b['# timestamp'] = pd.to_datetime(df_slack_b['# timestamp'])
df_slack_b.set_index('# timestamp',inplace=True)
df_slack_b = df_slack_b.loc[start:end]
df_slack_b = df_slack_b/1000 #kW

df_WS = pd.read_csv('glm_generation_'+city+'/'+market_file,parse_dates=[0])
df_WS.rename(columns={'Unnamed: 0':'timestamp'},inplace=True)
df_WS.set_index('timestamp',inplace=True)
df_WS = df_WS.loc[start:end]

df_WS['system_load_b'] = df_slack_b['measured_real_power']
df_WS['supply_cost_b'] = df_WS['system_load_b']/1000.*df_WS['RT']/12.
supply_cost_b = df_WS['supply_cost_b'].sum()

df_total_load_b = pd.read_csv(folder_b+'/total_load_all.csv',skiprows=range(8)) #in kW
df_total_load_b['# timestamp'] = df_total_load_b['# timestamp'].map(lambda x: str(x)[:-4])
df_total_load_b = df_total_load_b.iloc[:-1]
df_total_load_b['# timestamp'] = pd.to_datetime(df_total_load_b['# timestamp'])
df_total_load_b.set_index('# timestamp',inplace=True)
df_total_load_b = df_total_load_b.loc[start:end]
total_load_b = (df_total_load_b.sum(axis=1)/12.).sum() #kWh

# Supply costs for all (without losses)
df_WS['total_load_b'] = df_total_load_b.sum(axis=1)
df_WS['losses_b'] = df_WS['system_load_b'] - df_WS['total_load_b']
df_WS['supply_cost_wolosses_b'] = df_WS['total_load_b']/1000.*df_WS['RT']/12.
supply_cost_wolosses_b = df_WS['supply_cost_wolosses_b'].sum()
supply_wolosses_b = (df_WS['total_load_b']/12.).sum() # in kWh

try:
	df_inv_load_b = pd.read_csv(folder_b+'/total_P_Out.csv',skiprows=range(8)) #in W
	df_inv_load_b['# timestamp'] = df_inv_load_b['# timestamp'].map(lambda x: str(x)[:-4])
	df_inv_load_b = df_inv_load_b.iloc[:-1]
	df_inv_load_b['# timestamp'] = pd.to_datetime(df_inv_load_b['# timestamp'])
	df_inv_load_b.set_index('# timestamp',inplace=True)  
	df_inv_load_b = df_inv_load_b.loc[start:end]
	PV_supply_b = (df_inv_load_b.sum(axis=1)/1000./12.).sum() #in kWh
except:
	PV_supply_b = 0.0

print('RR with net-metering')
net_demand_b  = total_load_b - PV_supply_b

import pdb; pdb.set_trace()
retail_kWh_b = supply_cost_b/net_demand_b # gross retail tariff
retail_loss_kWh_b = (supply_cost_b - supply_cost_wolosses_b)/net_demand_b # loss component of retail tariff
retail_MWh_b = retail_kWh_b*1000.
print('RR (with loss component)')
print(retail_kWh_b)
print('loss component')
print(retail_loss_kWh_b)

# WS

df_slack_WS = pd.read_csv(folder_WS+'/load_node_149.csv',skiprows=range(8))
df_slack_WS['# timestamp'] = df_slack_WS['# timestamp'].map(lambda x: str(x)[:-4])
df_slack_WS = df_slack_WS.iloc[:-1]
df_slack_WS['# timestamp'] = pd.to_datetime(df_slack_WS['# timestamp'])
df_slack_WS.set_index('# timestamp',inplace=True)
df_slack_WS = df_slack_WS.loc[start:end]
df_slack_WS = df_slack_WS/1000 #kW

df_WS['system_load_WS'] = df_slack_WS['measured_real_power']
df_WS['supply_cost_WS'] = df_WS['system_load_WS']/1000.*df_WS['RT']/12.
supply_cost_WS = df_WS['supply_cost_WS'].sum()

df_total_load_WS = pd.read_csv(folder_WS+'/total_load_all.csv',skiprows=range(8)) #in kW
df_total_load_WS['# timestamp'] = df_total_load_WS['# timestamp'].map(lambda x: str(x)[:-4])
df_total_load_WS = df_total_load_WS.iloc[:-1]
df_total_load_WS['# timestamp'] = pd.to_datetime(df_total_load_WS['# timestamp'])
df_total_load_WS.set_index('# timestamp',inplace=True)
df_total_load_WS = df_total_load_WS.loc[start:end]
total_load_WS = (df_total_load_WS.sum(axis=1)/12.).sum() #kWh

try:
	df_inv_load_WS = pd.read_csv(folder_WS+'/total_P_Out.csv',skiprows=range(8)) #in W
	df_inv_load_WS['# timestamp'] = df_inv_load_WS['# timestamp'].map(lambda x: str(x)[:-4])
	df_inv_load_WS = df_inv_load_WS.iloc[:-1]
	df_inv_load_WS['# timestamp'] = pd.to_datetime(df_inv_load_WS['# timestamp'])
	df_inv_load_WS.set_index('# timestamp',inplace=True)  
	df_inv_load_WS = df_inv_load_WS.loc[start:end]
	PV_supply_WS = (df_inv_load_WS.sum(axis=1)/1000./12.).sum() #in kWh
except:
	PV_supply_WS = 0.0


##################
#
# House data
#
##################

#Data benchmark

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

# Data WS

df_hvac_load_WS = pd.read_csv(folder_WS+'/hvac_load_all.csv',skiprows=range(8))
df_hvac_load_WS['# timestamp'] = df_hvac_load_WS['# timestamp'].map(lambda x: str(x)[:-4])
df_hvac_load_WS = df_hvac_load_WS.iloc[:-1]
df_hvac_load_WS['# timestamp'] = pd.to_datetime(df_hvac_load_WS['# timestamp'])
df_hvac_load_WS.set_index('# timestamp',inplace=True) 
df_hvac_load_WS = df_hvac_load_WS.loc[start:end]

df_prices = pd.read_csv(folder_WS+'/df_prices.csv',index_col=[0],parse_dates=True)
df_prices_1min = df_prices.copy()
df_prices_1min = df_prices_1min.loc[start:end]

# Load and supply costs for each house : house - RR kWh total - RR cost total - RR kWh hvac - RR cost hvac - LEM kWh hvac - LEM cost hvac
df_results = pd.DataFrame(index = df_total_load_WS.columns, columns=['kWh unresp','cost unresp','bill unresp','RR kWh hvac','RR cost hvac','LEM kWh hvac','LEM cost hvac'],data=0.0)
for house in df_results.index:
	import pdb; pdb.set_trace()
	# Procurement cost unresponsive load
	df_results.at[house,'kWh unresp'] = (df_total_load_b[house]/12.).sum()
	df_results.at[house,'cost unresp'] = ((df_WS['RT'] + retail_loss_kWh_b)*(df_total_load_b[house] - df_hvac_load_b[house])/12./1000.).sum()
	df_results.at[house,'bill unresp'] = (retail_kWh_b*(df_total_load_b[house] - df_hvac_load_b[house])/12./1000.).sum() # given RR under 'no LEM'
	# Procurement cost under fixed retail rate
	df_results.at[house,'RR kWh hvac'] = (df_hvac_load_b[house]/12.).sum()
	df_results.at[house,'RR cost hvac'] = ((df_WS['RT'] + retail_loss_kWh_b)*df_hvac_load_b[house]/12./1000.).sum()
	df_results.at[house,'RR bill hvac'] = (retail_kWh_b*df_hvac_load_b[house]/12./1000.).sum()  # given RR under 'no LEM'
	# Procurement cost under LEM
	df_results.at[house,'LEM kWh hvac'] = (df_hvac_load_WS[house]/12.).sum()
	df_results.at[house,'LEM cost hvac'] = ((df_WS['RT'] + retail_loss_kWh_b)*df_hvac_load_WS[house]/12./1000.).sum()
	df_results.at[house,'LEM bill hvac'] = df_results.at[house,'LEM cost hvac']

##################
#
# Calculations
#
##################

#Benchmark
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
	sum_u = (df_u[col] - df_hvac_load_WS[col]/12.*df_prices_1min['clearing_price']/1000.).sum()
	df_welfare['LEM_u'].loc[col] = df_u[col].sum()
	df_welfare['LEM_cost'].loc[col] = (df_hvac_load_WS[col]/12.*df_prices_1min['clearing_price']/1000.).sum()
	df_welfare['LEM_T_mean'].loc[col] = df_T[col].mean()
	df_welfare['LEM_T_var'].loc[col] = df_T[col].var()
	df_welfare['LEM_av_retail'].loc[col] = 1000*(df_hvac_load_WS[col]/12.*df_prices_1min['clearing_price']/1000.).sum()/(df_hvac_load_WS[col].sum()/12.)

	df_u_b[col] = (df_u_b[col] - T_com)
	df_u_b[col] = -alpha*df_u_b[col].pow(2)
	sum_u_b = (df_u_b[col] - df_hvac_load_b[col]/12.*retail_MWh_b/1000.).sum()
	df_welfare['fixed_u'].loc[col] = df_u_b[col].sum()
	df_welfare['fixed_cost'].loc[col] = (df_hvac_load_b[col]/12.*retail_MWh_b/1000.).sum()
	df_welfare['fixed_T_mean'].loc[col] = df_T_b[col].mean()
	df_welfare['fixed_T_var'].loc[col] = df_T_b[col].var()
	df_welfare['fixed_av_retail'].loc[col] = retail_MWh_b
	
	#df_welfare.to_csv(folder_WS + '/df_welfare.csv')
	df = df_HVAC.join(df_welfare)
	#df.to_csv(folder_WS + '/df_welfare_withparameters.csv')
	#import pdb; pdb.set_trace()

df_welfare = df.copy()
#df_welfare = pd.read_csv(folder_WS + '/df_welfare_withparameters.csv',index_col=[0],parse_dates=True)
df_welfare['u_change'] = (df_welfare['LEM_u'] - df_welfare['LEM_cost']) - (df_welfare['fixed_u'] - df_welfare['fixed_cost'])
#import pdb; pdb.set_trace()
print('Average utility change under LEM')
print(df_welfare['u_change'].mean())
print('Total utility change')
print(df_welfare['u_change'].sum())

#import pdb; pdb.set_trace()

df_retail_rate = pd.DataFrame(index=range(len(df_welfare)),columns=['house_to_LEM','RR'],data=0.0)

df_welfare.sort_values(by='u_change',inplace=True)
df_retail_rate['house_to_LEM'] = df_welfare.index
df_retail_rate = df_retail_rate.join(df_results,on='house_to_LEM')
i = 0
for house in df_welfare.index:
	supply_cost_hvac_i = df_retail_rate['RR cost hvac'].iloc[:(i+1)].sum() # hvacs not on RR
	supply_hvac_i = df_retail_rate['RR kWh hvac'].iloc[:(i+1)].sum() # hvacs not on RR
	RR_i = (supply_cost_wolosses_b - supply_cost_hvac_i)/(supply_wolosses_b - supply_hvac_i)
	df_retail_rate.at[i,'RR'] = RR_i
	i += 1

import pdb; pdb.set_trace()
#Histogram utility change
fig = ppt.figure(figsize=(8,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.plot(df_retail_rate['RR'])
#ax.vlines(0,0,50)
#ax.set_ylim(0,50)
ax.set_xlabel('Number of houses in LEM')
ax.set_ylabel('Retail rate [USD/kWh]')
ppt.savefig(folder_WS+'/RR_change_'+str(ind_WS)+'.png', bbox_inches='tight')






