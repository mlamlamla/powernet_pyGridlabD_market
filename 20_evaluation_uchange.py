#This file compares the WS market participation with stationary price bids to fixed price scenarie

import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt

max_y = 145

# Energy procurement cost scenario
run = 'Diss'
ind_WS = 203 #124
ind_b = 90
ind_WS = 289

df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])

df_HVAC = pd.read_csv(run+'/HVAC_settings/'+df_settings['settings_file'].loc[ind_WS],index_col=[0])
start = pd.to_datetime(df_settings['start_time'].loc[ind_WS]) + pd.Timedelta(days=1)
end = pd.to_datetime(df_settings['end_time'].loc[ind_WS])
#import pdb; pdb.set_trace()

# ind_WS = 65
#start = pd.Timestamp(2016,1,1)
#end = pd.Timestamp(2017,1,1)

# ind_WS = 66
# start = pd.Timestamp(2016,9,12)
# end = pd.Timestamp(2016,9,19)

# Capacity scenario
# ind_WS = 64
# start = pd.Timestamp(2016,12,19)
# end = pd.Timestamp(2016,12,26)

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
df_WS.drop_duplicates(subset='timestamp',keep='last',inplace=True)
df_WS.set_index('timestamp',inplace=True)
df_WS = df_WS.loc[start:end]

assert len(df_WS) == len(df_slack)

# df_WS['system_load'] = df_slack['measured_real_power']
# df_WS['supply_cost'] = df_WS['system_load']/1000.*df_WS['RT']/12.
# supply_cost = df_WS['supply_cost'].sum()

df_WS['system_load'] = df_slack['measured_real_power']
supply_wlosses = (df_WS['system_load']/1000./12.).sum() # MWh
df_WS['supply_cost'] = df_WS['system_load']/1000.*df_WS['RT']/12.
supply_cost_wlosses = df_WS['supply_cost'].sum()

df_total_load_b = pd.read_csv(folder+'/total_load_all.csv',skiprows=range(8)) #in kW
df_total_load_b['# timestamp'] = df_total_load_b['# timestamp'].map(lambda x: str(x)[:-4])
df_total_load_b = df_total_load_b.iloc[:-1]
df_total_load_b['# timestamp'] = pd.to_datetime(df_total_load_b['# timestamp'])
df_total_load_b.set_index('# timestamp',inplace=True)
df_total_load_b = df_total_load_b.loc[start:end]
total_load = (df_total_load_b.sum(axis=1)/12.).sum() #kWh

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

df_hvac_load_WS = pd.read_csv(folder_WS+'/hvac_load_all.csv',skiprows=range(8))
df_hvac_load_WS['# timestamp'] = df_hvac_load_WS['# timestamp'].map(lambda x: str(x)[:-4])
df_hvac_load_WS = df_hvac_load_WS.iloc[:-1]
df_hvac_load_WS['# timestamp'] = pd.to_datetime(df_hvac_load_WS['# timestamp'])
df_hvac_load_WS.set_index('# timestamp',inplace=True) 
df_hvac_load_WS = df_hvac_load_WS.loc[start:end]

df_total_load_WS = pd.read_csv(folder_WS+'/total_load_all.csv',skiprows=range(8))
df_total_load_WS['# timestamp'] = df_total_load_WS['# timestamp'].map(lambda x: str(x)[:-4])
df_total_load_WS = df_total_load_WS.iloc[:-1]
df_total_load_WS['# timestamp'] = pd.to_datetime(df_total_load_WS['# timestamp'])
df_total_load_WS.set_index('# timestamp',inplace=True) 
df_total_load_WS = df_total_load_WS.loc[start:end]

df_slack_WS = pd.read_csv(folder_WS + '/load_node_149.csv',skiprows=range(8))
df_slack_WS['# timestamp'] = df_slack_WS['# timestamp'].map(lambda x: str(x)[:-4])
df_slack_WS = df_slack_WS.iloc[:-1]
df_slack_WS['# timestamp'] = pd.to_datetime(df_slack_WS['# timestamp'])
df_slack_WS.set_index('# timestamp',inplace=True)
df_slack_WS = df_slack_WS.loc[start:end]
df_slack_WS = df_slack_WS/1000 #kW

df_prices = pd.read_csv(folder_WS+'/df_prices.csv',index_col=[0],parse_dates=True)
df_prices_1min = df_prices.copy()
df_prices_1min = df_prices_1min.loc[start:end]

# Calculate fixed retail rat for unresponsive load

loss_comp = (df_prices_1min['clearing_price'] - df_WS['RT']).median() # if unconstrained !!
total_consumption = (df_slack_WS['measured_real_power']/12./1000.).sum()
total_energy_proccost = (df_WS['RT']*df_slack_WS['measured_real_power']/12./1000.).sum()
total_consumpton_HVAC = (df_hvac_load_WS/12./1000.).sum().sum()
total_income_HVAC = (df_prices_1min['clearing_price']*df_hvac_load_WS.sum(axis=1)/12./1000.).sum()
retail_MWh_LEM = (total_energy_proccost - total_income_HVAC)/(total_consumption - total_consumpton_HVAC)
retail_kWh_LEM = retail_MWh_LEM/1000.

#Benchmark
if recalculate_df_welfare:
	df_u = df_T.copy()
	df_u_b = df_T_b.copy()
	#df_welfare = pd.DataFrame(index=df_u.columns,columns=['fixed_u','fixed_cost','fixed_T_mean','fixed_T_var','fixed_av_retail','LEM_u','LEM_cost','LEM_T_mean','LEM_T_var','LEM_av_retail'])
	df_welfare = pd.DataFrame(index=df_u.columns,columns=['fixed_u','fixed_cost_HVAC','fixed_kWh_unresp','fixed_cost_unresp','fixed_T_mean','fixed_T_min','fixed_T_max','fixed_T_min5','fixed_T_max95','fixed_T_var','fixed_av_retail','LEM_u','LEM_cost_HVAC','LEM_kWh_unresp','LEM_cost_unresp','LEM_T_mean','LEM_T_min','LEM_T_max','LEM_T_min5','LEM_T_max95','LEM_T_var','LEM_av_retail_total','LEM_av_retail_HVAC','corr_total_WS','corr_HVAC_WS'])
	for col in df_u.columns:
		#print(col)
		#import pdb; pdb.set_trace()

		alpha = df_HVAC['alpha'].loc[col]
		T_com = df_HVAC['comf_temperature'].loc[col]
		#import pdb; pdb.set_trace()
		df_u[col] = (df_u[col] - T_com)
		df_u[col] = -alpha*df_u[col].pow(2)
		sum_u = (df_u[col] - df_hvac_load_WS[col]/12.*df_prices_1min['clearing_price']/1000.).sum() # This includes the loss component
		df_welfare['LEM_u'].loc[col] = df_u[col].sum()
		df_welfare['LEM_cost_HVAC'].loc[col] = (df_hvac_load_WS[col]/12.*df_prices_1min['clearing_price']/1000.).sum() # clearing price already includes loss component
		df_welfare['LEM_kWh_unresp'].loc[col] = ((df_total_load_WS[col] - df_hvac_load_WS[col])/12.).sum()
		df_welfare['LEM_cost_unresp'].loc[col] = ((df_total_load_WS[col] - df_hvac_load_WS[col])/12.).sum()*retail_kWh_LEM
		df_welfare['LEM_T_mean'].loc[col] = df_T[col].mean()
		df_welfare['LEM_T_var'].loc[col] = df_T[col].var()
		df_welfare['LEM_av_retail_total'].loc[col] = 1000*(df_total_load_WS[col]/12.*df_prices_1min['clearing_price']/1000.).sum()/(df_total_load_WS[col].sum()/12.)
		df_welfare['LEM_av_retail_HVAC'].loc[col] = 1000*(df_hvac_load_WS[col]/12.*df_prices_1min['clearing_price']/1000.).sum()/(df_hvac_load_WS[col].sum()/12.)

		df_u_b[col] = (df_u_b[col] - T_com)
		df_u_b[col] = -alpha*df_u_b[col].pow(2)
		sum_u_b = (df_u_b[col] - df_hvac_load_b[col]/12.*retail_MWh/1000.).sum()
		df_welfare['fixed_u'].loc[col] = df_u_b[col].sum()
		df_welfare['fixed_cost_HVAC'].loc[col] = (df_hvac_load_b[col]/12.*retail_MWh/1000.).sum()
		df_welfare['fixed_kWh_unresp'].loc[col] = ((df_total_load_b[col] - df_hvac_load_b[col])/12.).sum()
		df_welfare['fixed_cost_unresp'].loc[col] = ((df_total_load_b[col] - df_hvac_load_b[col])/12.).sum()*retail_MWh/1000.
		df_welfare['fixed_T_mean'].loc[col] = df_T_b[col].mean()
		df_welfare['fixed_T_var'].loc[col] = df_T_b[col].var()
		df_welfare['fixed_av_retail'].loc[col] = retail_MWh

		df_welfare['corr_total_WS'].loc[col] = df_total_load_b[col].corr(df_WS['RT'])
		df_welfare['corr_HVAC_WS'].loc[col] = df_hvac_load_b[col].corr(df_WS['RT'])

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
		
		df_welfare.to_csv(folder_WS + '/df_welfare_wcost.csv')
		df = df_HVAC.join(df_welfare)
		df.to_csv(folder_WS + '/df_welfare_wcost_withparameters.csv')

df_welfare = pd.read_csv(folder_WS + '/df_welfare_wcost_withparameters.csv',index_col=[0],parse_dates=True)

# Bill change overall

print('Bills overall: fixed --> LEM')
print(df_welfare['fixed_cost_unresp'].sum() + df_welfare['fixed_cost_HVAC'].sum())
print(df_welfare['LEM_cost_unresp'].sum() + df_welfare['LEM_cost_HVAC'].sum())
print('Total savings')
print((df_welfare['fixed_cost_unresp'].sum() + df_welfare['fixed_cost_HVAC'].sum()) -(df_welfare['LEM_cost_unresp'].sum() + df_welfare['LEM_cost_HVAC'].sum()))

print('Total savings unresponsive load')
print((df_welfare['fixed_cost_unresp'].sum()) -(df_welfare['LEM_cost_unresp'].sum()))
print('Total savings HVAC')
print((df_welfare['fixed_cost_HVAC'].sum()) -(df_welfare['LEM_cost_HVAC'].sum()))

print('RR wo LEM: '+str(retail_MWh/1000.))
print('RR for unresp load w LEM: '+str(retail_MWh_LEM/1000.))

df_welfare['bill_savings_total'] = (df_welfare['fixed_cost_unresp'] + df_welfare['fixed_cost_HVAC']) -(df_welfare['LEM_cost_unresp'] + df_welfare['LEM_cost_HVAC'])
print('Min bill savings: '+str(df_welfare['bill_savings_total'].min()))
print('Max bill savings: '+str(df_welfare['bill_savings_total'].max()))
print('Correlation P_cool:' + str(df_welfare['bill_savings_total'].astype('float64').corr(df_welfare['P_cool'].astype('float64'))))
print('Correlation fixed cost:' + str(df_welfare['bill_savings_total'].astype('float64').corr(df_welfare['fixed_cost_HVAC'].astype('float64'))))

df_welfare['bill_savings_rel'] = df_welfare['bill_savings_total']/df_welfare['fixed_cost_HVAC']*100.
print('Correlation rel savings with alpha: ' + str(df_welfare['bill_savings_rel'].astype('float64').corr(df_welfare['alpha'].astype('float64'))))
print('Correlation rel savings with cooling setpoint: ' + str(df_welfare['bill_savings_rel'].astype('float64').corr(df_welfare['cooling_setpoint'].astype('float64'))))
print('Correlation rel savings with beta: ' + str(df_welfare['bill_savings_rel'].astype('float64').corr(df_welfare['beta'].astype('float64'))))

#import pdb; pdb.set_trace()

df_welfare['bill_savings_total'].astype('float64').corr(df_welfare['alpha'].astype('float64'))

# Welfare change with regard to HVAC

df_welfare['u_change_HVAC'] = (df_welfare['LEM_u'] - df_welfare['LEM_cost_HVAC']) - (df_welfare['fixed_u'] - df_welfare['fixed_cost_HVAC'])
df_welfare['u_change_tot'] = df_welfare['u_change_HVAC'] - df_welfare['LEM_cost_unresp'] + df_welfare['fixed_cost_unresp']

#import pdb; pdb.set_trace()
print('Average utility change HVAC only')
print(df_welfare['u_change_HVAC'].mean())
print('Total utility change')
print(df_welfare['u_change_HVAC'].sum())

print('Average utility change')
print(df_welfare['u_change_tot'].mean())
print('Total utility change')
print(df_welfare['u_change_tot'].sum())

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
fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.hist(df_welfare['u_change_HVAC'],bins=20,color='0.75',edgecolor='0.5')
#ax.set_ylim(0,75)
if df_welfare['u_change_HVAC'].min() > 0.0:
	ax.set_xlim(0,df_welfare['u_change_HVAC'].max()*1.05)
else:
	ax.vlines(0,0,max_y,'k',lw=1)
ax.set_xlabel('Utility change [USD]')
if max_y > 0.0:
	ax.set_ylim(0,max_y)
ax.set_ylabel('Number of houses')
ppt.savefig(folder_WS+'/hist_uchange_'+str(ind_WS)+'_HVAC.png', bbox_inches='tight')
ppt.savefig(folder_WS+'/hist_uchange_'+str(ind_WS)+'_HVAC.pdf', bbox_inches='tight')

#Histogram utility change
fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.hist(df_welfare['u_change_tot'],bins=20,color='0.75',edgecolor='0.5')
#ax.set_ylim(0,75)
if df_welfare['u_change_tot'].min() > 0.0:
	ax.set_xlim(0,df_welfare['u_change_tot'].max()*1.05)
else:
	ax.vlines(0,0,max_y,'k',lw=1)
ax.set_xlabel('Utility change [USD]')
if max_y > 0.0:
	ax.set_ylim(0,max_y)
ax.set_ylabel('Number of houses')
ppt.savefig(folder_WS+'/hist_uchange_'+str(ind_WS)+'_tot.png', bbox_inches='tight')
ppt.savefig(folder_WS+'/hist_uchange_'+str(ind_WS)+'_tot.pdf', bbox_inches='tight')

import pdb; pdb.set_trace()