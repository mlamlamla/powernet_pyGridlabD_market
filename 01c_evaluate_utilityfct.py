import pandas as pd
import numpy as np
import matplotlib.pyplot as ppt

# Estimate utility parameters for week
start = pd.Timestamp(2016,8,1)
end = pd.Timestamp(2016,8,8)
start = pd.Timestamp(2016,12,19)
end = pd.Timestamp(2016,12,26)
start = pd.Timestamp(2016,7,18)
end = pd.Timestamp(2016,7,25)
start = pd.Timestamp(2016,9,12)
end = pd.Timestamp(2016,9,19)

#Get load and prices
ind_b = 46
folder = 'Diss/Diss_' + "{:04d}".format(ind_b) #+ '_5min'
run = 'Diss'
city = 'Austin'
market_file = 'Ercot_HBSouth.csv'

#file = folder+'/settings/HVAC_settings.csv'

# retail_kWh = 0.03245935410676796 # (july) # 0.02391749988554048 (year) #USD/kWh
# retail_kWh = 0.0704575196993475 #first week of august
# retail_kWh = 0.02254690804746962 #mid-Dec
# retail_kWh = 0.04218369798830142 #3rd week July

print(start)
print(end)

#Get technical characterisics
file = folder+'/settings/HVAC_settings_'+str(start).split(' ')[0]+'_'+str(end).split(' ')[0]+'_mod.csv'
df_settings = pd.read_csv(file,index_col=[0])

#Cacluclate retail arte
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

net_demand  = total_load - PV_supply

retail_kWh = supply_cost/net_demand
print(retail_kWh)
	#import pdb; pdb.set_trace()

df_settings_ext = df_settings.copy()
df_settings_ext['cooling_setpoint'] = 0.0
df_settings_ext['heating_setpoint'] = 0.0

file = 'IEEE_123_homes_1min.glm'

glm = open(file,'r') 

for line in glm:
	if '\tname GLD_' in line:
		house = line.split(' ')[1].split(';')[0]
	elif '\tcooling_setpoint ' in line:
		cooling = float(line.split(' ')[1].split(';')[0])
		df_settings_ext['cooling_setpoint'].loc[house] = cooling
	elif '\theating_setpoint ' in line:
		heating = float(line.split(' ')[1].split(';')[0])
		df_settings_ext['heating_setpoint'].loc[house] = heating
		
glm.close()

#Default
df_settings_ext['alpha'] = retail_kWh*(1.-df_settings_ext['beta'])/(df_settings_ext['cooling_setpoint'] - df_settings_ext['heating_setpoint'])*(1./df_settings_ext['gamma_cool'] + 1./df_settings_ext['gamma_heat'])
df_settings_ext['comf_temperature'] = df_settings_ext['cooling_setpoint'] - retail_kWh*(1. - df_settings_ext['beta'])/(2*df_settings_ext['alpha']*df_settings_ext['gamma_cool'])
for ind in df_settings_ext.index:
	if df_settings_ext['heating_system'].loc[ind] == 'GAS':
		df_settings_ext['comf_temperature'].loc[ind] = (df_settings_ext['cooling_setpoint'].loc[ind] + df_settings_ext['heating_setpoint'].loc[ind])/2.
		df_settings_ext['alpha'].loc[ind] = retail_kWh*(1.-df_settings_ext['beta'].loc[ind])/(df_settings_ext['cooling_setpoint'].loc[ind] - df_settings_ext['heating_setpoint'].loc[ind])*1./(2*df_settings_ext['gamma_cool'].loc[ind])

df_settings_ext.to_csv(run+'/HVAC_settings_'+str(start).split(' ')[0]+'_'+str(end).split(' ')[0]+'_ext.csv')

#Temperature for price changes
p = retail_kWh
m = -1
df_settings_ext['T_equ'] = df_settings_ext['comf_temperature'] - (1. - df_settings_ext['beta'])/(2*df_settings_ext['alpha']*df_settings_ext['gamma_cool']*m)*p
#import pdb; pdb.set_trace()

