# Estimates alpha, beta, and gamma from year-long measurement under real-time price

# Estimates the distribution of beta depending on the estimation period

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as ppt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
ppt.rcParams['mathtext.fontset'] = 'stix'
ppt.rcParams['font.family'] = 'STIXGeneral'

share_t = 1/12. # 5 min / 1 hour

def estimate_houseparameters(house, df_house_year, start, end, df_estimates):
	# Single out relevant estimation period
	df_house = df_house_year.loc[start:end]

	# Calculate mean HVAC power for cooling and heating
	P_heat = (df_house['HEAT']*df_house['P_t']).sum()/df_house['HEAT'].sum() # average heat P
	if np.isnan(P_heat):
		df_house['P_heat'] = 0.0 # If no heating takes place in the period of interest
		P_heat = 0.0
	elif df_estimates['heating_system'].loc[house] =='GAS':
		df_house['P_heat'] = 0.0 # If heating is GAS
		P_heat = 0.0
	else:
		df_house['P_heat'] = P_heat
	P_cool = (df_house['COOL']*df_house['P_t']).sum()/df_house['COOL'].sum() # average cool P
	if np.isnan(P_cool):
		df_house['P_cool'] = 0.0 # If no cooling takes place in the period of interest
		P_cool = 0.0
	else:
		df_house['P_cool'] = P_cool

	# Prepare independent variables
	# theta_t+1 - theta^out = beta * (theta_t - theta^out) + gamma_HEAT * (HEAT*P_heat) + gamma_COOL * (COOL*P_cool)

	df_house['HEAT_P'] = df_house['HEAT']*df_house['P_heat']*share_t
	df_house['COOL_P'] = -df_house['COOL']*df_house['P_cool']*share_t

	#Estimate parameters

	reg = linear_model.LinearRegression(fit_intercept=False)
	if (P_heat > 0.0) and (P_cool > 0.0):
		reg.fit(df_house[['DeltaT_t','HEAT_P','COOL_P']], df_house['DeltaT_t+1'])
		beta = reg.coef_[0]
		gamma_heat = reg.coef_[1]
		gamma_cool = reg.coef_[2]
	elif (P_heat == 0.0) and (P_cool > 0.0):
		reg.fit(df_house[['DeltaT_t','COOL_P']], df_house['DeltaT_t+1'])
		beta = reg.coef_[0]
		gamma_heat = 0.0
		gamma_cool = reg.coef_[1]
	elif (P_heat > 0.0) and (P_cool == 0.0):
		reg.fit(df_house[['DeltaT_t','HEAT_P']], df_house['DeltaT_t+1'])
		beta = reg.coef_[0]
		gamma_heat = reg.coef_[1]
		gamma_cool = 0.0
	else:
		reg.fit(df_house[['DeltaT_t']], df_house['DeltaT_t+1'])
		beta = reg.coef_[0]
		gamma_heat = 0.0
		gamma_cool = 0.0

	if gamma_heat < 0.0:
		import pdb; pdb.set_trace()

	try:
		df_house['T_t+1_est'] = beta*df_house['T_t'] + (1-beta)*df_house['T_out_t'] + gamma_heat*df_house['HEAT_P'] + gamma_cool*df_house['COOL_P'] # already includes 1/12 + sign for cooling
		R2 = r2_score(df_house['T_t+1'], df_house['T_t+1_est'])
	except:
		R2 = 0.0

	df_estimates['beta'].loc[house] = beta
	df_estimates['gamma_cool'].loc[house] = gamma_cool
	df_estimates['P_cool'].loc[house] = P_cool
	df_estimates['gamma_heat'].loc[house] = gamma_heat
	df_estimates['P_heat'].loc[house] = P_heat
	df_estimates['R2'].loc[house] = R2

	return df_estimates

def get_retailrate(start,end):

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

	net_demand  = total_load - PV_supply

	retail_kWh = supply_cost/net_demand
	return retail_kWh


def plot_error(df_house_OFF,name):
	fig = ppt.figure(figsize=(4,4),dpi=150)   
	ax = fig.add_subplot(111)
	lns1 = ax.scatter(df_house_OFF['T_t+1'],df_house_OFF['T_t+1_est'],c='0.5',marker='x',rasterized=True)
	#lns3 = ax.scatter(df_house_OFF['T_t+1'],df_house_OFF['T_t'],c='r')
	min_T = df_house_OFF['T_t+1'].min()
	max_T = df_house_OFF['T_t+1'].max()
	lns2 = ax.plot(np.arange(min_T,max_T,0.01),np.arange(min_T,max_T,0.01),'k')
	ax.set_xlabel('Actual temperature $\\theta_t$')
	ax.set_ylabel('Estimated temperature $\\widehat{\\theta}_t$')
	if not os.path.isdir(results_folder + '/' + house):
		os.mkdir(results_folder + '/' + house)
	ppt.savefig(results_folder + '/' + house + '/ParEst_'+str(start)+'_'+str(end)+'.pdf', bbox_inches='tight')
	ppt.close()
	# Calculate R2
	#import pdb; pdb.set_trace()
	SS_tot = ((df_house_OFF['T_t+1'] - df_house_OFF['T_t+1'].mean()).pow(2)).sum()
	SS_res = ((df_house_OFF['T_t+1'] - df_house_OFF['T_t+1_est']).pow(2)).sum()
	R2 = 1. - SS_res/SS_tot
	return R2

################
#
# Raw data for parameter estimation
#
################

# start = pd.Timestamp(2016,12,12)
# end = pd.Timestamp(2016,12,19)

ind_base = 90 # no PV 46 # 10 % PV
use_existing_beta = False
folder = 'Diss/Diss_' + "{:04d}".format(ind_base) # + '_5min'
results_folder = 'Diss'
city = 'Austin'
market_file = 'Ercot_HBSouth.csv'

start_backup = pd.Timestamp(2016,1,1)
end_backup = pd.Timestamp(2017,1,1)

################
#
# Read in temperatures for parameter estimation
#
################

#Total hvac load
df_hvac_load_year = pd.read_csv(folder+'/hvac_load_all.csv',skiprows=range(8))
df_hvac_load_year['# timestamp'] = df_hvac_load_year['# timestamp'].map(lambda x: str(x)[:-4])
df_hvac_load_year['# timestamp'] = pd.to_datetime(df_hvac_load_year['# timestamp'])
df_hvac_load_year.set_index('# timestamp',inplace=True)
df_hvac_load_ON_year = df_hvac_load_year.copy()
df_hvac_load_ON_year[df_hvac_load_ON_year > 0] = 1. # HVAC ON/OFF

#Temperature inside
df_T_in_year = pd.read_csv(folder+'/T_all.csv',skiprows=range(8))
df_T_in_year['# timestamp'] = df_T_in_year['# timestamp'].map(lambda x: str(x)[:-4])
df_T_in_year['# timestamp'] = pd.to_datetime(df_T_in_year['# timestamp'])
df_T_in_year.set_index('# timestamp',inplace=True)

#Temperature inside
df_T_out_year = pd.read_csv(folder+'/T_out.csv',skiprows=range(8))
df_T_out_year['# timestamp'] = df_T_out_year['# timestamp'].map(lambda x: str(x)[:-4])
df_T_out_year['# timestamp'] = pd.to_datetime(df_T_out_year['# timestamp'])
df_T_out_year.set_index('# timestamp',inplace=True)
df_T_out_year = df_T_out_year.loc[df_T_out_year.index.minute%5 == 0]
df_T_out_year.append(pd.DataFrame(index=[df_T_out_year.index[-1] + pd.Timedelta(minutes=5)],columns=['temperature'],data=df_T_out_year['temperature'].values[-1]))

################
#
# Read out HVAC systems
#
################

df_settings_week = pd.DataFrame(index=df_hvac_load_year.columns,columns=['heating_system','heating_setpoint','cooling_system','cooling_setpoint','beta','gamma_cool','P_cool','gamma_heat','P_heat','R2'])

glm_in = open('IEEE_123_homes_1min.glm',"r")
i = 0
for line in glm_in:
	if 'cooling_system_type' in line:
		system = line.split(' ')[-1].split(';')[0]
		house_ind = df_settings_week.index[i]
		df_settings_week['cooling_system'].loc[house_ind] = system
	elif 'heating_system_type' in line:
		system = line.split(' ')[-1].split(';')[0]
		house_ind = df_settings_week.index[i]
		df_settings_week['heating_system'].loc[house_ind] = system
		i += 1
glm_in.close()

glm_in = open('IEEE_123_homes_1min.glm',"r")
for line in glm_in:
	if '\tname GLD_' in line:
		house = line.split(' ')[1].split(';')[0]
	elif '\tcooling_setpoint ' in line:
		cooling = float(line.split(' ')[1].split(';')[0])
		df_settings_week['cooling_setpoint'].loc[house] = cooling
	elif '\theating_setpoint ' in line:
		heating = float(line.split(' ')[1].split(';')[0])
		df_settings_week['heating_setpoint'].loc[house] = heating		
glm_in.close()

df_settings_year = df_settings_week.copy() # for missing values

start = pd.Timestamp(2016,1,4)
end = pd.Timestamp(2016,1,11)

while True:
	print(start)
	if end.year == 2017:
		break

	retail_kWh = get_retailrate(start,end)
	#import pdb; pdb.set_trace()

	################
	#
	# Estimate parameters for each house
	#
	################

	for house in df_hvac_load_year.columns:
		#print(house)

		# Assemble data for house

		df_house = pd.DataFrame(index=df_hvac_load_year.index,columns=['T_t+1','T_t','T_out_t','hvac_t','P_t','HEAT','P_heat','COOL','P_cool'],data=0.0)
		df_house['T_t'] = df_T_in_year[house].values
		try:
			df_house['T_out_t'] = df_T_out_year.values
		except:
			df_house['T_out_t'].iloc[:-1] = df_T_out_year.values.reshape(len(df_T_out_year),)
		df_house['hvac_t'] = df_hvac_load_ON_year[house].values
		df_house['P_t'] = df_hvac_load_year[house].values
		df_house['T_t+1'].iloc[:-1] = df_house['T_t'].iloc[1:].values
		df_house['DeltaT_t'] = df_house['T_t'] - df_house['T_out_t'] # Mode: HEAT or COOL
		df_house['DeltaT_t+1'] = df_house['T_t+1'] - df_house['T_out_t']
		df_house['HEAT'].loc[(df_house['T_t'] <= df_house['T_t+1']) & (df_house['hvac_t'] > 0.0)] = 1
		df_house['COOL'].loc[(df_house['T_t'] > df_house['T_t+1']) & (df_house['hvac_t'] > 0.0)] = 1
		df_house_year = df_house.copy()

		# Cut year-long ts for estimation to relevant estimation period

		# df_hvac_load = df_hvac_load_year.loc[start:end] # HVAC power
		# df_hvac_load_ON = df_hvac_load_ON_year.loc[start:end] # HVAC ON/OFF
		# df_T_in = df_T_in_year.loc[start:end]
		# df_T_out = df_T_out_year.loc[start:end]

		# Estimate house parameters and use backup values if 0.0
		#import pdb; pdb.set_trace()

		df_settings_year = estimate_houseparameters(house,df_house_year,start_backup,end_backup,df_settings_year)
		df_settings_week = estimate_houseparameters(house,df_house_year,start,end,df_settings_week)
		if df_settings_week['P_cool'].loc[house] == 0.0:
			#import pdb; pdb.set_trace()
			df_settings_week['P_cool'].loc[house] = df_settings_year['P_cool'].loc[house]
			df_settings_week['gamma_cool'].loc[house] = df_settings_year['gamma_cool'].loc[house]
		if df_settings_week['P_heat'].loc[house] == 0.0:
			#import pdb; pdb.set_trace()
			df_settings_week['P_heat'].loc[house] = df_settings_year['P_heat'].loc[house]
			df_settings_week['gamma_heat'].loc[house] = df_settings_year['gamma_heat'].loc[house]

	# Include alpha

	df_settings_ext = df_settings_week.copy()
	df_settings_ext['alpha'] = 0.0
	df_settings_ext['comf_temperature'] = 0.0
	for ind in df_settings_ext.index:
		if df_settings_ext['heating_system'].loc[ind] != 'GAS':
			df_settings_ext['alpha'].loc[ind] = (1.-df_settings_ext['beta'].loc[ind])/2*retail_kWh/(df_settings_ext['cooling_setpoint'].loc[ind] - df_settings_ext['heating_setpoint'].loc[ind])*(1./df_settings_ext['gamma_cool'].loc[ind] + 1./df_settings_ext['gamma_heat'].loc[ind])
			# Last working version (morning 10/09) - no /2
			#df_settings_ext['alpha'].loc[ind] = (1.-df_settings_ext['beta'].loc[ind])*retail_kWh/(df_settings_ext['cooling_setpoint'].loc[ind] - df_settings_ext['heating_setpoint'].loc[ind])*(1./df_settings_ext['gamma_cool'].loc[ind] + 1./df_settings_ext['gamma_heat'].loc[ind])
			df_settings_ext['comf_temperature'].loc[ind] = df_settings_ext['cooling_setpoint'].loc[ind] - retail_kWh*(1. - df_settings_ext['beta'].loc[ind])/(2*df_settings_ext['alpha'].loc[ind]*df_settings_ext['gamma_cool'].loc[ind])

	df_settings_ext_noGAS = df_settings_ext.loc[df_settings_ext['heating_system'] != 'GAS']
	comf_temp_share = ((df_settings_ext_noGAS['comf_temperature'] - df_settings_ext_noGAS['heating_setpoint'])/(df_settings_ext_noGAS['cooling_setpoint'] - df_settings_ext_noGAS['heating_setpoint'])).mean()

	for ind in df_settings_ext.index:
		if df_settings_ext['heating_system'].loc[ind] == 'GAS':
			#df_settings_ext['comf_temperature'].loc[ind] = 0.5*df_settings_ext['cooling_setpoint'].loc[ind] + 0.5*df_settings_ext['heating_setpoint'].loc[ind]
			#df_settings_ext['alpha'].loc[ind] = retail_kWh*(1.-df_settings_ext['beta'].loc[ind])/(df_settings_ext['cooling_setpoint'].loc[ind] - df_settings_ext['heating_setpoint'].loc[ind])*1./(2*df_settings_ext['gamma_cool'].loc[ind])
			df_settings_ext['comf_temperature'].loc[ind] = comf_temp_share*df_settings_ext['cooling_setpoint'].loc[ind] + (1. - comf_temp_share)*df_settings_ext['heating_setpoint'].loc[ind]
			df_settings_ext['alpha'].loc[ind] = retail_kWh*(1.-df_settings_ext['beta'].loc[ind])/(df_settings_ext['cooling_setpoint'].loc[ind] - df_settings_ext['comf_temperature'].loc[ind])*1./(2*df_settings_ext['gamma_cool'].loc[ind])
		
	#import pdb; pdb.set_trace()
	df_settings_ext.to_csv(results_folder +'/HVAC_settings/HVAC_settings_' +str(start).split(' ')[0]+'_'+str(end).split(' ')[0]+'_'+str(ind_base)+'_OLS.csv')
	#import pdb; pdb.set_trace()

	start += pd.Timedelta(weeks=1)
	end += pd.Timedelta(weeks=1)