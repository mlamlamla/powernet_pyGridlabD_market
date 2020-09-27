#Evaluates the physical parameters of the system

import pandas as pd
import numpy as np
import matplotlib.pyplot as ppt
ppt.rcParams['mathtext.fontset'] = 'stix'
ppt.rcParams['font.family'] = 'STIXGeneral'

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
	ppt.savefig(folder+'/settings/'+house+'_temperature_estimation_'+name+'.pdf', bbox_inches='tight')
	ppt.close()


ind = 46
use_existing_beta = False
start = pd.Timestamp(2016,12,19)
end = pd.Timestamp(2016,12,26)

folder = 'Diss/Diss_' + "{:04d}".format(ind) # + '_5min'

#df_settings_summer = pd.read_csv('Diss/Diss_' + "{:04d}".format(40) + '_5min/settings/HVAC_settings.csv',index_col=[0])

#Total hvac load
df_hvac_load = pd.read_csv(folder+'/hvac_load_all.csv',skiprows=range(8))
df_hvac_load['# timestamp'] = df_hvac_load['# timestamp'].map(lambda x: str(x)[:-4])
df_hvac_load['# timestamp'] = pd.to_datetime(df_hvac_load['# timestamp'])
df_hvac_load.set_index('# timestamp',inplace=True)
df_hvac_load = df_hvac_load.loc[start:end]
df_hvac_load_ON = df_hvac_load.copy()

#Read out HVAC systems
df_settings = pd.DataFrame(index=df_hvac_load.columns,columns=['heating_system','cooling_system','beta','gamma_cool','P_cool','gamma_heat','P_heat'])
glm_in = open('IEEE_123_homes_1min.glm',"r")
i = 0
for line in glm_in:
	if 'cooling_system_type' in line:
		system = line.split(' ')[-1].split(';')[0]
		house_ind = df_settings.index[i]
		df_settings['cooling_system'].loc[house_ind] = system
	elif 'heating_system_type' in line:
		system = line.split(' ')[-1].split(';')[0]
		house_ind = df_settings.index[i]
		df_settings['heating_system'].loc[house_ind] = system
		i += 1

#Temperature inside
df_T_in = pd.read_csv(folder+'/T_all.csv',skiprows=range(8))
df_T_in['# timestamp'] = df_T_in['# timestamp'].map(lambda x: str(x)[:-4])
df_T_in['# timestamp'] = pd.to_datetime(df_T_in['# timestamp'])
df_T_in.set_index('# timestamp',inplace=True)
df_T_in = df_T_in.loc[start:end]

#Temperature inside
df_T_mass = pd.read_csv(folder+'/Tm_all.csv',skiprows=range(8))
df_T_mass['# timestamp'] = df_T_mass['# timestamp'].map(lambda x: str(x)[:-4])
df_T_mass['# timestamp'] = pd.to_datetime(df_T_mass['# timestamp'])
df_T_mass.set_index('# timestamp',inplace=True)
df_T_mass = df_T_mass.loc[start:end]

#Temperature inside
df_T_out = pd.read_csv(folder+'/T_out.csv',skiprows=range(8))
df_T_out['# timestamp'] = df_T_out['# timestamp'].map(lambda x: str(x)[:-4])
df_T_out['# timestamp'] = pd.to_datetime(df_T_out['# timestamp'])
df_T_out.set_index('# timestamp',inplace=True)
df_T_out = df_T_out.loc[start:end]
df_T_out = df_T_out.loc[df_T_out.index.minute%5 == 0]
df_T_out.append(pd.DataFrame(index=[df_T_out.index[-1] + pd.Timedelta(minutes=5)],columns=['temperature'],data=df_T_out['temperature'].values[-1]))
#import pdb; pdb.set_trace()

#Compile dataset
for house in df_hvac_load.columns:
	print(house)
	df_hvac_load_ON[house].loc[df_hvac_load_ON[house] > 0] = 1.

	#import pdb; pdb.set_trace()

	df_house = pd.DataFrame(index=df_hvac_load.index,columns=['T_t','Tm_t','T_out_t','hvac_t','P_t','T_t+1'],data=0.0)
	df_house['T_t'] = df_T_in[house].values
	df_house['Tm_t'] = df_T_mass[house].values
	try:
		df_house['T_out_t'] = df_T_out.values
	except:
		df_house['T_out_t'].iloc[:-1] = df_T_out.values.reshape(len(df_T_out),)
	df_house['hvac_t'] = df_hvac_load_ON[house].values
	df_house['P_t'] = df_hvac_load[house].values
	df_house['T_t+1'].iloc[:-1] = df_house['T_t'].iloc[1:].values
	#df_house = df_house.iloc[:-1]

	#Estimate beta - Analysis for houses with HVAC OFF

	df_house_OFF = df_house.copy()
	df_house_OFF = df_house_OFF.loc[df_house_OFF['hvac_t'] == 0.0]
	df_house_OFF['Delta_t'] = df_house_OFF['T_out_t'] - df_house_OFF['T_t']
	df_house_OFF['Delta_t+1'] = df_house_OFF['T_out_t'] - df_house_OFF['T_t+1']
	df_house_OFF['mass-in'] = df_house_OFF['Tm_t'] - df_house_OFF['T_t']

	beta = (df_house_OFF['Delta_t']*df_house_OFF['Delta_t+1']).mean()/((df_house_OFF['Delta_t']*df_house_OFF['Delta_t']).mean())
	if use_existing_beta:
		beta = df_settings_summer['beta'].loc[house]

	#Evaluate beta estimates

	#Only cooling
	df_house_OFF_cool = df_house_OFF.loc[df_house_OFF['Delta_t'] > 0.0]
	df_house_OFF_cool = df_house_OFF_cool.loc[df_house_OFF_cool['Delta_t+1'] > 0.0]
	df_house_OFF_cool['T_t+1_est'] = beta*df_house_OFF_cool['T_t'] + (1-beta)*df_house_OFF_cool['T_out_t']
	df_house_OFF_cool['error'] = df_house_OFF_cool['T_t+1'] - df_house_OFF_cool['T_t+1_est']
	if len(df_house_OFF_cool) > 0:
		plot_error(df_house_OFF_cool,'OFF_COOL')

	#Only heating
	df_house_OFF_heat = df_house_OFF.loc[df_house_OFF['Delta_t'] < 0.0]
	df_house_OFF_heat = df_house_OFF_heat.loc[df_house_OFF_heat['Delta_t+1'] < 0.0]
	df_house_OFF_heat['T_t+1_est'] = beta*df_house_OFF_heat['T_t'] + (1-beta)*df_house_OFF_heat['T_out_t']
	df_house_OFF_heat['error'] = df_house_OFF_heat['T_t+1'] - df_house_OFF_heat['T_t+1_est']
	if len(df_house_OFF_heat) > 0:
		plot_error(df_house_OFF_heat,'OFF_HEAT')

	#Estimate P and gamma: Analysis for houses with HVAC ON

	df_house_ON = df_house.copy()
	df_house_ON = df_house_ON.loc[df_house_ON['hvac_t'] > 0.0]
	df_house_ON['Delta_t'] = df_house_ON['T_out_t'] - df_house_ON['T_t']
	df_house_ON['Delta_t+1'] = df_house_ON['T_out_t'] - df_house_ON['T_t+1']
	df_house_ON['mass-in'] = df_house_ON['Tm_t'] - df_house_ON['T_t']

	#Only cooling
	df_house_ON_cool = df_house_ON.loc[df_house_ON['Delta_t'] > 0.0]
	df_house_ON_cool = df_house_ON_cool.loc[df_house_ON_cool['Delta_t+1'] > 0.0]
	df_house_ON_cool['T_OFF_est'] = beta*df_house_ON_cool['T_t'] + (1-beta)*df_house_ON_cool['T_out_t']
	gamma_cool = ((df_house_ON_cool['T_OFF_est'] - df_house_ON_cool['T_t+1'])/((1-beta)*df_house_ON_cool['P_t'])).mean()
	df_house_ON_cool['T_t+1_est'] = beta*df_house_ON_cool['T_t'] + (1-beta)*(df_house_ON_cool['T_out_t'] - gamma_cool*df_house_ON_cool['P_t'])
	df_house_ON_cool['error'] = df_house_ON_cool['T_t+1'] - df_house_ON_cool['T_t+1_est']
	if len(df_house_ON_cool) > 0:
		P_cool = df_house_ON_cool['P_t'].mean()
		plot_error(df_house_ON_cool,'ON_COOL')
	else:
		P_cool = 0.0
		gamma_cool = 0.0

	#Only heating
	#import pdb; pdb.set_trace()
	if (df_settings['heating_system'].loc[house] == 'RESISTANCE') or (df_settings['heating_system'].loc[house] == 'HEAT_PUMP'):
		df_house_ON_heat = df_house_ON.loc[df_house_ON['Delta_t'] < 0.0]
		df_house_ON_heat = df_house_ON_heat.loc[df_house_ON_heat['Delta_t+1'] < 0.0]
		df_house_ON_heat['T_OFF_est'] = beta*df_house_ON_heat['T_t'] + (1-beta)*df_house_ON_heat['T_out_t']
		gamma_heat = -((df_house_ON_heat['T_OFF_est'] - df_house_ON_heat['T_t+1'])/((1-beta)*df_house_ON_heat['P_t'])).mean()
		df_house_ON_heat['T_t+1_est'] = beta*df_house_ON_heat['T_t'] + (1-beta)*(df_house_ON_heat['T_out_t'] + gamma_heat*df_house_ON_heat['P_t'])
		df_house_ON_heat['error'] = df_house_ON_heat['T_t+1'] - df_house_ON_heat['T_t+1_est']
		#import pdb; pdb.set_trace()
		if len(df_house_ON_heat) > 0:
			plot_error(df_house_ON_heat,'ON_HEAT')
			P_heat = df_house_ON_heat['P_t'].mean()
		else:
			P_heat = 0.0
			gamma_heat = 0.0
	else:
		P_heat = 0.0
		gamma_heat = 0.0

	df_settings['beta'].loc[house] = beta
	df_settings['gamma_heat'].loc[house] = gamma_heat
	df_settings['P_heat'].loc[house] = P_heat
	df_settings['gamma_cool'].loc[house] = gamma_cool
	df_settings['P_cool'].loc[house] = P_cool
	df_settings.to_csv(folder+'/settings/HVAC_settings_'+str(start).split(' ')[0]+'_'+str(end).split(' ')[0]+'.csv')

