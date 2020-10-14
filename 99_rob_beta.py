# Estimates the distribution of beta depending on the estimation period

import os
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
	if not os.path.isdir(results_folder + '/' + house):
		os.mkdir(results_folder + '/' + house)
	ppt.savefig(results_folder + '/' + house + '/ParEst_'+name+'_'+str(start)+'_'+str(end)+'.pdf', bbox_inches='tight')
	ppt.close()
	# Calculate R2
	import pdb; pdb.set_trace()
	SS_tot = ((df_house_OFF['T_t+1'] - df_house_OFF['T_t+1'].mean()).pow(2)).sum()
	SS_res = ((df_house_OFF['T_t+1'] - df_house_OFF['T_t+1_est']).pow(2)).sum()
	R2 = 1. - SS_res/SS_tot
	return R2

# Get estimates for year
ind = 46
use_existing_beta = False
folder = 'Diss/Diss_' + "{:04d}".format(ind) # + '_5min'
results_folder = 'Diss/Robustness' # + '_5min'
no_house = 0

# Results
df_estimates = pd.DataFrame(columns=['start','end','beta','gamma_cool','P_cool','gamma_heat','P_heat','R2_OFF_COOL','R2_OFF_HEAT','R2_ON_COOL','R2_ON_HEAT'])

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

house = df_hvac_load_year.columns[no_house]

start_backup = pd.Timestamp(2016,1,1)
end_backup = pd.Timestamp(2017,1,1)
df_settings = pd.read_csv(folder+'/settings/HVAC_settings_'+str(start_backup).split(' ')[0]+'_'+str(end_backup).split(' ')[0]+'.csv',index_col=[0])
print(df_settings['cooling_system'].loc[house])
print(df_settings['heating_system'].loc[house])

#Temperature inside
df_T_in_year = pd.read_csv(folder+'/T_all.csv',skiprows=range(8))
df_T_in_year['# timestamp'] = df_T_in_year['# timestamp'].map(lambda x: str(x)[:-4])
df_T_in_year['# timestamp'] = pd.to_datetime(df_T_in_year['# timestamp'])
df_T_in_year.set_index('# timestamp',inplace=True)

#Temperature inside
df_T_mass_year = pd.read_csv(folder+'/Tm_all.csv',skiprows=range(8))
df_T_mass_year['# timestamp'] = df_T_mass_year['# timestamp'].map(lambda x: str(x)[:-4])
df_T_mass_year['# timestamp'] = pd.to_datetime(df_T_mass_year['# timestamp'])
df_T_mass_year.set_index('# timestamp',inplace=True)

#Temperature inside
df_T_out_year = pd.read_csv(folder+'/T_out.csv',skiprows=range(8))
df_T_out_year['# timestamp'] = df_T_out_year['# timestamp'].map(lambda x: str(x)[:-4])
df_T_out_year['# timestamp'] = pd.to_datetime(df_T_out_year['# timestamp'])
df_T_out_year.set_index('# timestamp',inplace=True)

# Relevant parameter estimation periods
dict_dates = [(pd.Timestamp(2016,1,1),pd.Timestamp(2017,1,1))]
for month in range(1,12):
	dict_dates += [(pd.Timestamp(2016,month,1),pd.Timestamp(2016,month+1,1))]
start_week_day = pd.Timestamp(2016,1,1)
for week in range(52):
	dict_dates += [(start_week_day,start_week_day + pd.Timedelta(weeks=1))]
	start_week_day += pd.Timedelta(weeks=1)

# Start estimation
for start, end in dict_dates:

	# Cut year-long ts for estimation to relevant estimation period

	df_hvac_load = df_hvac_load_year.loc[start:end]
	df_hvac_load_ON = df_hvac_load.copy()

	df_T_in = df_T_in_year.loc[start:end]

	df_T_mass = df_T_mass_year.loc[start:end]

	df_T_out = df_T_out_year.loc[start:end]
	df_T_out = df_T_out.loc[df_T_out.index.minute%5 == 0]
	df_T_out.append(pd.DataFrame(index=[df_T_out.index[-1] + pd.Timedelta(minutes=5)],columns=['temperature'],data=df_T_out['temperature'].values[-1]))

	# Assemble temperature ts

	df_hvac_load_ON[house].loc[df_hvac_load_ON[house] > 0] = 1.

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

	#Estimate beta - Analysis for houses with HVAC OFF

	df_house_OFF = df_house.copy()
	df_house_OFF = df_house_OFF.loc[df_house_OFF['hvac_t'] == 0.0]
	df_house_OFF['Delta_t'] = df_house_OFF['T_out_t'] - df_house_OFF['T_t']
	df_house_OFF['Delta_t+1'] = df_house_OFF['T_out_t'] - df_house_OFF['T_t+1']
	df_house_OFF['mass-in'] = df_house_OFF['Tm_t'] - df_house_OFF['T_t']

	beta = (df_house_OFF['Delta_t']*df_house_OFF['Delta_t+1']).mean()/((df_house_OFF['Delta_t']*df_house_OFF['Delta_t']).mean())

	#Evaluate beta estimates

	#Only cooling
	df_house_OFF_cool = df_house_OFF.loc[df_house_OFF['Delta_t'] > 0.0]
	df_house_OFF_cool = df_house_OFF_cool.loc[df_house_OFF_cool['Delta_t+1'] > 0.0]
	df_house_OFF_cool['T_t+1_est'] = beta*df_house_OFF_cool['T_t'] + (1-beta)*df_house_OFF_cool['T_out_t']
	df_house_OFF_cool['error'] = df_house_OFF_cool['T_t+1'] - df_house_OFF_cool['T_t+1_est']
	if len(df_house_OFF_cool) > 0:
		R2_OFF_COOL = plot_error(df_house_OFF_cool,'OFF_COOL')

	#Only heating
	df_house_OFF_heat = df_house_OFF.loc[df_house_OFF['Delta_t'] < 0.0]
	df_house_OFF_heat = df_house_OFF_heat.loc[df_house_OFF_heat['Delta_t+1'] < 0.0]
	df_house_OFF_heat['T_t+1_est'] = beta*df_house_OFF_heat['T_t'] + (1-beta)*df_house_OFF_heat['T_out_t']
	df_house_OFF_heat['error'] = df_house_OFF_heat['T_t+1'] - df_house_OFF_heat['T_t+1_est']
	if len(df_house_OFF_heat) > 0:
		R2_OFF_HEAT = plot_error(df_house_OFF_heat,'OFF_HEAT')

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
		R2_ON_COOL = plot_error(df_house_ON_cool,'ON_COOL')
	else:
		P_cool = 0.0
		gamma_cool = 0.0
		R2_ON_COOL = 0.0

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
			R2_ON_HEAT = plot_error(df_house_ON_heat,'ON_HEAT')
			P_heat = df_house_ON_heat['P_t'].mean()
		else:
			P_heat = 0.0
			gamma_heat = 0.0
			R2_ON_HEAT = 0.0
	else:
		P_heat = 0.0
		gamma_heat = 0.0
		R2_ON_HEAT = 0.0

	df = pd.DataFrame(columns=df_estimates.columns,data=[[start,end,beta,gamma_cool,P_cool,gamma_heat,P_heat,R2_OFF_COOL,R2_OFF_HEAT,R2_ON_COOL,R2_ON_HEAT]])
	df_estimates = df_estimates.append(df)

df_estimates.to_csv(results_folder +'/' + house + '/df_estimates.csv')