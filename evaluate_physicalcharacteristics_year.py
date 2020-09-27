#Evaluates the physical parameters of the system

import pandas as pd
import numpy as np
import matplotlib.pyplot as ppt
ppt.rcParams['mathtext.fontset'] = 'stix'
ppt.rcParams['font.family'] = 'STIXGeneral'

ind = 35
folder = 'Diss/Diss_' + "{:04d}".format(ind)
N = 1000000

#Total hvac load
df_hvac_load = pd.read_csv(folder+'/hvac_load_all.csv',skiprows=range(8),nrows=N)
df_hvac_load['# timestamp'] = df_hvac_load['# timestamp'].map(lambda x: str(x)[:-4])
df_hvac_load['# timestamp'] = pd.to_datetime(df_hvac_load['# timestamp'])
df_hvac_load.set_index('# timestamp',inplace=True)

df_hvac_load_ON = df_hvac_load.copy()

#Temperature inside
df_T_in = pd.read_csv(folder+'/T_all.csv',skiprows=range(8),nrows=N)
df_T_in['# timestamp'] = df_T_in['# timestamp'].map(lambda x: str(x)[:-4])
df_T_in['# timestamp'] = pd.to_datetime(df_T_in['# timestamp'])
df_T_in.set_index('# timestamp',inplace=True)

#Temperature inside
df_T_mass = pd.read_csv(folder+'/Tm_all.csv',skiprows=range(8),nrows=N)
df_T_mass['# timestamp'] = df_T_mass['# timestamp'].map(lambda x: str(x)[:-4])
df_T_mass['# timestamp'] = pd.to_datetime(df_T_mass['# timestamp'])
df_T_mass.set_index('# timestamp',inplace=True)

#Temperature inside
df_T_out = pd.read_csv(folder+'/T_outdoor.csv',skiprows=range(8),nrows=N)
df_T_out['# timestamp'] = df_T_out['# timestamp'].map(lambda x: str(x)[:-4])
df_T_out['# timestamp'] = pd.to_datetime(df_T_out['# timestamp'])
df_T_out.set_index('# timestamp',inplace=True)

cols = []
for month in range(1,13):
	cols += ['beta_'+str(month),'gamma_'+str(month),'P_'+str(month)]
df_settings = pd.DataFrame(index=df_hvac_load.columns,columns=cols)

#Compile dataset
for house in df_hvac_load.columns:
	df_hvac_load_ON[house].loc[df_hvac_load_ON[house] > 0] = 1.

	#import pdb; pdb.set_trace()

	df_house = pd.DataFrame(index=df_hvac_load.index,columns=['T_t','Tm_t','T_out_t','hvac_t','P_t','T_t+1'],data=0.0)
	df_house['T_t'] = df_T_in[house].values
	df_house['Tm_t'] = df_T_mass[house].values
	try:
		df_house['T_out_t'] = df_T_out.values.reshape(len(df_T_out),) #Not end of the period
	except:
		df_house['T_out_t'].iloc[:-1] = df_T_out.values.reshape(len(df_T_out),)
	df_house['hvac_t'] = df_hvac_load_ON[house].values
	df_house['P_t'] = df_hvac_load[house].values
	df_house['T_t+1'].iloc[:-1] = df_house['T_t'].iloc[1:].values
	df_house = df_house.iloc[:-1]

	#Analysis for houses with HVAC OFF

	df_house_OFF = df_house.copy()
	df_house_OFF = df_house_OFF.loc[df_house_OFF['hvac_t'] == 0.0]

	df_house_OFF['Delta_t'] = df_house_OFF['T_out_t'] - df_house_OFF['T_t']
	df_house_OFF['Delta_t+1'] = df_house_OFF['T_out_t'] - df_house_OFF['T_t+1']
	df_house_OFF['mass-in'] = df_house_OFF['Tm_t'] - df_house_OFF['T_t']
	df_house_OFF['T_t+1_est'] = 0.0
	df_house_OFF['error'] = 0.0

	#Only cooling
	df_house_OFF = df_house_OFF.loc[df_house_OFF['Delta_t'] < 0.0]
	df_house_OFF = df_house_OFF.loc[df_house_OFF['Delta_t+1'] < 0.0]

	#Calculate beta
	betas = []
	for month in range(1,13):
		index_month = df_house_OFF.loc[df_house_OFF.index.month == month].index
		beta = (df_house_OFF['Delta_t'].loc[index_month]*df_house_OFF['Delta_t+1'].loc[index_month]).mean()/((df_house_OFF['Delta_t'].loc[index_month]*df_house_OFF['Delta_t'].loc[index_month]).mean())
		betas += [beta]
		print(str(month)+': '+str(beta))

		#Estimate T
		#import pdb; pdb.set_trace()
		df_house_OFF['T_t+1_est'].loc[index_month] = beta*df_house_OFF['T_t'].loc[index_month] + (1-beta)*df_house_OFF['T_out_t'].loc[index_month]
		df_house_OFF['error'].loc[index_month] = df_house_OFF['T_t+1'].loc[index_month] - df_house_OFF['T_t+1_est'].loc[index_month]
		df_settings['beta_'+str(month)].loc[house] = beta

	#Plot
	# fig = ppt.figure(figsize=(4,4),dpi=150)   
	# ax = fig.add_subplot(111)
	# lns1 = ax.scatter(df_house_OFF['T_t+1'],df_house_OFF['T_t+1_est'],c='0.5',marker='x')
	# #lns3 = ax.scatter(df_house_OFF['T_t+1'],df_house_OFF['T_t'],c='r')
	# min_T = df_house_OFF['T_t+1'].min()
	# max_T = df_house_OFF['T_t+1'].max()
	# lns2 = ax.plot(np.arange(min_T,max_T,0.01),np.arange(min_T,max_T,0.01),'k')
	# ax.set_xlabel('Actual temperature $\\theta_t$')
	# ax.set_ylabel('Estimated temperature $\\widehat{\\theta}_t$')
	# ppt.savefig(folder+'/settings/'+house+'_temperature_estimation_OFF.pdf', bbox_inches='tight')
	# ppt.close()

	#Analysis for houses with HVAC ON

	df_house_ON = df_house.copy()
	df_house_ON = df_house_ON.loc[df_house_ON['hvac_t'] > 0.0]
	df_house_ON['T_t+1_est'] = 0.0
	df_house_ON['error'] = 0.0
	df_house_ON['T_OFF_est'] = 0.0
	df_house_ON['Delta_t'] = df_house_ON['T_out_t'] - df_house_ON['T_t']
	df_house_ON['Delta_t+1'] = df_house_ON['T_out_t'] - df_house_ON['T_t+1']

	#Only cooling
	df_house_ON = df_house_ON.loc[df_house_ON['Delta_t'] < 0.0]
	df_house_ON = df_house_ON.loc[df_house_ON['Delta_t+1'] < 0.0]

	gammas = []
	for month in range(1,13):
		index_month = df_house_ON.loc[df_house_ON.index.month == month].index
		df_house_ON['T_OFF_est'].loc[index_month] = betas[month-1]*df_house_ON['T_t'].loc[index_month] + (1-betas[month-1])*df_house_ON['T_out_t'].loc[index_month]
		P = (df_house_ON['P_t'].loc[index_month]).mean()
		gamma = ((df_house_ON['T_OFF_est'].loc[index_month]).mean() - (df_house_ON['T_t+1'].loc[index_month]).mean())/((1-betas[month-1])*P)
		gammas += [gamma]

		#Estimate T
		df_house_ON['T_t+1_est'].loc[index_month] = betas[month-1]*df_house_ON['T_t'].loc[index_month] + (1-betas[month-1])*(df_house_ON['T_out_t'].loc[index_month] - gamma*P)
		df_house_ON['error'].loc[index_month] = df_house_ON['T_t+1'].loc[index_month] - df_house_ON['T_t+1_est'].loc[index_month]
		df_settings['gamma_'+str(month)].loc[house] = gamma
		df_settings['P_'+str(month)].loc[house] = P

	#Plot
	# fig = ppt.figure(figsize=(4,4),dpi=150)   
	# ax = fig.add_subplot(111)
	# lns1 = ax.scatter(df_house_ON['T_t+1'],df_house_ON['T_t+1_est'],c='0.5',marker='x')
	# #lns3 = ax.scatter(df_house_ON['T_t+1'],df_house_ON['T_t'],c='r')

	# min_T = df_house_ON['T_t+1'].min()
	# max_T = df_house_ON['T_t+1'].max()
	# lns2 = ax.plot(np.arange(min_T,max_T,0.01),np.arange(min_T,max_T,0.01),'k')
	# ax.set_xlabel('Actual temperature $\\theta_t$')
	# ax.set_ylabel('Estimated temperature $\\widehat{\\theta}_t$')
	# ppt.savefig(folder+'/settings/'+house+'_temperature_estimation_ON.pdf', bbox_inches='tight')
	# ppt.close()

	df_settings.to_csv(folder+'/settings/HVAC_settings_heating.csv')

