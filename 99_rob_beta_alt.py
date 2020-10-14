# Estimates the distribution of beta depending on the estimation period

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as ppt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
ppt.rcParams['mathtext.fontset'] = 'stix'
ppt.rcParams['font.family'] = 'STIXGeneral'

def estimate_houseparameters(df_house_year, start, end, df_estimates):
	#print(start)
	#import pdb; pdb.set_trace()
	df_house = df_house_year.loc[start:end]
	P_heat = (df_house['HEAT']*df_house['P_t']).sum()/df_house['HEAT'].sum() # average heat P
	if np.isnan(P_heat):
		df_house['P_heat'] = 0.0 # If no heating takes place in the period of interest
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

	df_house['DeltaT_t+1'] = df_house['T_t+1'] - df_house['T_out_t']
	df_house['HEAT_P'] = df_house['HEAT']*df_house['P_heat']
	df_house['COOL_P'] = df_house['COOL']*df_house['P_cool']

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
		df_house['T_t+1_est'] = beta*df_house['T_t'] + (1-beta)*df_house['T_out_t'] + gamma_heat*df_house['HEAT_P'] + gamma_cool*df_house['COOL_P']
	except:
		df = pd.DataFrame(columns=df_estimates.columns,data=[[start,end,beta,gamma_cool,P_cool,gamma_heat,P_heat,0.0]])
		df_estimates = df_estimates.append(df)
		return df_estimates, 0.0

	R2 = r2_score(df_house['T_t+1'], df_house['T_t+1_est'])

	df = pd.DataFrame(columns=df_estimates.columns,data=[[start,end,beta,gamma_cool,P_cool,gamma_heat,P_heat,R2]])
	df_estimates = df_estimates.append(df)
	return df_estimates, R2

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

# Get estimates for year
ind = 46
use_existing_beta = False
folder = 'Diss/Diss_' + "{:04d}".format(ind) # + '_5min'
results_folder = 'Diss/Robustness' # + '_5min'
results_folder = folder + '/settings/houses'
no_house = 0

start_backup = pd.Timestamp(2016,1,1)
end_backup = pd.Timestamp(2017,1,1)
df_settings = pd.read_csv(folder+'/settings/HVAC_settings_'+str(start_backup).split(' ')[0]+'_'+str(end_backup).split(' ')[0]+'.csv',index_col=[0])

# Results
df_estimates = pd.DataFrame(columns=['start','end','beta','gamma_cool','P_cool','gamma_heat','P_heat','R2'])

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
# df_T_mass_year = pd.read_csv(folder+'/Tm_all.csv',skiprows=range(8))
# df_T_mass_year['# timestamp'] = df_T_mass_year['# timestamp'].map(lambda x: str(x)[:-4])
# df_T_mass_year['# timestamp'] = pd.to_datetime(df_T_mass_year['# timestamp'])
# df_T_mass_year.set_index('# timestamp',inplace=True)

#Temperature inside
df_T_out_year = pd.read_csv(folder+'/T_out.csv',skiprows=range(8))
df_T_out_year['# timestamp'] = df_T_out_year['# timestamp'].map(lambda x: str(x)[:-4])
df_T_out_year['# timestamp'] = pd.to_datetime(df_T_out_year['# timestamp'])
df_T_out_year.set_index('# timestamp',inplace=True)
df_T_out_year = df_T_out_year.loc[df_T_out_year.index.minute%5 == 0]
df_T_out_year.append(pd.DataFrame(index=[df_T_out_year.index[-1] + pd.Timedelta(minutes=5)],columns=['temperature'],data=df_T_out_year['temperature'].values[-1]))

# Relevant parameter estimation periods
dict_dates = [(pd.Timestamp(2016,1,1),pd.Timestamp(2017,1,1))]
for month in range(1,13):
	if month < 12:
		dict_dates += [(pd.Timestamp(2016,month,1),pd.Timestamp(2016,month+1,1))]
	else:
		dict_dates += [(pd.Timestamp(2016,month,1),pd.Timestamp(2017,1,1))]
start_week_day = pd.Timestamp(2016,1,4)
for week in range(52):
	dict_dates += [(start_week_day,start_week_day + pd.Timedelta(weeks=1))]
	start_week_day += pd.Timedelta(weeks=1)

house = df_hvac_load_year.columns[no_house]
print(df_settings['cooling_system'].loc[house])
print(df_settings['heating_system'].loc[house])
os.mkdir(results_folder +'/' + house)

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
df_house['HEAT'].loc[(df_house['T_t'] <= df_house['T_t+1']) & (df_house['hvac_t'] > 0.0)] = 1
df_house['COOL'].loc[(df_house['T_t'] > df_house['T_t+1']) & (df_house['hvac_t'] > 0.0)] = 1
df_house_year = df_house.copy()

# Start estimation
for start, end in dict_dates:
	print(str(start) + ' to ' + str(end))

	# Cut year-long ts for estimation to relevant estimation period

	df_hvac_load = df_hvac_load_year.loc[start:end] # HVAC power
	df_hvac_load_ON = df_hvac_load_ON_year.loc[start:end] # HVAC ON/OFF

	df_T_in = df_T_in_year.loc[start:end]

	#df_T_mass = df_T_mass_year.loc[start:end]

	df_T_out = df_T_out_year.loc[start:end]

	# Assemble dataset for estimation

	df_estimates, R2 = estimate_houseparameters(df_house_year,start,end,df_estimates)
	df_estimates.to_csv(results_folder +'/' + house + '/df_estimates_byweek.csv')

# Evaluate quality of different estimation periods

start_week_day = pd.Timestamp(2016,1,4) # this is Monday

df_estimates_eval = pd.DataFrame(columns=['start','end','year_R2','month_R2','week_R2'])

for week in range(52):

	#import pdb; pdb.set_trace()

	# Year: estimate R2 if year-based estimates get applied to week

	df_estimates_year = df_estimates.loc[(df_estimates['start'] == pd.Timestamp(2016,1,1)) & (df_estimates['end'] == pd.Timestamp(2017,1,1))]
	beta = df_estimates_year['beta'].iloc[0]
	gamma_heat = df_estimates_year['gamma_heat'].iloc[0]
	gamma_cool = df_estimates_year['gamma_cool'].iloc[0]
	P_heat = df_estimates_year['P_heat'].iloc[0]
	P_cool = df_estimates_year['P_cool'].iloc[0]
	#R2_year = df_estimates_year['R2'].iloc[0]

	df_house = df_house_year.loc[start_week_day:(start_week_day + pd.Timedelta(weeks=1))]
	P_heat = (df_house['HEAT']*df_house['P_t']).sum()/df_house['HEAT'].sum() # average heat P
	if np.isnan(P_heat):
		df_house['P_heat'] = 0.0 # If no heating takes place in the period of interest
	else:
		df_house['P_heat'] = P_heat
	P_cool = (df_house['COOL']*df_house['P_t']).sum()/df_house['COOL'].sum() # average cool P
	if np.isnan(P_cool):
		df_house['P_cool'] = 0.0 # If no cooling takes place in the period of interest
	else:
		df_house['P_cool'] = P_cool
	df_house['DeltaT_t+1'] = df_house['T_t+1'] - df_house['T_out_t']
	df_house['HEAT_P'] = df_house['HEAT']*df_house['P_heat']
	df_house['COOL_P'] = df_house['COOL']*df_house['P_cool']
	df_house['T_t+1_est'] = beta*df_house['T_t'] + (1-beta)*df_house['T_out_t'] + gamma_heat*df_house['HEAT_P'] + gamma_cool*df_house['COOL_P']
	R2_year = r2_score(df_house['T_t+1'], df_house['T_t+1_est'])

	# Month

	start_month = pd.Timestamp(2016,start_week_day.month,1)
	try:
		end_month = pd.Timestamp(2016,start_week_day.month+1,1)
	except:
		end_month = pd.Timestamp(2017,1,1)
	df_estimates_month = df_estimates.loc[(df_estimates['start'] == start_month) & (df_estimates['end'] == end_month)]
	beta = df_estimates_month['beta'].iloc[0]
	gamma_heat = df_estimates_month['gamma_heat'].iloc[0]
	gamma_cool = df_estimates_month['gamma_cool'].iloc[0]
	P_heat = df_estimates_month['P_heat'].iloc[0]
	P_cool = df_estimates_month['P_cool'].iloc[0]
	#R2_month = df_estimates_month['R2'].iloc[0]

	df_house = df_house_year.loc[start_week_day:(start_week_day + pd.Timedelta(weeks=1))]
	P_heat = (df_house['HEAT']*df_house['P_t']).sum()/df_house['HEAT'].sum() # average heat P
	if np.isnan(P_heat):
		df_house['P_heat'] = 0.0 # If no heating takes place in the period of interest
	else:
		df_house['P_heat'] = P_heat
	P_cool = (df_house['COOL']*df_house['P_t']).sum()/df_house['COOL'].sum() # average cool P
	if np.isnan(P_cool):
		df_house['P_cool'] = 0.0 # If no cooling takes place in the period of interest
	else:
		df_house['P_cool'] = P_cool
	df_house['DeltaT_t+1'] = df_house['T_t+1'] - df_house['T_out_t']
	df_house['HEAT_P'] = df_house['HEAT']*df_house['P_heat']
	df_house['COOL_P'] = df_house['COOL']*df_house['P_cool']
	df_house['T_t+1_est'] = beta*df_house['T_t'] + (1-beta)*df_house['T_out_t'] + gamma_heat*df_house['HEAT_P'] + gamma_cool*df_house['COOL_P']
	R2_month = r2_score(df_house['T_t+1'], df_house['T_t+1_est'])

	# Week

	df_estimates_week = df_estimates.loc[(df_estimates['start'] == start_week_day) & (df_estimates['end'] == (start_week_day + pd.Timedelta(weeks=1)))]
	try:
		beta = df_estimates_week['beta'].iloc[0]
	except:
		import pdb; pdb.set_trace()
	gamma_heat = df_estimates_week['gamma_heat'].iloc[0]
	gamma_cool = df_estimates_week['gamma_cool'].iloc[0]
	P_heat = df_estimates_week['P_heat'].iloc[0]
	P_cool = df_estimates_week['P_cool'].iloc[0]
	#R2_week = df_estimates_week['R2'].iloc[0]

	df_house = df_house_year.loc[start_week_day:(start_week_day + pd.Timedelta(weeks=1))]
	P_heat = (df_house['HEAT']*df_house['P_t']).sum()/df_house['HEAT'].sum() # average heat P
	if np.isnan(P_heat):
		df_house['P_heat'] = 0.0 # If no heating takes place in the period of interest
	else:
		df_house['P_heat'] = P_heat
	P_cool = (df_house['COOL']*df_house['P_t']).sum()/df_house['COOL'].sum() # average cool P
	if np.isnan(P_cool):
		df_house['P_cool'] = 0.0 # If no cooling takes place in the period of interest
	else:
		df_house['P_cool'] = P_cool
	df_house['DeltaT_t+1'] = df_house['T_t+1'] - df_house['T_out_t']
	df_house['HEAT_P'] = df_house['HEAT']*df_house['P_heat']
	df_house['COOL_P'] = df_house['COOL']*df_house['P_cool']
	df_house['T_t+1_est'] = beta*df_house['T_t'] + (1-beta)*df_house['T_out_t'] + gamma_heat*df_house['HEAT_P'] + gamma_cool*df_house['COOL_P']
	R2_week = r2_score(df_house['T_t+1'], df_house['T_t+1_est'])

	df = pd.DataFrame(columns=df_estimates_eval.columns,data=[[start_week_day,(start_week_day + pd.Timedelta(weeks=1)),R2_year,R2_month,R2_week]])
	df_estimates_eval = df_estimates_eval.append(df)

	# Next Week

	start_week_day += pd.Timedelta(weeks=1)

df_estimates_eval.to_csv(results_folder +'/' + house + '/df_estimates_robustness.csv')
import pdb; pdb.set_trace()
