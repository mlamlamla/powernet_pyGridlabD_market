#This file compiles the year-long savings per house

import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt
from sklearn.linear_model import LinearRegression

# Default
run = 'Diss'
ind_b = 90
folder_b = run + '/Diss_'+"{:04d}".format(ind_b) #+'_5min'

# Find relevant runs
df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])
inds = [157,159,129,160,161,162,163,164,165,166,167,168,156,169,170,171]
inds += [125,124,126,128,127]
inds += [*range(172,202)]

df_settings = df_settings.loc[inds]

##################
#
# Evaluate comfort : from 26b_housewise_welfaregains_compile.py
#
##################

for ind_WS in df_settings.index:
	
	df_HVAC = pd.read_csv(run+'/HVAC_settings/'+df_settings['settings_file'].loc[ind_WS],index_col=[0])
	start = pd.to_datetime(df_settings['start_time'].loc[ind_WS]) + pd.Timedelta(days=1)
	end = pd.to_datetime(df_settings['end_time'].loc[ind_WS])
	print(ind_WS)
	print(start)
	print(end)

	folder_WS = run + '/Diss_'+"{:04d}".format(ind_WS)

	df_welfare = pd.read_csv(folder_WS + '/df_welfare_wcost_withparameters.csv',index_col=[0])
	#import pdb; pdb.set_trace()
	#df_welfare = df_welfare[['alpha', 'fixed_u', 'fixed_cost_HVAC', 'fixed_kWh_unresp', 'fixed_cost_unresp', 'LEM_u', 'LEM_cost_HVAC', 'LEM_kWh_unresp', 'LEM_cost_unresp']]

	# Add welfare over year
	try:
		df_welfare_year += df_welfare
		df_welfare_year['heating_system'] = df_welfare['heating_system']
		df_welfare_year['cooling_system'] = df_welfare['cooling_system']
	except:
		df_welfare_year = df_welfare.copy()

df_welfare_year['alpha'] = df_welfare_year['alpha']/len(inds)
df_welfare_year['beta'] = df_welfare_year['beta']/len(inds)
df_welfare_year['P_cool'] = df_welfare_year['P_cool']/len(inds)
df_welfare_year['P_heat'] = df_welfare_year['P_heat']/len(inds)
df_welfare_year['P_mean'] = (df_welfare_year['P_cool'] + df_welfare_year['P_heat'])/2.
#df_welfare_year['P_mean'].loc[df_welfare_year['heating_system'] == 'GAS'] = df_welfare_year['P_heat'].loc[df_welfare_year['heating_system'] == 'GAS']

# Comfort change

print('Mean comfort change: '+str((df_welfare_year['LEM_u'] - df_welfare_year['fixed_u']).mean()))
print('Min comfort change: '+str((df_welfare_year['LEM_u'] - df_welfare_year['fixed_u']).min()))
print('Max comfort change: '+str((df_welfare_year['LEM_u'] - df_welfare_year['fixed_u']).max()))

max_y = 100

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
#lns = ppt.hist(df_welfare_year['LEM_u'] - df_welfare_year['fixed_u'],bins=20,color='0.75',edgecolor='0.5')
lns = ppt.hist(df_welfare_year['LEM_u'] - df_welfare_year['fixed_u'],bins=range(-35,40,5),color='0.75',edgecolor='0.5')

#ax.set_ylim(0,75)
if (df_welfare_year['LEM_u'] - df_welfare_year['fixed_u']).min() > 0.0:
	ax.set_xlim(0,df_welfare['u_change'].max()*1.05)
else:
	ax.vlines(0,0,max_y,'k',lw=1)
ax.set_xlabel('Comfort change $[USD]$')
if max_y > 0.0:
	ax.set_ylim(0,max_y)
ax.set_ylabel('Number of houses')
ppt.savefig(run+'/28_hist_comfortchange_year.png', bbox_inches='tight')
ppt.savefig(run+'/28_hist_comfortchange_year.pdf', bbox_inches='tight')

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(np.log(df_welfare['alpha']),df_welfare_year['LEM_u'] - df_welfare_year['fixed_u'],marker='x',color='0.6')
ax.set_xlabel('$log(\\alpha)$')
#ax.set_ylim(1.0,3.5)
ax.set_ylabel('Comfort change $[USD]$')
ppt.savefig(run + '/28_scatter_comfortchange_year_alpha.png', bbox_inches='tight')
ppt.savefig(run + '/28_scatter_comfortchange_year_alpha.pdf', bbox_inches='tight')

# Bill savings

df_welfare_year['fixed_cost'] = df_welfare_year['fixed_cost_HVAC'] + df_welfare_year['fixed_cost_unresp']
df_welfare_year['LEM_cost'] = df_welfare_year['LEM_cost_HVAC'] + df_welfare_year['LEM_cost_unresp']

print('Total bill savings: '+str(df_welfare_year['fixed_cost'].sum() - df_welfare_year['LEM_cost'].sum()))
print('Total bill savings wrt unresp load: '+str(df_welfare_year['fixed_cost_unresp'].sum() - df_welfare_year['LEM_cost_unresp'].sum()))
print('Total bill savings wrt HVAC: '+str(df_welfare_year['fixed_cost_HVAC'].sum() - df_welfare_year['LEM_cost_HVAC'].sum()))

print('Max bill savings: '+str((df_welfare_year['fixed_cost'] - df_welfare_year['LEM_cost']).max()))
print('Min bill savings: '+str((df_welfare_year['fixed_cost'] - df_welfare_year['LEM_cost']).min()))

# Correlation between parameters and abs savings
# Correlation between parameters and rel savings

df_welfare_year['bill_savings_total'] = df_welfare_year['fixed_cost'] - df_welfare_year['LEM_cost']

print('Correlation P_cool:' + str(df_welfare_year['bill_savings_total'].astype('float64').corr(df_welfare_year['P_cool'].astype('float64'))))
print('Correlation P_heat:' + str(df_welfare_year['bill_savings_total'].astype('float64').corr(df_welfare_year['P_heat'].astype('float64'))))
print('Correlation P_mean:' + str(df_welfare_year['bill_savings_total'].astype('float64').corr(df_welfare_year['P_mean'].astype('float64'))))
print('Correlation fixed cost:' + str(df_welfare_year['bill_savings_total'].astype('float64').corr(df_welfare_year['fixed_cost_HVAC'].astype('float64'))))

df_welfare_year['bill_savings_rel'] = df_welfare_year['bill_savings_total']/df_welfare_year['fixed_cost']*100.
print('Correlation rel savings with alpha: ' + str(df_welfare_year['bill_savings_rel'].astype('float64').corr(df_welfare_year['alpha'].astype('float64'))))
print('Correlation rel savings with log(alpha): ' + str(df_welfare_year['bill_savings_rel'].astype('float64').corr(np.log(df_welfare_year['alpha'].astype('float64')))))
print('Correlation rel savings with cooling setpoint: ' + str(df_welfare_year['bill_savings_rel'].astype('float64').corr(df_welfare_year['cooling_setpoint'].astype('float64'))))
print('Correlation rel savings with beta: ' + str(df_welfare_year['bill_savings_rel'].astype('float64').corr(df_welfare_year['beta'].astype('float64'))))

import pdb; pdb.set_trace()

df_welfare_year['surplus_change'] = df_welfare_year['LEM_u'] - df_welfare_year['fixed_u'] + df_welfare_year['bill_savings_total']
df_welfare_year.sort_values(by='surplus_change',ascending=False,inplace=True)
df_welfare_year['rank'] = range(1,len(df_welfare_year)+1)
df_welfare_year['surplus_change'].sum()

df_welfare_system_comfort = pd.read_csv(run + '/' + '26b_housewise_comfort_changes.csv',index_col=[0]) # Comfort
df_welfare_system_cost = pd.read_csv(run + '/' + '26b_housewise_cost_changes.csv',index_col=[0])
df_welfare_system = pd.read_csv(run + '/' + '26b_housewise_welfare_changes.csv',index_col=[0])
df_welfare_system[['agg_welfare_changes', 'cum_welfare_change_year', 'cum_welfare_change_year_rel', 'house_no']]
df_welfare_system['rank'] = range(1,len(df_welfare_system)+1)
df_welfare_year['rank'].corr(df_welfare_system['rank'])

# Is comfort preference of houses consistent across weeks? (or can households change from low to high preference)
# Are the households which are the most valuable to the system also those which profit the most?

