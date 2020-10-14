# Analyzes physical and utility parameters during different weeks

import pandas as pd
import matplotlib.pyplot as ppt

house_no = 0
ind_base = 90

start = pd.Timestamp(2016,1,4)
end = pd.Timestamp(2016,1,11)
results_folder = 'Diss'
df_distribution = pd.DataFrame()
R2_all = []

while True:
	print(start)
	if end.year == 2017:
		break

	df_settings_ext = pd.read_csv(results_folder +'/HVAC_settings/HVAC_settings_' +str(start).split(' ')[0]+'_'+str(end).split(' ')[0]+'_'+str(ind_base)+'_OLS.csv',index_col=[0])
	
	if len(df_distribution) == 0:
		df_distribution = pd.DataFrame(index=[start],columns=df_settings_ext.columns,data=[df_settings_ext.iloc[house_no].values])
	else:
		df = pd.DataFrame(index=[start],columns=df_settings_ext.columns,data=[df_settings_ext.iloc[house_no].values])
		df_distribution = df_distribution.append(df)
	import pdb; pdb.set_trace()
	R2_all += df_settings_ext['R2'].tolist()

	start += pd.Timedelta(weeks=1)
	end += pd.Timedelta(weeks=1)

#import pdb; pdb.set_trace()
max_y = 20

#Histogram beta
fig = ppt.figure(figsize=(8,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.hist(df_distribution['beta'],bins=20,color='0.75',edgecolor='0.5')
#ax.vlines(0,0,max_y)
ax.set_ylim(0,max_y)
#ax.set_xlabel('Insulation $\beta$')
ax.set_ylabel('Number of weeks')
ppt.savefig(results_folder +'/HVAC_settings/hist_beta_'+df_settings_ext.index[house_no]+'.png', bbox_inches='tight')
ppt.close()

#Histogram gamma heat
fig = ppt.figure(figsize=(8,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.hist(df_distribution['gamma_heat'],bins=20,color='0.75',edgecolor='0.5')
#ax.vlines(0,0,max_y)
ax.set_ylim(0,max_y)
#ax.set_xlabel('Heating efficiency $$\gamma_heat$$')
ax.set_ylabel('Number of weeks')
ppt.savefig(results_folder +'/HVAC_settings/hist_gamma_heat_'+df_settings_ext.index[house_no]+'.png', bbox_inches='tight')

#Histogram gamma cool
fig = ppt.figure(figsize=(8,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.hist(df_distribution['gamma_cool'],bins=20,color='0.75',edgecolor='0.5')
#ax.vlines(0,0,max_y)
ax.set_ylim(0,max_y)
#ax.set_xlabel('Cooling efficiency $$\gamma_cool$$')
ax.set_ylabel('Number of weeks')
ppt.savefig(results_folder +'/HVAC_settings/hist_gamma_cool_'+df_settings_ext.index[house_no]+'.png', bbox_inches='tight')

#Histogram alpha
fig = ppt.figure(figsize=(8,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.hist(df_distribution['alpha'],bins=20,color='0.75',edgecolor='0.5')
#ax.vlines(0,0,max_y)
ax.set_ylim(0,max_y)
#ax.set_xlabel('Utility $$\alpha$$')
ax.set_ylabel('Number of weeks')
ppt.savefig(results_folder +'/HVAC_settings/hist_alpha_'+df_settings_ext.index[house_no]+'.png', bbox_inches='tight')

#Histogram R2
fig = ppt.figure(figsize=(8,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.hist(df_distribution['R2'],bins=20,color='0.75',edgecolor='0.5')
#ax.vlines(0,0,max_y)
ax.set_ylim(0,max_y)
#ax.set_xlabel('Utility $$\alpha$$')
ax.set_ylabel('Number of weeks')
ppt.savefig(results_folder +'/HVAC_settings/hist_R2_'+df_settings_ext.index[house_no]+'.png', bbox_inches='tight')

#Histogram R2 all
fig = ppt.figure(figsize=(8,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.hist(R2_all,bins=20,color='0.75',edgecolor='0.5')
#ax.vlines(0,0,max_y)
ax.set_ylim(0,max_y)
#ax.set_xlabel('Utility $$\alpha$$')
ax.set_ylabel('Number of weeks')
ppt.savefig(results_folder +'/HVAC_settings/hist_R2_all.png', bbox_inches='tight')