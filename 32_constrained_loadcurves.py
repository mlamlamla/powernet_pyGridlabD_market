#This file compares the WS market participation with stationary price bids to fixed price scenarie

#Constrained system for utlity: load during peak day + load duratin curve

import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt
import matplotlib.dates as mdates

#for peak-day analysis
ind_b = 90 #year-long simulation without LEM or constraint
ind_unconstrained = 127
ind_Cs = [130,132,134,136] #54 - 62 for Dec

# Graph settings
q = 5
L_min = 1.5
L_max = 2.8

################
#
#Load curves for the week: 1%
#
#################

df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])
start = pd.to_datetime(df_settings['start_time'].loc[ind_unconstrained]) + pd.Timedelta(days=1)
end = pd.to_datetime(df_settings['end_time'].loc[ind_unconstrained])

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = []

#No LEM

folder_b = 'Diss/Diss_' + "{:04d}".format(ind_b) #+'_5min' #No LEM
df_slack_b = pd.read_csv(folder_b + '/load_node_149.csv',skiprows=range(8))
df_slack_b['# timestamp'] = df_slack_b['# timestamp'].map(lambda x: str(x)[:-4])
df_slack_b = df_slack_b.iloc[:-1]
df_slack_b['# timestamp'] = pd.to_datetime(df_slack_b['# timestamp'])
df_slack_b.set_index('# timestamp',inplace=True)
df_slack_b = df_slack_b.loc[start:end]

df_loadduration = df_slack_b.loc[start:end].sort_values('measured_real_power',ascending=False)
df_loadduration.index = range(len(df_loadduration))
df_loadduration.index = df_loadduration.index/len(df_loadduration)*100 #get percentage
lns += ax.plot(df_loadduration['measured_real_power']/1000000,color='0.5',label='No LEM')
#import pdb; pdb.set_trace()

print('No LEM: '+str(df_loadduration['measured_real_power'].iloc[0]/1000000))

folder_WS = 'Diss/Diss_' + "{:04d}".format(ind_unconstrained)
df_slack_WS = pd.read_csv(folder_WS+'/load_node_149.csv',skiprows=range(8))
df_slack_WS['# timestamp'] = df_slack_WS['# timestamp'].map(lambda x: str(x)[:-4])
df_slack_WS = df_slack_WS.iloc[:-1]
df_slack_WS['# timestamp'] = pd.to_datetime(df_slack_WS['# timestamp'])
df_slack_WS.set_index('# timestamp',inplace=True)
df_slack_WS = df_slack_WS.loc[start:end]

#Unconstrained LEM
df_loadduration = df_slack_WS.loc[start:end].sort_values('measured_real_power',ascending=False)
df_loadduration.index = range(len(df_loadduration))
df_loadduration.index = df_loadduration.index/len(df_loadduration)*100 #get percentage
lns += ax.plot(df_loadduration['measured_real_power']/1000000,color='0.25',label='LEM, C = $\\infty$')
#import pdb; pdb.set_trace()

print('LEM, no constraint: '+str(df_loadduration['measured_real_power'].iloc[0]/1000000))

i = 0
ls = ['-','--',':','dashdot']
cs = ['0.6','0.4','0.2','0.1']
for ind in ind_Cs:
	folder = 'Diss/Diss_'+"{:04d}".format(ind)
	df_slack = pd.read_csv(folder+'/load_node_149.csv',skiprows=range(8))
	df_slack['# timestamp'] = df_slack['# timestamp'].map(lambda x: str(x)[:-4])
	df_slack = df_slack.iloc[:-1]
	df_slack['# timestamp'] = pd.to_datetime(df_slack['# timestamp'])
	df_slack.set_index('# timestamp',inplace=True)
	df_slack = df_slack.loc[start:end]
	df_loadduration = df_slack.sort_values('measured_real_power',ascending=False)
	df_loadduration.index = range(len(df_loadduration))
	df_loadduration.index = df_loadduration.index/len(df_loadduration)*100 #get percentage
	lns += ax.plot(df_loadduration['measured_real_power']/1000000,color=cs[i],ls='--',label='LEM, C = '+str(df_settings['line_capacity'].loc[ind]/1000)+' MW')
	print('LEM, constraint '+str(df_settings['line_capacity'].loc[ind]/1000)+', '+str(df_loadduration['measured_real_power'].iloc[0]/1000000))
	i += 1

ax.set_xlabel('Quantile [%]')
ax.set_xlim(0.0,q)
ax.set_ylim(L_min,L_max)
ax.set_ylabel('Aggregate system load [MW]')
labs = [l.get_label() for l in lns]
L = ax.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
ppt.savefig('Diss/32_loadcurves_under_constraints_'+str(q)+'perc.png', bbox_inches='tight')
ppt.savefig('Diss/32_loadcurves_under_constraints_'+str(q)+'perc.pdf', bbox_inches='tight')