import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt
import matplotlib.dates as mdates

run = 'Diss'
ind_b = 90 #year-long simulation without LEM or constraint
df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])

ind_constrained = 134

city = 'Austin'
folder_b = run + '/Diss_'+"{:04d}".format(ind_b) #+'_5min'
folder_WS = run + '/Diss_'+"{:04d}".format(ind_constrained)

# Calculate RR in benchmark case

folder = folder_b
market_file = df_settings['market_data'].loc[ind_constrained]
market_file = 'Ercot_LZ_SOUTH.csv'

# Unconstrained

df_slack = pd.read_csv(folder+'/load_node_149.csv',skiprows=range(8))
df_slack['# timestamp'] = df_slack['# timestamp'].map(lambda x: str(x)[:-4])
df_slack = df_slack.iloc[:-1]
df_slack['# timestamp'] = pd.to_datetime(df_slack['# timestamp'])
df_slack.set_index('# timestamp',inplace=True)
peak_day = df_slack.loc[df_slack['measured_real_power'] == df_slack['measured_real_power'].max()].index[0]
print('The peak day of the year is '+str(peak_day))
start = pd.Timestamp(peak_day.year,peak_day.month,peak_day.day)
end = pd.Timestamp(peak_day.year,peak_day.month,peak_day.day + 1)
df_slack = df_slack.loc[start:end]
df_slack = df_slack/1000 #kW

df_WS = pd.read_csv('glm_generation_'+city+'/'+market_file,parse_dates=[0])
df_WS.rename(columns={'Unnamed: 0':'timestamp'},inplace=True)
df_WS.set_index('timestamp',inplace=True)
df_WS = df_WS.loc[start:end]

df_WS['system_load_b'] = df_slack['measured_real_power']

# Constrained

df_slack = pd.read_csv(folder_WS+'/load_node_149.csv',skiprows=range(8))
df_slack['# timestamp'] = df_slack['# timestamp'].map(lambda x: str(x)[:-4])
df_slack = df_slack.iloc[:-1]
df_slack['# timestamp'] = pd.to_datetime(df_slack['# timestamp'])
df_slack.set_index('# timestamp',inplace=True)
df_slack = df_slack.loc[start:end]
df_slack = df_slack/1000 #kW

df_prices = pd.read_csv(folder_WS + '/df_prices.csv',index_col=[0],parse_dates=True).loc[start:end]

df_WS['system_load_WS'] = df_slack['measured_real_power']
df_WS['LEM_prices'] = df_prices['clearing_price']

import pdb; pdb.set_trace()

###############
#
# Specific day
#
#################

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = []

#Load
#Benchmark / no LEM
lns += ax.plot(df_WS['system_load_b']/1000.,color='0.75',label='Load benchmark')
#Constrained case
lns += ax.plot(df_WS['system_load_WS']/1000.,color='0.25',label='Load LEM, C = '+str(df_settings['line_capacity'].loc[ind_constrained]/1000)+' MW')
ax.hlines(df_settings['line_capacity'].loc[ind_constrained]/1000,start,end,'0.25',':')

#Price
ax2 = ax.twinx()
lns += ax2.plot(df_WS['RT'],color='0.75',ls='--',label='WS price')
lns += ax2.plot(df_WS['LEM_prices'],color='0.25',ls='--',label='LEM price')

ax.set_xlabel('Time')
ax.set_xlim(start,end)
ax.set_ylim(0.0,2.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_ylabel('Aggregate system load [MW]')
ax2.set_ylabel('Price [USD/MWh]')
ax2.set_ylim(0.0,150.)
labs = [l.get_label() for l in lns]
L = ax.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
ppt.savefig('Diss/Diss_'+"{:04d}".format(ind_constrained)+'/aggload_p_'+str(ind_constrained)+'.png', bbox_inches='tight')
ppt.savefig('Diss/Diss_'+"{:04d}".format(ind_constrained)+'/aggload_p_'+str(ind_constrained)+'.pdf', bbox_inches='tight')

import pdb; pdb.set_trace()