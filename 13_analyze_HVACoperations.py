import pandas as pd
import matplotlib.pyplot as ppt
import matplotlib.dates as mdates

run = 'Diss'
folder = run + '/Diss_0090'

house_no = 0

start = pd.Timestamp(2016,7,1,6)
end = pd.Timestamp(2016,7,1,9)

df_HVAC = pd.read_csv(run + '/HVAC_settings/HVAC_settings_2016-06-27_2016-07-03_90_OLS.csv',index_col=[0])

# Total hvac load
df_hvac_load = pd.read_csv(folder+'/hvac_load_all.csv',skiprows=range(8))
df_hvac_load['# timestamp'] = df_hvac_load['# timestamp'].map(lambda x: str(x)[:-4])
df_hvac_load = df_hvac_load.iloc[:-1]
df_hvac_load['# timestamp'] = pd.to_datetime(df_hvac_load['# timestamp'])
df_hvac_load.set_index('# timestamp',inplace=True)                           
df_hvac_load = df_hvac_load.loc[start:end]

# Temperature
df_T = pd.read_csv(folder+'/T_all.csv',skiprows=range(8))
df_T['# timestamp'] = df_T['# timestamp'].map(lambda x: str(x)[:-4])
df_T = df_T.iloc[:-1]
df_T['# timestamp'] = pd.to_datetime(df_T['# timestamp'])
df_T.set_index('# timestamp',inplace=True)
df_T = df_T.loc[start:end]

house = df_T.columns[house_no]
T_cool = df_HVAC['cooling_setpoint'].loc[house]
T_heat = df_HVAC['heating_setpoint'].loc[house]
T_com = df_HVAC['comf_temperature'].loc[house]

# Plot

ppt.rc('font', size=16)
fig, [ax1, ax2] = ppt.subplots(2, 1,figsize=(8,8), sharex=True)

ppt.ioff()
#ax = fig.add_subplot(111)

lns = ax1.step(df_hvac_load.index,df_hvac_load[house],where='post',lw=3)
ax1.set_ylabel('HVAC load [kW]')
ax1.set_ylim(bottom=0.)

#ax2 = ax.twinx()
lns2 = ax2.plot(df_T[house],'r',lw=3)
ax2.set_ylabel('Internal temperature $[ ^\\circ F]$')

ax1.set_xlim(start,end)
#ax2.set_ylim(60.,80.)
ax2.hlines(T_cool,df_hvac_load.index[0],df_hvac_load.index[-1],'r','--',alpha=0.5,lw=1.)
ax2.hlines(T_heat,df_hvac_load.index[0],df_hvac_load.index[-1],'r','--',alpha=0.5,lw=1.)
ax2.hlines(T_com,df_hvac_load.index[0],df_hvac_load.index[-1],'r',alpha=0.5,lw=1.)

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ppt.savefig(run+'/13_HVAC_operations.png', bbox_inches='tight')
ppt.savefig(run+'/13_HVAC_operations.pdf', bbox_inches='tight')

import pdb; pdb.set_trace()
