#Get T offset from no market control to derive correction of T in bidding

import pandas as pd
import matplotlib.pyplot as ppt
from matplotlib.dates import DateFormatter, HourLocator

df_setpoints = pd.read_csv('Diss/Diss_0004/df_house_state.csv')

df_T = pd.read_csv('Diss/Diss_0003/T_all.csv',skiprows=range(8),nrows=43200)
df_T['# timestamp'] = df_T['# timestamp'].map(lambda x: str(x)[:-4])
df_T = df_T.iloc[:-1]
df_T['# timestamp'] = pd.to_datetime(df_T['# timestamp'])
df_T.set_index('# timestamp',inplace=True)

house = df_T.columns[0]

start = pd.Timestamp(2016,1,8)
end = pd.Timestamp(2016,1,15)

df_T_day = df_T.loc[start:(end+pd.Timedelta(days=1))]
#import pdb; pdb.set_trace()

fig = ppt.figure(figsize=(9,3),dpi=150)   
ax = fig.add_subplot(111)
lns1 = ax.plot(df_T_day[house],color='xkcd:sky blue')
ax.set_xlabel('Time')
ax.set_ylabel('T [degF]')

# ax.set_xlim(xmin=start,xmax=end+pd.Timedelta(days=1))
# ax.hlines(df_setpoints['heating_setpoint'].loc[df_setpoints['house_name'] == house].iloc[0],start,end+pd.Timedelta(days=1))
# ax.hlines(df_setpoints['cooling_setpoint'].loc[df_setpoints['house_name'] == house].iloc[0],start,end+pd.Timedelta(days=1))

ax.set_xlim(xmin=pd.Timestamp(2016,1,8,18,0),xmax=pd.Timestamp(2016,1,9,0,0))
ax.hlines(df_setpoints['heating_setpoint'].loc[df_setpoints['house_name'] == house].iloc[0],start,end+pd.Timedelta(days=1))
ax.hlines(df_setpoints['cooling_setpoint'].loc[df_setpoints['house_name'] == house].iloc[0],start,end+pd.Timedelta(days=1))

#myFmt = DateFormatter("%d")
ax.xaxis.set_major_locator(HourLocator(interval = 1))
ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

#ppt.savefig('Diss/Diss_0003/temperature_offset.pdf', bbox_inches='tight')
ppt.savefig('Diss/Diss_0003/temperature_offset_winter.png', bbox_inches='tight')
import pdb; pdb.set_trace()

df_T = pd.read_csv('Diss/Diss_0003/T_all.csv',skiprows=range(8),nrows=(43200*7))
df_T['# timestamp'] = df_T['# timestamp'].map(lambda x: str(x)[:-4])
df_T = df_T.iloc[:-1]
df_T['# timestamp'] = pd.to_datetime(df_T['# timestamp'])
df_T.set_index('# timestamp',inplace=True)

house = df_T.columns[0]

start = pd.Timestamp(2016,7,8)
end = pd.Timestamp(2016,7,15)

df_T_day = df_T.loc[start:(end+pd.Timedelta(days=1))]
#import pdb; pdb.set_trace()

fig = ppt.figure(figsize=(9,3),dpi=150)   
ax = fig.add_subplot(111)
lns1 = ax.plot(df_T_day[house],color='xkcd:sky blue')
ax.set_xlabel('Time')
ax.set_ylabel('T [degF]')
ax.set_xlim(xmin=start,xmax=end+pd.Timedelta(days=1))

ax.hlines(df_setpoints['heating_setpoint'].loc[df_setpoints['house_name'] == house].iloc[0],start,end+pd.Timedelta(days=1))
ax.hlines(df_setpoints['cooling_setpoint'].loc[df_setpoints['house_name'] == house].iloc[0],start,end+pd.Timedelta(days=1))

#ppt.savefig('Diss/Diss_0003/temperature_offset.pdf', bbox_inches='tight')
ppt.savefig('Diss/Diss_0003/temperature_offset_summer.png', bbox_inches='tight')