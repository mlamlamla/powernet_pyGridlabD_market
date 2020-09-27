#Calculate duty cycle HVAC by time

import pandas as pd
import matplotlib.pyplot as ppt

#Total hvac load
df_hvac_load = pd.read_csv('Diss/Diss_0003/hvac_load_all.csv',skiprows=range(8))
df_hvac_load['# timestamp'] = df_hvac_load['# timestamp'].map(lambda x: str(x)[:-4])
df_hvac_load = df_hvac_load.iloc[:-1]
df_hvac_load['# timestamp'] = pd.to_datetime(df_hvac_load['# timestamp'])
df_hvac_load.set_index('# timestamp',inplace=True)                           

import pdb; pdb.set_trace()
df_hvac_load[df_hvac_load > 0.0] = 1.0

df_pivot = pd.DataFrame(index=df.groupby([df.index.month,df.index.hour]).mean().mean(axis=1).index,columns=['duty_cycle'],data=df.groupby([df.index.month,df.index.hour]).mean().mean(axis=1).values)
df_pivot.index.set_names(['month', 'hour'],inplace=True)
df_pivot.reset_index(inplace=True)
df_pivot = df_pivot.pivot(index='month',columns='hour',values='duty_cycle')
df_pivot.to_csv('glm_generation_Austin/HVAC_dutycycle.csv')