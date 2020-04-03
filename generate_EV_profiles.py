#Generates EV input file from Chargepoint data
#Marie-Louise Arlt

import os
import pandas as pd
import numpy as np
import datetime

#Target file
#df_EV = pd.read_csv('~/Documents/powernet/powernet_markets_mysql/EV_events_2016.csv',index_col=[0])
#print(df_EV.iloc[:5])

#EV names
df_EV_names = pd.read_csv('~/Documents/powernet/powernet_markets_mysql/glm_generation_Austin/EV_placement.csv',index_col=[0])
for ind in df_EV_names.index:
	df_EV_names.at[ind,'name'] = 'EV_B1_N'+str(df_EV_names.EV_node_num[ind])+'_'+str('{:04}'.format(df_EV_names.EV_house_index[ind]))
print(df_EV_names.iloc[:5])

#Get files in charging profiles which comply with requirements
subfolders = [f.path for f in os.scandir(os.getcwd()) if f.is_dir() ]
#subfolders = [subfolders[0]]
included_profiles = []

f = 0
for subfolder in subfolders:
	list_files = os.listdir(subfolder) # dir is your directory path
	included_profiles += [[]]
	for file in list_files:
		try:
			df = pd.read_csv(subfolder +'/' + file,parse_dates=['Interval Start Time (Local)'])
			delta = df['Interval Start Time (Local)'].iloc[-1] - df['Interval Start Time (Local)'].iloc[0]
			if delta > pd.Timedelta('8 days'):
				included_profiles[f] += [file]
		except:
			pass
			#print(file)
			#print(df.columns)
	f += 1

#Write to dict
events = dict()
f = 0
key = 0
energy_charged_kWh_max = 0.0
rate_kW_max = 0.0
l = 0

df_sessions = pd.DataFrame(columns=['sum_charge','rate_charge'])

for subfolder in subfolders:
	for file in included_profiles[f]:
		df = pd.read_csv(subfolder +'/' + file,parse_dates=['Interval Start Time (Local)'])
		prev_session_id = 0
		prev_time = df['Interval Start Time (Local)'].iloc[0]
		events[key] = []
		av_rate_list = []
		for ind in df.index:
			session_id = int(df['Session ID'].loc[ind])
			if session_id != prev_session_id: #New session started
				#Finish up old event
				try:
					end_time = prev_time
					events[key] += [(start_time,end_time,energy_charged_kWh,sum(av_rate_list)/len(av_rate_list))]
					df_sessions.at[l,'sum_charge'] = energy_charged_kWh
					df_sessions.at[l,'rate_charge'] = round(sum(av_rate_list)/len(av_rate_list),1)
					l += 1
					if energy_charged_kWh > energy_charged_kWh_max:
						energy_charged_kWh_max = energy_charged_kWh
				except:
					pass #for the first time step
				prev_session_id = session_id

				#Read in new event
				start_time = df['Interval Start Time (Local)'].loc[ind]
				prev_time = df['Interval Start Time (Local)'].loc[ind]
				energy_charged_kWh = df['Interval Energy'].loc[ind]
				av_rate_kW = float(df['Average Power'].loc[ind])
				if av_rate_kW > rate_kW_max:
					rate_kW_max = av_rate_kW
				av_rate_list += [av_rate_kW]
			elif session_id == prev_session_id:
				energy_charged_kWh += df['Interval Energy'].loc[ind]
				interval = pd.Timedelta(str(df['Interval Duration (Secs)'].loc[ind])+' seconds')

				prev_time = df['Interval Start Time (Local)'].loc[ind] + interval
		key += 1
	f += 1
print('Max charge '+str(energy_charged_kWh_max))
print('Max rate '+str(rate_kW_max))
max_sto_vol = energy_charged_kWh_max/0.8 #Standard battery size

df_sessions.sort_values(by='rate_charge',inplace=True)
df_sessions['rate_charge']
print(df_sessions)

#New table
cols = []
for ind in df_EV_names.index:
	cols += [df_EV_names['name'].loc[ind], df_EV_names['name'].loc[ind]+'_u', df_EV_names['name'].loc[ind]+'_SOC']
df_events = pd.DataFrame(index=range(1000),columns = cols)

no_profiles = 0
for profiles in included_profiles:
	no_profiles += len(profiles)
needed_profiles = len(df_EV_names.index)

#RELATE TO ARBITRARY DATE, eg 07/01/2016
start_date_ALL = datetime.date(2016, 6, 30) # minus one day to start with 07/01
start_date_TOTAL = pd.Timestamp(2016, 7, 1) #datetime.date(2016, 7, 1)
end_date_TOTAL = pd.Timestamp(2016, 7, 8) #datetime.date(2016, 7, 8)

for EV in df_EV_names['name']:
	generate_events = True
	while generate_events:
		#Randomly pick key
		key = np.random.choice(range(no_profiles),p=[1./no_profiles]*no_profiles)
		event_list = events[key]
		#Initial event in this list
		start_time0, end_time0, __, __ = event_list[0]
		delta_date = start_date_ALL - datetime.date(start_time0.year, start_time0.month, start_time0.day)
		start_time0_ref = start_time0 + delta_date
		end_time0_ref = end_time0 + delta_date
		l = 0
		for event in event_list:
			#Next event in this list
			start_time,end_time,energy_charged_kWh,av_power = event
			start_time_ref = start_time + delta_date
			end_time_ref = end_time + delta_date
			#import pdb; pdb.set_trace()
			if start_time_ref > end_date_TOTAL: #Skip days after 7/7
				if l == 0:
					print(l) #Re-generate event line with new randomly drawn profile
				else:
					generate_events = False
				break
			if end_time_ref > start_date_TOTAL: #Skip first day
				if l == 0 and start_time_ref < start_date_TOTAL: #Overnight charging on first day
					SOC_0 = 0.5 #better estimate?
					start_time_ref = start_date_TOTAL
				else:

					#CLUSTER ACC TO BATTERY TYPES
					SOC_0 = (max_sto_vol - energy_charged_kWh)/max_sto_vol
					SOC_0 = np.random.uniform(0.2,1.0)
					
				if energy_charged_kWh > 0.1: #Skip events with communication mistakes
					df_events.at[l,EV] = start_time_ref
					df_events.at[l,EV+'_u'] = av_power
					df_events.at[l,EV+'_SOC'] = SOC_0
					df_events.at[l+1,EV] = end_time_ref
					l += 2
		
print(df_events.iloc[:10])
df_events = df_events.dropna(axis=0,how='all')
df_events.to_csv('EV_events.csv')
