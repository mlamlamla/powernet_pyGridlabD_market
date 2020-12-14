#This file compares the WS market participation with stationary price bids to fixed price scenarie

#Constrained system for utlity: load during peak day + load duratin curve

import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt
import matplotlib.dates as mdates

ind_b = 46 #year-long simulation without LEM or constraint

#for peak-day analysis
ind_constrained = 54
start_peakday = pd.Timestamp(2016,12,19)
end_peakday = pd.Timestamp(2016,12,20)

#for load duration curve
ind_Cs = [61,59,54,56] #54 - 62 for Dec
house_no = 0

#Get start and end date
if ind_constrained in [47,63]:
	ind_unconstrained = 63 #update with +1d 63
	start = pd.Timestamp(2016,8,1)
	end = pd.Timestamp(2016,8,8)
	retail_kWh = 0.0704575196993475 #first week of august
	df_HVAC = pd.read_csv('Diss/HVAC_settings_2016-08-01_2016-08-08_ext.csv',index_col=[0])
elif ind_constrained in [49,50,51,52,53,54,55,56,57,58,59,60,61,62,64]:
	ind_unconstrained = 64 #update with +1d 64
	start = pd.Timestamp(2016,12,19)
	end = pd.Timestamp(2016,12,26)
	retail_kWh = 0.02254690804746962 #mid-Dec
	df_HVAC = pd.read_csv('Diss/HVAC_settings_2016-12-19_2016-12-26_ext.csv',index_col=[0])

retail_MWh = retail_kWh*1000.

#Data
df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])

folder_b = 'Diss/Diss_00'+str(ind_b) #+'_5min' #No LEM
df_slack_b = pd.read_csv(folder_b+'/load_node_149.csv',skiprows=range(8))
df_slack_b['# timestamp'] = df_slack_b['# timestamp'].map(lambda x: str(x)[:-4])
df_slack_b = df_slack_b.iloc[:-1]
df_slack_b['# timestamp'] = pd.to_datetime(df_slack_b['# timestamp'])
df_slack_b.set_index('# timestamp',inplace=True)
df_slack_b = df_slack_b.loc[start:end]

folder = 'Diss/Diss_00'+str(ind_unconstrained) #Prices from unconstrained LEM (== WS price)
df_prices = pd.read_csv(folder+'/df_prices.csv',index_col=[0],parse_dates=True)
df_prices_1min_b = df_prices.copy()
df_prices_1min_b = df_prices_1min_b.loc[start:end]

df_slack_bWS = pd.read_csv(folder+'/load_node_149.csv',skiprows=range(8))
df_slack_bWS['# timestamp'] = df_slack_bWS['# timestamp'].map(lambda x: str(x)[:-4])
df_slack_bWS = df_slack_bWS.iloc[:-1]
df_slack_bWS['# timestamp'] = pd.to_datetime(df_slack_bWS['# timestamp'])
df_slack_bWS.set_index('# timestamp',inplace=True)
df_slack_bWS = df_slack_bWS.loc[start:end]

folder_WS = 'Diss/Diss_00'+str(ind_constrained)
df_slack_WS = pd.read_csv(folder_WS+'/load_node_149.csv',skiprows=range(8))
df_slack_WS['# timestamp'] = df_slack_WS['# timestamp'].map(lambda x: str(x)[:-4])
df_slack_WS = df_slack_WS.iloc[:-1]
df_slack_WS['# timestamp'] = pd.to_datetime(df_slack_WS['# timestamp'])
df_slack_WS.set_index('# timestamp',inplace=True)
df_slack_WS = df_slack_WS.loc[start:end]

df_prices = pd.read_csv(folder_WS+'/df_prices.csv',index_col=[0],parse_dates=True)
df_prices_1min_LEM = df_prices.copy()
df_prices_1min_LEM = df_prices_1min_LEM.loc[start:end]

df_PV_LEM = pd.read_csv(folder_WS+'/total_P_Out.csv',skiprows=range(8))
df_PV_LEM['# timestamp'] = df_PV_LEM['# timestamp'].map(lambda x: str(x)[:-4])
df_PV_LEM['# timestamp'] = pd.to_datetime(df_PV_LEM['# timestamp'])
df_PV_LEM.set_index('# timestamp',inplace=True)
df_PV_LEM = df_PV_LEM.loc[start:end]

################
#
#Load in a week / on a peak day
#
#################

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = []

#Load
#Benchmark / no LEM
lns += ax.plot(df_slack_b['measured_real_power']/1000000,color='0.75',label='Load benchmark')
#Constrained case
lns += ax.plot(df_slack_WS['measured_real_power']/1000000,color='0.25',label='Load LEM, C = '+str(df_settings['line_capacity'].loc[ind_constrained]/1000)+' MW')
ax.hlines(df_settings['line_capacity'].loc[ind_constrained]/1000,start,end,'0.25',':')

#Price
ax2 = ax.twinx()
lns += ax2.plot(df_prices_1min_b['clearing_price'],color='0.75',ls='--',label='WS price')
lns += ax2.plot(df_prices_1min_LEM['clearing_price'],color='0.25',ls='--',label='LEM price')

ax.set_xlabel('Time')
ax.set_xlim(start_peakday,end_peakday)
ax.set_ylim(0.0,2.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_ylabel('Aggregate system load [MW]')
ax2.set_ylabel('Price [USD/MWh]')
ax2.set_ylim(0.0,150.)
labs = [l.get_label() for l in lns]
L = ax.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
ppt.savefig('Diss/aggload_p_'+str(ind_constrained)+'.png', bbox_inches='tight')
ppt.savefig('Diss/aggload_p_'+str(ind_constrained)+'.pdf', bbox_inches='tight')

################
#
#Load curves for the week: 1%
#
#################

q = 5
L_min = 1.5
L_max = 2.8

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = []

#No LEM
df_loadduration = df_slack_b.loc[start:end].sort_values('measured_real_power',ascending=False)
df_loadduration.index = range(len(df_loadduration))
df_loadduration.index = df_loadduration.index/len(df_loadduration)*100 #get percentage
lns += ax.plot(df_loadduration['measured_real_power']/1000000,color='0.5',label='No LEM')
#import pdb; pdb.set_trace()

print('No LEM: '+str(df_loadduration['measured_real_power'].iloc[0]/1000000))

#Unconstrained LEM
df_loadduration = df_slack_bWS.loc[start:end].sort_values('measured_real_power',ascending=False)
df_loadduration.index = range(len(df_loadduration))
df_loadduration.index = df_loadduration.index/len(df_loadduration)*100 #get percentage
lns += ax.plot(df_loadduration['measured_real_power']/1000000,color='0.25',label='LEM, C = $\\infty$')
#import pdb; pdb.set_trace()

print('LEM, no constraint: '+str(df_loadduration['measured_real_power'].iloc[0]/1000000))

i = 0
ls = ['-','--',':','dashdot']
cs = ['0.6','0.4','0.2','0.1']
for ind in ind_Cs:
	folder = 'Diss/Diss_00'+str(ind)
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
ppt.savefig('Diss/loadcurves_under_constraints_'+str(q)+'perc.png', bbox_inches='tight')
ppt.savefig('Diss/loadcurves_under_constraints_'+str(q)+'perc.pdf', bbox_inches='tight')
#import pdb; pdb.set_trace()

################
#
# Market income
#
#################

df_summary = pd.DataFrame(columns=['market_income'])
for ind in [57,56,55,54,58,59,60,61,62,64]:
	folder_WS = 'Diss/Diss_00'+str(ind)

	df_prices = pd.read_csv(folder_WS+'/df_prices.csv',index_col=[0],parse_dates=True)
	df_prices_1min_LEM = df_prices.copy()
	df_prices_1min_LEM = df_prices_1min_LEM.loc[start:end]

	df_PV_LEM = pd.read_csv(folder_WS+'/total_P_Out.csv',skiprows=range(8))
	df_PV_LEM['# timestamp'] = df_PV_LEM['# timestamp'].map(lambda x: str(x)[:-4])
	df_PV_LEM['# timestamp'] = pd.to_datetime(df_PV_LEM['# timestamp'])
	df_PV_LEM.set_index('# timestamp',inplace=True)
	df_PV_LEM = df_PV_LEM.loc[start:end]

	df_prices_1min_LEM['WS_supply'] = (df_prices_1min_LEM['clearing_quantity'] - df_PV_LEM.sum(axis=1)/1000.)/1000.
	df_prices_1min_LEM['WS_supply'].loc[df_prices_1min_LEM['WS_supply'] < 0.0] = 0.0
	market_income = ((df_prices_1min_LEM['clearing_price'] - df_prices_1min_b['clearing_price'])*df_prices_1min_LEM['WS_supply']/12.).sum()

	C = df_settings['line_capacity'].loc[ind]/1000
	df_summary = df_summary.append(pd.DataFrame(index=[C],columns=['market_income'],data=[[market_income]]))

df_summary_round = df_summary.round(2)
df_summary_round.sort_index(ascending=False,inplace=True)
print(df_summary_round)
table = open('Diss/df_summary_market_income.tex','w')
table.write(df_summary_round.to_latex(index=True,escape=False,na_rep=''))
table.close()

