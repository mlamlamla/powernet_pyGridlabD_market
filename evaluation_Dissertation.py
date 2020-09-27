#This file compares the WS market participation with stationary price bids to fixed price scenarie

#Unconstrained LEM: Impact on customers + dependence on alpha

import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt

ind_b = 46
ind_WS = 63

if ind_WS in [47,63]:
	start = pd.Timestamp(2016,8,1)
	end = pd.Timestamp(2016,8,8)
	retail_kWh = 0.0704575196993475 #first week of august
	df_HVAC = pd.read_csv('Diss/HVAC_settings_2016-08-01_2016-08-08_ext.csv',index_col=[0])
elif ind_WS in [49,50,51,52,53,54,55,56,57,58,59,60,61,62,64]:
	start = pd.Timestamp(2016,12,19)
	end = pd.Timestamp(2016,12,26)
	retail_kWh = 0.02254690804746962 #mid-Dec
	df_HVAC = pd.read_csv('Diss/HVAC_settings_2016-12-19_2016-12-26_ext.csv',index_col=[0])

recreate_data = False
recalculate_df_welfare = False

house_no = 0

folder_b = 'Diss/Diss_00'+str(ind_b) #+'_5min'
folder_WS = 'Diss/Diss_00'+str(ind_WS)
retail_MWh = retail_kWh*1000.

#Data
if recreate_data:
	df_T_b = pd.read_csv(folder_b+'/T_all.csv',skiprows=range(8))
	df_T_b['# timestamp'] = df_T_b['# timestamp'].map(lambda x: str(x)[:-4])
	df_T_b = df_T_b.iloc[:-1]
	df_T_b['# timestamp'] = pd.to_datetime(df_T_b['# timestamp'])
	df_T_b.set_index('# timestamp',inplace=True)
	df_T_b = df_T_b.loc[start:end]

	df_hvac_load_b = pd.read_csv(folder_b+'/hvac_load_all.csv',skiprows=range(8))
	df_hvac_load_b['# timestamp'] = df_hvac_load_b['# timestamp'].map(lambda x: str(x)[:-4])
	df_hvac_load_b = df_hvac_load_b.iloc[:-1]
	df_hvac_load_b['# timestamp'] = pd.to_datetime(df_hvac_load_b['# timestamp'])
	df_hvac_load_b.set_index('# timestamp',inplace=True) 
	df_hvac_load_b = df_hvac_load_b.loc[start:end]

	df_T = pd.read_csv(folder_WS+'/T_all.csv',skiprows=range(8))
	df_T['# timestamp'] = df_T['# timestamp'].map(lambda x: str(x)[:-4])
	df_T = df_T.iloc[:-1]
	df_T['# timestamp'] = pd.to_datetime(df_T['# timestamp'])
	df_T.set_index('# timestamp',inplace=True)
	df_T = df_T.loc[start:end]

	df_hvac_load = pd.read_csv(folder_WS+'/hvac_load_all.csv',skiprows=range(8))
	df_hvac_load['# timestamp'] = df_hvac_load['# timestamp'].map(lambda x: str(x)[:-4])
	df_hvac_load = df_hvac_load.iloc[:-1]
	df_hvac_load['# timestamp'] = pd.to_datetime(df_hvac_load['# timestamp'])
	df_hvac_load.set_index('# timestamp',inplace=True) 
	df_hvac_load = df_hvac_load.loc[start:end]

	df_prices = pd.read_csv(folder_WS+'/df_prices.csv',index_col=[0],parse_dates=True)
	df_prices_1min = df_prices.copy()
	df_prices_1min = df_prices_1min.loc[start:end]

	# #Temperature of a single house vs. price: Dow does temperature depend on price?
	#import pdb; pdb.set_trace()
	fig = ppt.figure(figsize=(8,4),dpi=150)   
	ppt.ioff()
	ax = fig.add_subplot(111)
	try:
		lns = ppt.scatter(df_prices_1min['clearing_price'].iloc[:-1],df_T[df_T.columns[house_no]],label='House '+str(house_no))
	except:
		lns = ppt.scatter(df_prices_1min['clearing_price'],df_T[df_T.columns[house_no]],label='House '+str(house_no))
	ax.set_xlabel('Price')
	ax.set_ylabel('Temperature')
	ax.set_xlim(-25,100)
	ppt.savefig('Diss/temperature_vs_price_fct_'+str(ind_WS)+'.png', bbox_inches='tight')
	#import pdb; pdb.set_trace()

#Benchmark
if recalculate_df_welfare:
	df_u = df_T.copy()
	df_u_b = df_T_b.copy()
	df_welfare = pd.DataFrame(index=df_u.columns,columns=['fixed_u','fixed_cost','fixed_T_mean','fixed_T_min','fixed_T_max','fixed_T_min5','fixed_T_max95','fixed_T_var','fixed_av_retail','LEM_u','LEM_cost','LEM_T_mean','LEM_T_min','LEM_T_max','LEM_T_min5','LEM_T_max95','LEM_T_var','LEM_av_retail'])
	for col in df_u.columns:
		print(col)

		alpha = df_HVAC['alpha'].loc[col]
		T_com = df_HVAC['comf_temperature'].loc[col]
		#import pdb; pdb.set_trace()
		df_u[col] = (df_u[col] - T_com)
		df_u[col] = -alpha*df_u[col].pow(2)
		sum_u = (df_u[col] - df_hvac_load[col]/12.*df_prices_1min['clearing_price']/1000.).sum()
		df_welfare['LEM_u'].loc[col] = df_u[col].sum()
		df_welfare['LEM_cost'].loc[col] = (df_hvac_load[col]/12.*df_prices_1min['clearing_price']/1000.).sum()
		df_welfare['LEM_T_mean'].loc[col] = df_T[col].mean()
		df_welfare['LEM_T_min'].loc[col] = df_T[col].min()
		df_welfare['LEM_T_max'].loc[col] = df_T[col].max()
		df_welfare['LEM_T_min5'].loc[col] = df_T[col].quantile(q=0.05)
		df_welfare['LEM_T_max95'].loc[col] = df_T[col].quantile(q=0.95)
		df_welfare['LEM_T_var'].loc[col] = df_T[col].var()
		df_welfare['LEM_av_retail'].loc[col] = 1000*(df_hvac_load[col]/12.*df_prices_1min['clearing_price']/1000.).sum()/(df_hvac_load[col].sum()/12.)

		df_u_b[col] = (df_u_b[col] - T_com)
		df_u_b[col] = -alpha*df_u_b[col].pow(2)
		sum_u_b = (df_u_b[col] - df_hvac_load_b[col]/12.*retail_MWh/1000.).sum()
		df_welfare['fixed_u'].loc[col] = df_u_b[col].sum()
		df_welfare['fixed_cost'].loc[col] = (df_hvac_load_b[col]/12.*retail_MWh/1000.).sum()
		df_welfare['fixed_T_mean'].loc[col] = df_T_b[col].mean()
		df_welfare['fixed_T_min'].loc[col] = df_T_b[col].min()
		df_welfare['fixed_T_max'].loc[col] = df_T_b[col].max()
		df_welfare['fixed_T_min5'].loc[col] = df_T_b[col].quantile(q=0.05)
		df_welfare['fixed_T_max95'].loc[col] = df_T_b[col].quantile(q=0.95)
		df_welfare['fixed_T_var'].loc[col] = df_T_b[col].var()
		df_welfare['fixed_av_retail'].loc[col] = retail_MWh
		
		#df_welfare.to_csv(folder_WS + '/df_welfare.csv')

	df_welfare = df_HVAC.merge(df_welfare,left_index=True,right_index=True)
	df_welfare.to_csv(folder_WS + '/df_welfare_withparameters.csv')

#import pdb; pdb.set_trace()
df_welfare = pd.read_csv(folder_WS + '/df_welfare_withparameters.csv',index_col=[0],parse_dates=True)
df_welfare['u_change'] = (df_welfare['LEM_u'] - df_welfare['LEM_cost']) - (df_welfare['fixed_u'] - df_welfare['fixed_cost'])
print(df_welfare['u_change'].mean())


#Histogram utility change

if start.month == 12:
	df_welfare = df_welfare.loc[df_welfare['heating_system'] != 'GAS']

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
#lns = ppt.hist(df_welfare['u_change'],bins=20,color='0.75',edgecolor='0.5')
lns = ppt.hist(df_welfare['u_change'],bins=np.arange(-100,100,0.5),color='0.75',edgecolor='0.5')
#ax.vlines(0,0,120,'0.6','--')
ax.set_xlim(-8,12)
ppt.xticks(np.arange(-8,14,2))
ax.set_ylim(0,130)
ax.set_xlabel('Utility change')
ax.set_ylabel('Number of houses')
ppt.savefig('Diss/hist_uchange_'+str(ind_WS)+'.pdf', bbox_inches='tight')
ppt.savefig('Diss/hist_uchange_'+str(ind_WS)+'.png', bbox_inches='tight')
#import pdb; pdb.set_trace()
#Remove customers with gas

#alpha - Delta u

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(np.log(df_welfare['alpha']),df_welfare['u_change'],color='0.6')
ax.set_xlabel('$\\alpha$ [log]')
ax.set_ylabel('Utility change')
ppt.savefig('Diss/hist_uchange_alpha_'+str(ind_WS)+'.png', bbox_inches='tight')

#beta - Delta u

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(df_welfare['beta'],df_welfare['u_change'],color='0.6')
ax.set_xlabel('$\\beta$')
ax.set_ylabel('Utility change')
ppt.savefig('Diss/hist_uchange_beta_'+str(ind_WS)+'.png', bbox_inches='tight')

#Temperature deviations under 

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(np.log(df_welfare['alpha']),df_welfare['LEM_T_mean']/df_welfare['comf_temperature'],marker='x',color='0.6')
ax.set_xlabel('$\\alpha$ [log]')
ax.set_ylim(1.0,1.1)
ax.set_ylabel('$\\overline{\\theta}/\\theta^{comf}$')
ppt.savefig('Diss/T_alpha_'+str(ind_WS)+'.pdf', bbox_inches='tight')

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(np.log(df_welfare['alpha']),df_welfare['LEM_T_mean']-df_welfare['comf_temperature'],marker='x',color='0.6')
ax.set_xlabel('$log(\\alpha)$')
#ax.set_ylim(0.0,7.0)
ax.set_ylabel('$\\overline{\\theta} - \\theta^{comf}$')
ppt.savefig('Diss/T_alpha_diff_'+str(ind_WS)+'.pdf', bbox_inches='tight')

((df_welfare['LEM_T_mean']/df_welfare['comf_temperature'])).loc[df_welfare['alpha'] >= df_welfare['alpha'].quantile(q=0.95)].mean()
((df_welfare['LEM_T_mean']-df_welfare['comf_temperature'])).loc[df_welfare['alpha'] >= df_welfare['alpha'].quantile(q=0.95)].mean()
((df_welfare['LEM_T_mean']/df_welfare['comf_temperature'])).loc[df_welfare['alpha'] <= df_welfare['alpha'].quantile(q=0.05)].mean()
((df_welfare['LEM_T_mean']-df_welfare['comf_temperature'])).loc[df_welfare['alpha'] <= df_welfare['alpha'].quantile(q=0.05)].mean()

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(np.log(df_welfare['alpha']),df_welfare['LEM_T_mean']/df_welfare['cooling_setpoint'],marker='x',color='0.6')
ax.set_xlabel('$log(\\alpha)$')
ax.set_xlim(-11,-6)
ax.set_ylim(0.98,1.03)
ax.set_ylabel('$\\overline{\\theta} / \\theta^{cool}$')
ppt.savefig('Diss/TmeanTset_alpha_'+str(ind_WS)+'.pdf', bbox_inches='tight')

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(np.log(df_welfare['alpha']),df_welfare['LEM_T_var'],marker='x',color='0.6')
ax.set_xlabel('$\\alpha$ [log]')
ax.set_ylabel('$\\sigma^2_{\\theta}$')
ppt.savefig('Diss/varT_alpha_'+str(ind_WS)+'.pdf', bbox_inches='tight')

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(np.log(df_welfare['alpha']),(df_welfare['LEM_T_max']-df_welfare['LEM_T_min'])/(df_welfare['fixed_T_max']-df_welfare['fixed_T_min']),marker='x',color='0.6')
ax.set_xlabel('$\\alpha$ [log]')
ax.set_ylabel('$\\Delta \\theta^{LEM} / \\Delta \\theta^{fixed}$')
ppt.savefig('Diss/diffT_alpha_'+str(ind_WS)+'.pdf', bbox_inches='tight')

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(np.log(df_welfare['alpha']),(df_welfare['LEM_T_max95']-df_welfare['LEM_T_min5'])/(df_welfare['fixed_T_max95']-df_welfare['fixed_T_min5']),marker='x',color='0.6')
ax.set_xlabel('$log(\\alpha)$')
ax.set_ylim(1.0,4.5)
ax.set_ylabel('$\\Delta \\theta^{LEM} / \\Delta \\theta^{fixed}$')
ppt.savefig('Diss/diffT5-95_alpha_'+str(ind_WS)+'.pdf', bbox_inches='tight')

((df_welfare['LEM_T_max95']-df_welfare['LEM_T_min5'])/(df_welfare['fixed_T_max95']-df_welfare['fixed_T_min5'])).loc[df_welfare['alpha'] > df_welfare['alpha'].quantile(q=0.95)].mean()
((df_welfare['LEM_T_max95']-df_welfare['LEM_T_min5'])).loc[df_welfare['alpha'] > df_welfare['alpha'].quantile(q=0.95)].mean()
((df_welfare['fixed_T_max95']-df_welfare['fixed_T_min5'])).loc[df_welfare['alpha'] > df_welfare['alpha'].quantile(q=0.95)].mean()

((df_welfare['LEM_T_max95']-df_welfare['LEM_T_min5'])/(df_welfare['fixed_T_max95']-df_welfare['fixed_T_min5'])).loc[df_welfare['alpha'] < df_welfare['alpha'].quantile(q=0.05)].mean()
((df_welfare['LEM_T_max95']-df_welfare['LEM_T_min5'])).loc[df_welfare['alpha'] < df_welfare['alpha'].quantile(q=0.05)].mean()
((df_welfare['fixed_T_max95']-df_welfare['fixed_T_min5'])).loc[df_welfare['alpha'] < df_welfare['alpha'].quantile(q=0.05)].mean()

import pdb; pdb.set_trace()
