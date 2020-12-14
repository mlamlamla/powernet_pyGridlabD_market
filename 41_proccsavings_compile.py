#This file compares the WS market participation with stationary price bids to fixed price scenarie

import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
#import pdb; pdb.set_trace()

# Default
run = 'Diss'
ind_b = 90

# Find relevant runs
df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])
inds = [157,159,129,160,161,162,163,164,165,166,167,168,156,169,170,171]
inds += [125,124,126,128,127]
inds += [*range(172,202)]
df_settings = df_settings.loc[inds]

try:
	df_results = pd.read_csv(run + '/' + 'weekly_procc_changes.csv',index_col=[0])
	new = False
except:
	df_results = pd.DataFrame(index=df_settings.index,columns=['RR','RR_wolosses','max_p','var_p','av_proc_cost_WS','supply_b','supply_cost_b','supply_WS','supply_cost_WS'],data=0.0)
	new = True

if new:
	for ind_WS in df_settings.index:
		df_HVAC = pd.read_csv(run+'/HVAC_settings/'+df_settings['settings_file'].loc[ind_WS],index_col=[0])
		start = pd.to_datetime(df_settings['start_time'].loc[ind_WS]) + pd.Timedelta(days=1)
		end = pd.to_datetime(df_settings['end_time'].loc[ind_WS])
		print(ind_WS)
		print(start)
		print(end)

		recread_data = True 
		recalculate_df_welfare = True

		house_no = 0

		folder_b = run + '/Diss_'+"{:04d}".format(ind_b) #+'_5min'
		folder_WS = run + '/Diss_'+"{:04d}".format(ind_WS)

		##################
		#
		# Calculate retail rate in fixed price scenario / no TS
		#
		##################

		folder = folder_b
		city = 'Austin'
		market_file = 'Ercot_HBSouth.csv'
		market_file = 'Ercot_LZ_SOUTH.csv'
		market_file = df_settings['market_data'].loc[ind_WS]
		market_file = 'Ercot_LZ_SOUTH.csv'

		df_slack = pd.read_csv(folder+'/load_node_149.csv',skiprows=range(8))
		df_slack['# timestamp'] = df_slack['# timestamp'].map(lambda x: str(x)[:-4])
		df_slack = df_slack.iloc[:-1]
		df_slack['# timestamp'] = pd.to_datetime(df_slack['# timestamp'])
		df_slack.set_index('# timestamp',inplace=True)
		df_slack = df_slack.loc[start:end]
		df_slack = df_slack/1000 #kW

		df_WS = pd.read_csv('glm_generation_'+city+'/'+market_file,parse_dates=[0])
		df_WS.rename(columns={'Unnamed: 0':'timestamp'},inplace=True)
		df_WS.set_index('timestamp',inplace=True)
		df_WS = df_WS.loc[start:end]

		df_WS['system_load'] = df_slack['measured_real_power']
		supply_wlosses = (df_WS['system_load']/1000./12.).sum() # MWh
		df_WS['supply_cost'] = df_WS['system_load']/1000.*df_WS['RT']/12.
		supply_cost_wlosses = df_WS['supply_cost'].sum()

		df_total_load = pd.read_csv(folder+'/total_load_all.csv',skiprows=range(8)) #in kW
		df_total_load['# timestamp'] = df_total_load['# timestamp'].map(lambda x: str(x)[:-4])
		df_total_load = df_total_load.iloc[:-1]
		df_total_load['# timestamp'] = pd.to_datetime(df_total_load['# timestamp'])
		df_total_load.set_index('# timestamp',inplace=True)
		df_total_load = df_total_load.loc[start:end]
		total_load = (df_total_load.sum(axis=1)/12.).sum() #kWh

		df_WS['res_load'] = df_total_load.sum(axis=1)
		supply_wolosses = (df_WS['res_load']/1000./12.).sum() # only residential load, not what is measured at trafo
		df_WS['res_cost'] = df_WS['res_load']/1000.*df_WS['RT']/12.
		supply_cost_wolosses = df_WS['res_cost'].sum()

		try:
			df_inv_load = pd.read_csv(folder+'/total_P_Out.csv',skiprows=range(8)) #in W
			df_inv_load['# timestamp'] = df_inv_load['# timestamp'].map(lambda x: str(x)[:-4])
			df_inv_load = df_inv_load.iloc[:-1]
			df_inv_load['# timestamp'] = pd.to_datetime(df_inv_load['# timestamp'])
			df_inv_load.set_index('# timestamp',inplace=True)  
			df_inv_load = df_inv_load.loc[start:end]
			PV_supply = (df_inv_load.sum(axis=1)/1000./12.).sum() #in kWh
		except:
			PV_supply = 0.0

		print('RR with net-metering (w/o losses')

		net_demand  = total_load - PV_supply
		retail_kWh = supply_cost_wlosses/net_demand
		retail_kWh_wolosses = supply_cost_wolosses/net_demand

		retail_MWh = retail_kWh*1000.
		df_results.at[ind_WS,'supply_b'] = supply_wlosses
		df_results.at[ind_WS,'supply_cost_b'] = df_WS['supply_cost'].sum()
		df_results.at[ind_WS,'RR'] = retail_MWh
		retail_MWh_wolosses = retail_kWh_wolosses*1000.
		df_results.at[ind_WS,'RR_wolosses'] = retail_MWh_wolosses
		df_results.at[ind_WS,'max_p'] = df_WS['RT'].max()
		df_results.at[ind_WS,'var_p'] = df_WS['RT'].var()
		df_results.at[ind_WS,'weighted_var_p'] = (df_WS['system_load']/df_WS['system_load'].sum()*(df_WS['RT'] - df_WS['RT'].mean()).pow(2)).sum()

		##################
		#
		# Evaluate procurement cost in TS
		#
		##################

		df_slack_WS = pd.read_csv(folder_WS+'/load_node_149.csv',skiprows=range(8))
		df_slack_WS['# timestamp'] = df_slack_WS['# timestamp'].map(lambda x: str(x)[:-4])
		df_slack_WS = df_slack_WS.iloc[:-1]
		df_slack_WS['# timestamp'] = pd.to_datetime(df_slack_WS['# timestamp'])
		df_slack_WS.set_index('# timestamp',inplace=True)
		df_slack_WS = df_slack_WS.loc[start:end]
		df_slack_WS = df_slack_WS/1000 #kW

		df_WS['system_load_WS'] = df_slack_WS['measured_real_power']
		supply_wlosses_WS = (df_WS['system_load_WS']/1000./12.).sum() # MWh
		df_WS['supply_cost_WS'] = df_WS['system_load_WS']/1000.*df_WS['RT']/12.
		supply_cost_wlosses_WS = df_WS['supply_cost_WS'].sum()

		#import pdb; pdb.set_trace()
		df_results.at[ind_WS,'supply_WS'] = supply_wlosses_WS
		df_results.at[ind_WS,'supply_cost_WS'] = df_WS['supply_cost_WS'].sum()
		df_results.at[ind_WS,'av_proc_cost_WS'] = supply_cost_wlosses_WS/supply_wlosses_WS

	df_results.to_csv(run + '/' + 'weekly_procc_changes.csv')

df_results['savings'] = df_results['supply_cost_b'] - df_results['supply_cost_WS']
print('Savings in procurement cost from switching to LEM: '+str( df_results['savings'].sum()))
print('Rel savings in procurement cost from switching to LEM: '+str( 100*df_results['savings'].sum()/df_results['supply_cost_b'].sum()) + '%')
print('\n')
print('Energy imported in benchmark: '+str(df_results['supply_b'].sum()))
print('Energy imported in LEM: '+str(df_results['supply_WS'].sum()))
print('Energy import change: '+str((df_results['supply_WS'].sum())/df_results['supply_b'].sum()*100. - 100.))
print('Max weekly savings: '+str(df_results['savings'].max()) + ' in ind ' + str(df_results.loc[df_results['savings'] == df_results['savings'].max()].index[0]))

df_results['difference'] = df_results['RR'] - df_results['av_proc_cost_WS'] # RR = av_proc_cost_b

# Histogram procurement cos change

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
max_y = 8.
ax = fig.add_subplot(111)
#lns = ppt.hist(df_results['savings'],bins=51,color='0.75',edgecolor='0.5')
lns = ppt.hist(df_results['savings'],bins=range(-50,2300,50),color='0.75',edgecolor='0.5')
#ax.set_ylim(0,75)
if df_results['savings'].min() > 0.0:
	ax.set_xlim(0,df_results['savings'].max()*1.05)
else:
	ax.vlines(0,0,max_y,'k',lw=1)
ax.set_xlabel('Procurement cost savings [USD]')
if max_y > 0.0:
	ax.set_ylim(0,max_y)
ax.set_ylabel('Number of weeks')
ppt.savefig(run+'/41_hist_proccost_change.png', bbox_inches='tight')
ppt.savefig(run+'/41_hist_proccost_change.pdf', bbox_inches='tight')

print('Median savings: '+str(df_results['savings'].median()))
print('Number of weeks with >400 USD savings: '+str(len(df_results['savings'].loc[df_results['savings'] > 400.])))
print('Number of weeks with >1000 USD savings: '+str(len(df_results['savings'].loc[df_results['savings'] > 1000.])))
print('Number of weeks with negative savings: '+str(len(df_results['savings'].loc[df_results['savings'] < 0.])))

#

reg = LinearRegression()
reg.fit(df_results['RR'].to_numpy().reshape(len(df_results),1),(df_results['difference']).to_numpy().reshape(len(df_results),1))
reg.coef_

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(df_results['RR'],df_results['difference'],marker='x',color='0.6')
#import pdb; pdb.set_trace()
lns2 = ppt.plot(df_results['RR'],reg.intercept_[0] + reg.coef_[0][0]*df_results['RR'],'-',color='0.25')
ax.set_xlabel('Average procurement cost [USD/MWh] (benchmark)')
#ax.set_xlim(left=0.0)
#ax.set_ylim(bottom=0.0)
ax.set_ylabel('Average procurement cost savings [USD/MWh]')
ppt.savefig(run + '/41_proccost_savings.png', bbox_inches='tight')
ppt.savefig(run + '/41_proccost_savings.pdf', bbox_inches='tight')

print('Savings as a function of procurement cost')
print(str(reg.intercept_[0]) + ' + ' + str(reg.coef_[0][0]) + '* RR')

#

reg2 = LinearRegression()
reg2.fit(np.sqrt(df_results['var_p'].to_numpy()).reshape(len(df_results),1),(df_results['difference']).to_numpy().reshape(len(df_results),1))
reg2.coef_

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(np.sqrt(df_results['var_p']),df_results['difference'],marker='x',color='0.6')
#import pdb; pdb.set_trace()
lns2 = ppt.plot(np.sqrt(df_results['var_p']),reg2.intercept_[0] + reg2.coef_[0][0]*np.sqrt(df_results['var_p']),'-',color='0.25')
ax.set_xlabel('WS price standard deviation [USD/MWh]')
#ax.set_xlim(left=0.0)
#ax.set_ylim(bottom=0.0)
ax.set_ylabel('Average procurement cost savings [USD/MWh]')
ppt.savefig(run + '/41_WSpricestd_savings.png', bbox_inches='tight')
ppt.savefig(run + '/41_WSpricestd_savings.pdf', bbox_inches='tight')

print('Savings as a function of price std')
print(str(reg2.intercept_[0]) + ' + ' + str(reg2.coef_[0][0]) + '* std')
print('R2: '+ str(reg2.score(np.sqrt(df_results['var_p'].to_numpy()).reshape(len(df_results),1),(df_results['difference']).to_numpy().reshape(len(df_results),1))))


#

Y = (df_results['difference']).to_numpy().reshape(len(df_results),1)
X = np.sqrt(df_results['var_p'])
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
print(results.summary())

df_results['std_p'] = np.sqrt(df_results['var_p'])
df_results_short = df_results.loc[df_results['std_p'] < 90.]
Y2 = (df_results_short['difference']).to_numpy().reshape(len(df_results_short),1)
X2 = df_results_short['std_p'].to_numpy().reshape(len(df_results_short),1)
X2 = sm.add_constant(X2)
model2 = sm.OLS(Y2,X2)
results2 = model2.fit()
print(results2.summary())

import pdb; pdb.set_trace()