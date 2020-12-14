import os
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as ppt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from numpy import arange 

ppt.rcParams['mathtext.fontset'] = 'stix'
ppt.rcParams['font.family'] = 'STIXGeneral'
ppt.rc('text.latex', preamble='\\usepackage(color')

#Analyze full loadportfolio
def analyze_year(folder,directory,s_settings,interval):
	#Get system data
	try:
		df_systemdata = pd.read_csv(directory+'/df_system.csv',index_col=[0],parse_dates=True)
	except:
		if not os.path.isdir(directory):
			os.mkdir(directory)
		df_systemdata = get_systemdata(folder,s_settings['market_data'],directory)
	print('HVAC consumption share: '+str(df_systemdata['hvac_load_houses'].sum()/df_systemdata['total_load_houses'].sum()*100))
	print('Grid losses: '+str(100*(1.-df_systemdata['total_load_houses'].sum()/df_systemdata['measured_real_power'].sum())))
	import pdb; pdb.set_trace()
	#import pdb; pdb.set_trace()
	procurement_cost = ((df_systemdata['measured_real_power']/(60./interval))*df_systemdata['RT']/1000.).sum()
	consumer_kWh = (df_systemdata['total_load_houses']/(60./interval)).sum()
	try:
		generation_kWh = (df_systemdata['PV_gen_houses']/(60./interval)).sum()
	except:
		generation_kWh = 0.0
	retail_kWh = procurement_cost/(consumer_kWh - generation_kWh)
	#/(df_systemdata['measured_real_power'].sum()/60.)
	print('Fixed retail tariff: '+str(retail_kWh))

	#df_systemdata = df_systemdata.iloc[(24*60):]

	#01: Plot load duration curve for the whole year
	get_loaddurationcurve(directory,df_systemdata)

	#02: Plot monthly average profiles
	plot_hourlyav_month(directory,df_systemdata)

	#03: Plot base load
	plot_baseloadav_month(directory,df_systemdata)

	#04: Plot HVAC
	plot_HVACav_month(directory,df_systemdata)

	#05: Plot PV
	if s_settings['PV_share'] > 0.0:
		plot_PVav_month(directory,df_systemdata)

	#06: Plot EV
	if s_settings['EV_share'] > 0.0:
		plot_EVav_month(directory,df_systemdata)

	#07: Plot EV
	if s_settings['Batt_share'] > 0.0:
		plot_batav_month(directory,df_systemdata)

	#08: Weekly mean and maximum
	get_weeklymeanmax(directory,df_systemdata)
	get_weeklymeanmax2(directory,df_systemdata)

	#09: Plot load duration curve of day peaks
	get_loaddurationcurve_days(directory,df_systemdata)

def analyze_week(folder,directory,start,end):
	df_systemdata = pd.read_csv(directory+'/df_system.csv',index_col=[0],parse_dates=True)
	df_systemdata_week = df_systemdata.loc[start:(end+pd.Timedelta(days=1))]
	#import pdb; pdb.set_trace()

	fig = ppt.figure(figsize=(9,3),dpi=150)   
	ax = fig.add_subplot(111)
	lns1 = ax.plot(df_systemdata_week['measured_real_power']/1000.,color='xkcd:sky blue')
	ax.set_xlabel('Time')
	ax.set_ylabel('Measured system peak load [MW]')
	ax.set_xlim(xmin=start,xmax=end+pd.Timedelta(days=1))
	ax.set_ylim(0.0,2.5)
	#ax.hlines(0,start,end+pd.Timedelta(days=1))
	ax2 = ax.twinx()
	lns2 = ax2.plot(df_systemdata_week['RT'],color='xkcd:orange')
	ax2.set_ylabel('WS market price [USD/MWh]')
	ax2.set_ylim(0.0)
	lns = lns1 + lns2 #+ lns3 + lns4
	labs = [l.get_label() for l in lns]
	L = ax.legend(lns, labs, bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=len(labs))
	#L.get_texts()[0].set_text('Total system load')
	#L.get_texts()[1].set_text('Total unresponsive system load')
	ppt.savefig(directory+'/week_'+str(start)+'.pdf', bbox_inches='tight')
	ppt.savefig(directory+'/week_'+str(start)+'.png', bbox_inches='tight')

def get_systemdata(folder,market_data,directory):
	#Physical: total system load at slack bus (node 149 in IEEE123)
    df_slack = pd.read_csv(folder+'/load_node_149.csv',skiprows=range(8))
    df_slack['# timestamp'] = df_slack['# timestamp'].map(lambda x: str(x)[:-4])
    df_slack = df_slack.iloc[:-1]
    df_slack['# timestamp'] = pd.to_datetime(df_slack['# timestamp'])
    df_slack.set_index('# timestamp',inplace=True)
    df_slack['measured_real_power'] = df_slack['measured_real_power']/1000
    print('Max measured real power: '+str(df_slack['measured_real_power'].iloc[15:].max()))
    
    #Total house load
    df_total_load = pd.read_csv(folder+'/total_load_all.csv',skiprows=range(8))
    df_total_load['# timestamp'] = df_total_load['# timestamp'].map(lambda x: str(x)[:-4])
    df_total_load = df_total_load.iloc[:-1]
    df_total_load['# timestamp'] = pd.to_datetime(df_total_load['# timestamp'])
    df_total_load.set_index('# timestamp',inplace=True)
    df_total_load = pd.DataFrame(index=df_total_load.index,columns=['total_load_houses'],data=df_total_load.sum(axis=1))
    print('Max total residential load: '+str(df_total_load['total_load_houses'].iloc[15:].max()))

    df_systemdata = df_slack.merge(df_total_load, how='outer', left_index=True, right_index=True)
    
    #Total hvac load
    df_hvac_load = pd.read_csv(folder+'/hvac_load_all.csv',skiprows=range(8))
    df_hvac_load['# timestamp'] = df_hvac_load['# timestamp'].map(lambda x: str(x)[:-4])
    df_hvac_load = df_hvac_load.iloc[:-1]
    df_hvac_load['# timestamp'] = pd.to_datetime(df_hvac_load['# timestamp'])
    df_hvac_load.set_index('# timestamp',inplace=True)                           
    df_hvac_load = pd.DataFrame(index=df_hvac_load.index,columns=['hvac_load_houses'],data=df_hvac_load.sum(axis=1))
    print('Max hvac load: '+str(df_hvac_load['hvac_load_houses'].iloc[15:].max()))

    df_systemdata = df_systemdata.merge(df_hvac_load, how='outer', left_index=True, right_index=True)

    try:
    	df_triplex = pd.read_csv(folder+'/triplex_meter.csv',skiprows=range(8))
    	df_triplex['# timestamp'] = df_triplex['# timestamp'].map(lambda x: str(x)[:-4])
    	df_triplex = df_triplex.iloc[:-1]
    	df_triplex['# timestamp'] = pd.to_datetime(df_triplex['# timestamp'])
    	df_triplex.set_index('# timestamp',inplace=True)
    	df_triplex = df_triplex/1000.
    except:
    	pass

    try:
    	df_inv_load = pd.read_csv(folder+'/total_P_Out.csv',skiprows=range(8))
    	df_inv_load['# timestamp'] = df_inv_load['# timestamp'].map(lambda x: str(x)[:-4])
    	df_inv_load = df_inv_load.iloc[:-1]
    	df_inv_load['# timestamp'] = pd.to_datetime(df_inv_load['# timestamp'])
    	df_inv_load.set_index('# timestamp',inplace=True)
    	df_inv_load = df_inv_load/1000.

    	list_dev = df_inv_load.columns.tolist()
    	list_PV = []
    	list_EV = []
    	list_bat = []
    	for dev in list_dev:
    		if 'PV' in dev:
    			list_PV += [dev]
    		elif 'EV' in dev:
    			list_EV += [dev]
    		elif 'Bat' in dev:
    			list_bat += [dev]
    	
    	df_PV = pd.DataFrame(index=df_inv_load.index,columns=['PV_gen_houses'],data=df_inv_load[list_PV].sum(axis=1))
    	df_systemdata = df_systemdata.merge(df_PV, how='outer', left_index=True, right_index=True)
    	df_EV = pd.DataFrame(index=df_inv_load.index,columns=['EV_load_houses'],data=df_inv_load[list_EV].sum(axis=1))
    	df_systemdata = df_systemdata.merge(df_EV, how='outer', left_index=True, right_index=True)
    	df_bat = pd.DataFrame(index=df_inv_load.index,columns=['bat_load_houses'],data=df_inv_load[list_bat].sum(axis=1))
    	df_systemdata = df_systemdata.merge(df_bat, how='outer', left_index=True, right_index=True)
    	#import pdb; pdb.set_trace()
    except:
    	pass

    start = df_systemdata.index[0]
    end = df_systemdata.index[-1]
    #import pdb; pdb.set_trace()
    df_WS = pd.read_csv('glm_generation_Austin/'+market_data,index_col=[0],parse_dates=True)
    df_systemdata = df_systemdata.merge(df_WS,how='outer',left_index=True,right_index=True)
    df_systemdata['DA'].fillna(method='ffill',inplace=True)
    df_systemdata['RT'].fillna(method='ffill',inplace=True)
    df_systemdata = df_systemdata.loc[start:end]

    df_systemdata.to_csv(directory+'/df_system.csv')
    return df_systemdata

#01
def get_loaddurationcurve(directory,df_systemdata):
	df_loadduration = df_systemdata.sort_values('measured_real_power',ascending=False)
	df_loadduration.index = range(len(df_loadduration))
	df_loadduration.index = df_loadduration.index/len(df_loadduration)*100 #get percentage
	#import pdb; pdb.set_trace()

	print('0% value: '+str(df_loadduration['measured_real_power'].iloc[0]))
	print('1% value: '+str(df_loadduration.loc[df_loadduration.index < 1]['measured_real_power'].iloc[-1]))
	print('2% value: '+str(df_loadduration.loc[df_loadduration.index < 2]['measured_real_power'].iloc[-1]))
	print('3% value: '+str(df_loadduration.loc[df_loadduration.index < 3]['measured_real_power'].iloc[-1]))
	print('4% value: '+str(df_loadduration.loc[df_loadduration.index < 4]['measured_real_power'].iloc[-1]))
	print('5% value: '+str(df_loadduration.loc[df_loadduration.index < 5]['measured_real_power'].iloc[-1]))
	print('Minimum load: '+str(df_loadduration['measured_real_power'].nsmallest(2)/1000.))
	print('Minimum load: '+str(df_loadduration['measured_real_power'].nsmallest(1)/1000.))

	fig = ppt.figure(figsize=(6,3),dpi=150)   
	ax = fig.add_subplot(111)
	lns = ax.plot(df_loadduration['measured_real_power']/1000.,'0.5')
	ax.set_xlabel('Percentiles [%]')
	ax.set_ylabel('Measured system load [MW]')
	ax.set_xlim(xmin=0.,xmax=100.)
	ax.set_ylim(ymin=0.)
	#ax.hlines(0,0,100)
	#lns = lns1 + lns2 + lns3 + lns4
	#labs = [l.get_label() for l in lns]
	#L = ax.legend(lns, labs, bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=len(labs))
	#L.get_texts()[0].set_text('Total system load')
	#L.get_texts()[1].set_text('Total unresponsive system load')
	ppt.savefig(directory+'/01_loaddurationcurve.pdf', bbox_inches='tight')
	ppt.savefig(directory+'/01_loaddurationcurve.png', bbox_inches='tight')

#02
def plot_hourlyav_month(directory,df_systemdata):
	fig = ppt.figure(figsize=(9,3),dpi=150)   
	ax = fig.add_subplot(111)
	lns = []
	delta_c = (0.8 - 0.2)/5.
	min_load = 0.0
	max_load = 0.0
	#import pdb; pdb.set_trace()
	for month in range(1,13):
		#import pdb; pdb.set_trace()
		try:
			df_systemdata_month = df_systemdata.loc[df_systemdata.index.month == month]
			df_systemdata_month['dt'] = df_systemdata_month.index.hour.astype(str) + ':' + df_systemdata_month.index.minute.astype(str)
			df_systemdata_month['dt'] = pd.to_datetime(df_systemdata_month['dt'], format='%H:%M')
			df_systemdata_month = df_systemdata_month.groupby('dt').mean()
			if month <= 6:
				color = str(0.8 - (0.2 + (0.8 - 0.2)/5*(month-1)))
				ls = '--'
			else:
				color = str(0.8 - (0.8 - (0.8 - 0.2)/5*(month-7)))
				ls = '-'
			lns += ppt.plot(df_systemdata_month['measured_real_power']/1000,color,ls=ls,label=str(month))
			ax.set_xlim(xmin=df_systemdata_month.index[0],xmax=df_systemdata_month.index[-1])
			if min_load < df_systemdata_month['measured_real_power'].min()/1000:
				min_load = df_systemdata_month['measured_real_power'].min()/1000
			if max_load < df_systemdata_month['measured_real_power'].max()/1000:
				max_load = df_systemdata_month['measured_real_power'].max()/1000
			#ax.hlines(0,df_systemdata_month.index[0],df_systemdata_month.index[-1])
			#print(month)
			#print(df_systemdata_month['measured_real_power'].max())
		except:
			pass
	ax.set_xlabel('Daytime')
	if min_load >= 0.0:
		ax.set_ylim(0,1.1*max_load)
	else:
		ax.set_ylim(1.1*min_load,1.1*max_load)
	ax.set_ylabel('Measured system load [MW]')
	ax.xaxis.set_major_locator(HourLocator(arange(0, 25, 3)))
	ax.xaxis.set_minor_locator(HourLocator(arange(0, 25, 1)))
	ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
	labs = [l.get_label() for l in lns]
	L = ax.legend(lns, labs, bbox_to_anchor=(0.5, -0.5), loc='lower center', ncol=6)
	ppt.savefig(directory+'/02_hourlyav_month.pdf', bbox_inches='tight')
	ppt.savefig(directory+'/02_hourlyav_month.png', bbox_inches='tight')
	#import pdb; pdb.set_trace()

#03
def plot_baseloadav_month(directory,df_systemdata):
	fig = ppt.figure(figsize=(9,3),dpi=150)   
	ax = fig.add_subplot(111)
	lns = []
	for month in range(1,13):
		#import pdb; pdb.set_trace()
		try:
			df_systemdata_month = df_systemdata.loc[df_systemdata.index.month == month]
			df_systemdata_month['dt'] = df_systemdata_month.index.hour.astype(str) + ':' + df_systemdata_month.index.minute.astype(str)
			df_systemdata_month['dt'] = pd.to_datetime(df_systemdata_month['dt'], format='%H:%M')
			df_systemdata_month = df_systemdata_month.groupby('dt').mean()
			lns += ppt.plot((df_systemdata_month['total_load_houses'] - df_systemdata_month['hvac_load_houses'])/1000,label=str(month))
			ax.set_xlim(xmin=df_systemdata_month.index[0],xmax=df_systemdata_month.index[-1])
		except:
			pass
	ax.set_xlabel('Daytime')
	ax.set_ylabel('Measured system load [MW]')
	ax.xaxis.set_major_locator(HourLocator(arange(0, 25, 3)))
	ax.xaxis.set_minor_locator(HourLocator(arange(0, 25, 1)))
	ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
	labs = [l.get_label() for l in lns]
	L = ax.legend(lns, labs, bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=len(labs))
	ppt.savefig(directory+'/03_baseloadav_month.pdf', bbox_inches='tight')
	ppt.savefig(directory+'/03_baseloadav_month.png', bbox_inches='tight')

#4
def plot_HVACav_month(directory,df_systemdata):
	fig = ppt.figure(figsize=(9,3),dpi=150)   
	ax = fig.add_subplot(111)
	lns = []
	for month in range(1,13):
		#import pdb; pdb.set_trace()
		try:
			df_systemdata_month = df_systemdata.loc[df_systemdata.index.month == month]
			df_systemdata_month['dt'] = df_systemdata_month.index.hour.astype(str) + ':' + df_systemdata_month.index.minute.astype(str)
			df_systemdata_month['dt'] = pd.to_datetime(df_systemdata_month['dt'], format='%H:%M')
			df_systemdata_month = df_systemdata_month.groupby('dt').mean()
			lns += ppt.plot(df_systemdata_month['hvac_load_houses']/1000,label=str(month))
			ax.set_xlim(xmin=df_systemdata_month.index[0],xmax=df_systemdata_month.index[-1])
		except:
			pass
	ax.set_xlabel('Daytime')
	ax.set_ylabel('Measured system load [MW]')
	ax.xaxis.set_major_locator(HourLocator(arange(0, 25, 3)))
	ax.xaxis.set_minor_locator(HourLocator(arange(0, 25, 1)))
	ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
	labs = [l.get_label() for l in lns]
	L = ax.legend(lns, labs, bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=len(labs))
	ppt.savefig(directory+'/04_HVACav_month.pdf', bbox_inches='tight')
	ppt.savefig(directory+'/04_HVACav_month.png', bbox_inches='tight')

#5
def plot_PVav_month(directory,df_systemdata):
	fig = ppt.figure(figsize=(9,3),dpi=150)   
	ax = fig.add_subplot(111)
	lns = []
	for month in range(1,13):
		#import pdb; pdb.set_trace()
		try:
			df_systemdata_month = df_systemdata.loc[df_systemdata.index.month == month]
			df_systemdata_month['dt'] = df_systemdata_month.index.hour.astype(str) + ':' + df_systemdata_month.index.minute.astype(str)
			df_systemdata_month['dt'] = pd.to_datetime(df_systemdata_month['dt'], format='%H:%M')
			df_systemdata_month = df_systemdata_month.groupby('dt').mean()
			lns += ppt.plot(df_systemdata_month['PV_gen_houses']/1000,label=str(month))
			ax.set_xlim(xmin=df_systemdata_month.index[0],xmax=df_systemdata_month.index[-1])
		except:
			pass
	ax.set_xlabel('Daytime')
	ax.set_ylabel('Measured system load [MW]')
	ax.xaxis.set_major_locator(HourLocator(arange(0, 25, 3)))
	ax.xaxis.set_minor_locator(HourLocator(arange(0, 25, 1)))
	ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
	labs = [l.get_label() for l in lns]
	L = ax.legend(lns, labs, bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=len(labs))
	ppt.savefig(directory+'/05_PVav_month.pdf', bbox_inches='tight')
	ppt.savefig(directory+'/05_PVav_month.png', bbox_inches='tight')

#6
def plot_EVav_month(directory,df_systemdata):
	fig = ppt.figure(figsize=(9,3),dpi=150)   
	ax = fig.add_subplot(111)
	lns = []
	for month in range(1,13):
		#import pdb; pdb.set_trace()
		try:
			df_systemdata_month = df_systemdata.loc[df_systemdata.index.month == month]
			df_systemdata_month['dt'] = df_systemdata_month.index.hour.astype(str) + ':' + df_systemdata_month.index.minute.astype(str)
			df_systemdata_month['dt'] = pd.to_datetime(df_systemdata_month['dt'], format='%H:%M')
			df_systemdata_month = df_systemdata_month.groupby('dt').mean()
			lns += ppt.plot(-df_systemdata_month['EV_load_houses']/1000,label=str(month))
			ax.set_xlim(xmin=df_systemdata_month.index[0],xmax=df_systemdata_month.index[-1])
		except:
			pass
	ax.set_xlabel('Daytime')
	ax.set_ylabel('Measured system load [MW]')
	ax.xaxis.set_major_locator(HourLocator(arange(0, 25, 3)))
	ax.xaxis.set_minor_locator(HourLocator(arange(0, 25, 1)))
	ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
	labs = [l.get_label() for l in lns]
	L = ax.legend(lns, labs, bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=len(labs))
	ppt.savefig(directory+'/06_EVav_month.pdf', bbox_inches='tight')
	ppt.savefig(directory+'/06_EVav_month.png', bbox_inches='tight')

#7
def plot_batav_month(directory,df_systemdata):
	fig = ppt.figure(figsize=(9,3),dpi=150)   
	ax = fig.add_subplot(111)
	lns = []
	for month in range(1,13):
		#import pdb; pdb.set_trace()
		try:
			df_systemdata_month = df_systemdata.loc[df_systemdata.index.month == month]
			df_systemdata_month['dt'] = df_systemdata_month.index.hour.astype(str) + ':' + df_systemdata_month.index.minute.astype(str)
			df_systemdata_month['dt'] = pd.to_datetime(df_systemdata_month['dt'], format='%H:%M')
			df_systemdata_month = df_systemdata_month.groupby('dt').mean()
			lns += ppt.plot(-df_systemdata_month['bat_load_houses']/1000,label=str(month))
			ax.set_xlim(xmin=df_systemdata_month.index[0],xmax=df_systemdata_month.index[-1])
		except:
			pass
	ax.set_xlabel('Daytime')
	ax.set_ylabel('Measured system load [MW]')
	ax.xaxis.set_major_locator(HourLocator(arange(0, 25, 3)))
	ax.xaxis.set_minor_locator(HourLocator(arange(0, 25, 1)))
	ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
	labs = [l.get_label() for l in lns]
	L = ax.legend(lns, labs, bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=len(labs))
	ppt.savefig(directory+'/07_batav_month.pdf', bbox_inches='tight')
	ppt.savefig(directory+'/07_batav_month.png', bbox_inches='tight')

#8
def get_weeklymeanmax(directory,df_systemdata):
	df_systemdata['proc_cost'] = (df_systemdata['measured_real_power']/12.)*df_systemdata['RT']
	df_systemdata['week'] = df_systemdata.index.week
	df_system_week = pd.DataFrame(index=range(df_systemdata['week'].min(),df_systemdata['week'].max()+1),columns=['week_max'],data=df_systemdata.groupby('week').max()['measured_real_power'].values)
	df_system_week['week_max'] = df_system_week['week_max']/1000.
	df_system_week['week_mean'] = df_systemdata.groupby('week').mean()['measured_real_power'].values/1000.
	df_system_week['week_median'] = df_systemdata.groupby('week').median()['measured_real_power'].values
	df_system_week['week_max_price'] = df_systemdata.groupby('week').max()['RT'].values
	df_system_week['week_mean_price'] = df_systemdata.groupby('week').mean()['RT'].values
	df_system_week['week_median_price'] = df_systemdata.groupby('week').median()['RT'].values
	df_system_week['week_var_price'] = df_systemdata.groupby('week').var()['RT'].values
	df_system_week['proc_cost'] = df_systemdata.groupby('week').sum()['proc_cost'].values
	df_system_week['procurement'] = df_systemdata.groupby('week').sum()['measured_real_power'].values/12.
	df_system_week = df_system_week.loc[df_system_week.index != 53] #Drop first week of January (partial)
	df_system_week = df_system_week.loc[df_system_week.index != 52] #Drop last week of Dec (partial)
	df_system_week['proc_cost_MWh'] = df_system_week['proc_cost']/df_system_week['procurement']
	
	df_system_week.sort_values('week_max',inplace=True) # --> 51,50,3; 51 is a holiday, and 3 has the maximum medium load between the three
	df_system_week.sort_values('week_mean',inplace=True) # --> 51,50,3; 51 is a holiday, and 3 has the maximum medium load between the three
	df_system_week.sort_values('week_median',inplace=True) # --> 51,50,3; 51 is a holiday, and 3 has the maximum medium load between the three
	#Highest summer week: 29 / 18. - 24.07.
	#And when are the highest prices?
	df_system_week.sort_values('week_var_price',inplace=True)
	#import pdb; pdb.set_trace()
	
	df_system_week.sort_index(inplace=True)
	for week_ind in df_system_week.index:
		print('Start week '+str(week_ind)+': '+str(df_systemdata.loc[df_systemdata['week'] == week_ind].index[0]))
	df_system_week[['proc_cost_MWh','week_max_price','week_mean_price']] = df_system_week[['proc_cost_MWh','week_max_price','week_mean_price']].round(2)
	df_system_week[['week_max','week_mean']] = df_system_week[['week_max','week_mean']].round(3)
	#df_system_week.drop('week_median',axis=1,inplace=True)
	#df_system_week.drop('week_median_price',axis=1,inplace=True)
	text_file = open(directory+"/base_case_results.txt", "w")

	df_system_week_txt = df_system_week[['proc_cost_MWh','week_max_price','week_mean_price','week_max','week_mean']]
	text_file.write(df_system_week_txt.to_latex())
	text_file.close()

def get_weeklymeanmax2(directory,df_systemdata):
	df_systemdata['proc_cost'] = (df_systemdata['measured_real_power']/12.)*df_systemdata['RT']
	av_proc_cost = df_systemdata['proc_cost'].sum()/(df_systemdata['measured_real_power']/12.).sum()
	print('Average procurement cost: '+str(av_proc_cost))

	df_systemdata['week'] = df_systemdata.index.week
	df_system_week = pd.DataFrame(index=range(df_systemdata['week'].min(),df_systemdata['week'].max()+1),columns=['week_max'],data=df_systemdata.groupby('week').max()['measured_real_power'].values)
	
	df_systemdata['data'] = df_systemdata.index
	df_system_week['start'] = df_systemdata.groupby('week')['data'].min()
	df_system_week['end'] = df_systemdata.groupby('week')['data'].max()
	df_system_week['time'] = ''
	for ind in df_system_week.index:
		df_system_week.at[ind,'time'] = "{:02d}".format(df_system_week['start'].loc[ind].month) +'/' + "{:02d}".format(df_system_week['start'].loc[ind].day) + ' - ' + "{:02d}".format(df_system_week['end'].loc[ind].month) +'/' + "{:02d}".format(df_system_week['end'].loc[ind].day)
	df_system_week['week_max'] = df_system_week['week_max']/1000.
	df_system_week['week_mean'] = df_systemdata.groupby('week').mean()['measured_real_power'].values/1000.
	df_system_week['week_median'] = df_systemdata.groupby('week').median()['measured_real_power'].values
	df_system_week['week_max_price'] = df_systemdata.groupby('week').max()['RT'].values
	df_system_week['week_mean_price'] = df_systemdata.groupby('week').mean()['RT'].values
	df_system_week['week_median_price'] = df_systemdata.groupby('week').median()['RT'].values
	#import pdb; pdb.set_trace()
	df_system_week['week_var_price'] = df_systemdata.groupby('week').var()['RT'].values
	df_system_week['week_std_price'] = df_systemdata.groupby('week').std()['RT'].values
	df_system_week['proc_cost'] = df_systemdata.groupby('week').sum()['proc_cost'].values
	df_system_week['procurement'] = df_systemdata.groupby('week').sum()['measured_real_power'].values/12.
	df_system_week = df_system_week.loc[df_system_week.index != 53] #Drop first week of January (partial)
	df_system_week = df_system_week.loc[df_system_week.index != 52] #Drop last week of Dec (partial)
	df_system_week['proc_cost_MWh'] = df_system_week['proc_cost']/df_system_week['procurement']
	
	df_system_week.sort_values('week_max',inplace=True) # --> 51,50,3; 51 is a holiday, and 3 has the maximum medium load between the three
	df_system_week.sort_values('week_mean',inplace=True) # --> 51,50,3; 51 is a holiday, and 3 has the maximum medium load between the three
	df_system_week.sort_values('week_median',inplace=True) # --> 51,50,3; 51 is a holiday, and 3 has the maximum medium load between the three
	#Highest summer week: 29 / 18. - 24.07.
	#And when are the highest prices?
	df_system_week.sort_values('week_var_price',inplace=True)
	#import pdb; pdb.set_trace()
	
	df_system_week.sort_index(inplace=True)
	for week_ind in df_system_week.index:
		print('Start week '+str(week_ind)+': '+str(df_systemdata.loc[df_systemdata['week'] == week_ind].index[0]))
	df_system_week.set_index('time',inplace=True)
	df_system_week[['proc_cost_MWh','week_max_price','week_mean_price']] = df_system_week[['proc_cost_MWh','week_max_price','week_mean_price']].round(2)
	df_system_week[['week_max','week_mean','week_std_price']] = df_system_week[['week_max','week_mean','week_std_price']].round(3)
	#df_system_week.drop('week_median',axis=1,inplace=True)
	#df_system_week.drop('week_median_price',axis=1,inplace=True)
	text_file = open(directory+"/base_case_results2.txt", "w")

	df_system_week_txt = df_system_week[['proc_cost_MWh','week_max_price','week_std_price','week_max']]
	text_file.write(df_system_week_txt.to_latex())
	text_file.close()
	#import pdb; pdb.set_trace()

#09
def get_loaddurationcurve_days(directory,df_systemdata):
	#import pdb; pdb.set_trace()
	df_systemdata['day'] = df_systemdata.index.date
	s_df = df_systemdata.groupby('day')['measured_real_power'].max()
	df_systemdata_days = pd.DataFrame(index=s_df.index,columns=['max_load'],data=s_df.values)
	df_loadduration = df_systemdata_days.sort_values('max_load',ascending=False)
	df_loadduration.index = range(len(df_loadduration))
	df_loadduration.index = df_loadduration.index/len(df_loadduration)*100 #get percentage
	#import pdb; pdb.set_trace()

	print('0% value: '+str(df_loadduration['max_load'].iloc[0]))
	print('1% value: '+str(df_loadduration.loc[df_loadduration.index < 1]['max_load'].iloc[-1]))
	print('2% value: '+str(df_loadduration.loc[df_loadduration.index < 2]['max_load'].iloc[-1]))
	print('3% value: '+str(df_loadduration.loc[df_loadduration.index < 3]['max_load'].iloc[-1]))
	print('4% value: '+str(df_loadduration.loc[df_loadduration.index < 4]['max_load'].iloc[-1]))
	print('5% value: '+str(df_loadduration.loc[df_loadduration.index < 5]['max_load'].iloc[-1]))

	fig = ppt.figure(figsize=(9,3),dpi=150)   
	ax = fig.add_subplot(111)
	lns = ax.plot(df_loadduration['max_load']/1000.)
	ax.set_xlabel('Percentiles [%]')
	ax.set_ylabel('Measured system peak load [MW]')
	ax.set_xlim(xmin=0,xmax=100)
	ax.hlines(0,0,100)
	#lns = lns1 + lns2 + lns3 + lns4
	#labs = [l.get_label() for l in lns]
	#L = ax.legend(lns, labs, bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=len(labs))
	#L.get_texts()[0].set_text('Total system load')
	#L.get_texts()[1].set_text('Total unresponsive system load')
	ppt.savefig(directory+'/09_loaddurationcurve_days.pdf', bbox_inches='tight')
	ppt.savefig(directory+'/09_loaddurationcurve_days.png', bbox_inches='tight')

