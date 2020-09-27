import gldimport
import os
import random
import pandas
import json
import numpy as np
import datetime
from datetime import timedelta
from dateutil import parser
import HH_functions as HHfct
import battery_functions as Bfct
import EV_functions as EVfct
import PV_functions as PVfct
import market_functions as Mfct
import time

from HH_global import results_folder, flexible_houses, C, p_max, market_data, which_price, city, month
from HH_global import interval, prec, price_intervals, allocation_rule, unresp_factor, load_forecast
from HH_global import FIXED_TARIFF, include_SO, EV_data

def on_init(t):
	global t0;
	t0 = time.time()

	global step;
	step = 0

	batteries = gldimport.find_objects('class=battery')
	global batterylist, EVlist;
	batterylist, EVlist = gldimport.sort_batteries(batteries)
	
	global df_EV_state;
	if EV_data == 'None':
		df_EV_state = EVfct.get_settings_EVs_rnd(EVlist,interval)
	else:
		df_EV_state = EVfct.get_settings_EVs(EVlist,interval)

	print('Initialize finished after '+str(time.time()-t0))
	return True

def init(t):
	print('Objective-specific Init')
	return True

#Global precommit
#Should be mostly moved to market precommit
def on_precommit(t):
	dt_sim_time = parser.parse(gridlabd.get_global('clock')).replace(tzinfo=None)

	#Run market only every five minutes
	if not ((dt_sim_time.second == 0) and (dt_sim_time.minute % (interval/60) == 0)):
		return t
	
	else: #interval in minutes #is not start time
		print('Start precommit: '+str(dt_sim_time))
		global step;
		global df_house_state, df_battery_state, df_EV_state, df_PV_state;
		global df_buy_bids, df_supply_bids, df_awarded_bids;

		#Update physical values for new period
		#global df_house_state;
		global batterylist, EVlist;
		if len(EVlist) > 0:
			if EV_data == 'None':
				df_EV_state = EVfct.update_EV_rnd(dt_sim_time,df_EV_state)
			else:
				df_EV_state = EVfct.update_EV(dt_sim_time,df_EV_state)

		EVs_connected = len(df_EV_state.loc[df_EV_state['connected'] == 1])
		EVs_charging = len(df_EV_state.loc[df_EV_state['u_t'] > 0.0])
		if (len(EVlist) > 0) and (EVs_connected > 0):
			if EVs_charging > 0:
				df_EV_state = EVfct.charge_EV(dt_sim_time,df_EV_state) #Charges if SOC_max is not achieved yet
		step += 1
		return t

def on_term(t):
	print('Simulation ended, saving results')
	#saving_results()

	global t0;
	t1 = time.time()
	print('Time needed (min):')
	print((t1-t0)/60)
	return None

def saving_results():
	#Save settings of objects
	global df_house_state;
	df_house_state.to_csv(results_folder+'/df_house_state.csv')
	global df_battery_state
	df_battery_state.to_csv(results_folder+'/df_battery_state.csv')
	global df_EV_state
	df_EV_state.to_csv(results_folder+'/df_EV_state.csv')
	global df_PV_state;
	df_PV_state.to_csv(results_folder+'/df_PV_state.csv')

	#Saving former mysql
	global df_prices;
	df_prices.to_csv(results_folder+'/df_prices.csv')
	global df_supply_bids;
	df_supply_bids.to_csv(results_folder+'/df_supply_bids.csv')
	global df_buy_bids;
	df_buy_bids.to_csv(results_folder+'/df_buy_bids.csv')
	global df_awarded_bids;
	df_awarded_bids.to_csv(results_folder+'/df_awarded_bids.csv')

	#Saving mysql databases
	#import download_databases
	#download_databases.save_databases(timestamp)
	#mysql_functions.clear_databases(table_list) #empty up database

	#Saving globals
	file = 'HH_global.py'
	new_file = results_folder+'/HH_global.py'
	glm = open(file,'r') 
	new_glm = open(new_file,'w') 
	j = 0
	for line in glm:
	    new_glm.write(line)
	glm.close()
	new_glm.close()

	#Do evaluations
	return

#Object-specific precommit
def precommit(obj,t) :
	print(t)
	tt =  int(300*((t/300)+1))
	print('Market precommit')
	print(tt)
	return gridlabd.NEVER #t #True #tt 


