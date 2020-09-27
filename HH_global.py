import os

#Result file
results_folder = 'Diss/Diss_0065'
if not os.path.exists(results_folder):
	os.makedirs(results_folder)

#glm parameters
city = 'Austin'
month = 'july'
start_time_str = '2016/07/31 00:00'
end_time_str = '2016/08/08 00:00'
player_dir = 'players_Austin'
tmy_file = '722540TYA.tmy3'
slack_node = 'node_149'

#Flexible appliances
settings_file = 'HVAC_settings_2016-08-01_2016-08-08_ext.csv'
flexible_houses = 437
PV_share = 0.1
EV_share = 0.0
EV_data = 'None'
EV_speed = 'slow'
Batt_share = 0.0
assert PV_share >= Batt_share, 'More batteries than PV'
#Market parameters
C = 100000.0
market_data = 'Ercot_HBSouth.csv'
p_max = 10000.0
load_forecast = 'myopic'
unresp_factor = 0.0
FIXED_TARIFF = False
interval = 300
allocation_rule = 'by_price'

#Appliance specifications
delta = 3.0 #temperature bandwidth - HVAC inactivity
ref_price = 'forward'
price_intervals = 36 #p average calculation 
which_price = 'none' #battery scheduling

#include System Operator
include_SO = False

#precision in bidding and clearing price
prec = 4
M = 10000 #large number
ip_address = 'none'
