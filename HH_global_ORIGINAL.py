import os

#Result file
results_folder = '0012_Results'
if not os.path.exists(results_folder):
	os.makedirs(results_folder)

#glm parameters
start_time_str='2016-07-01 00:00:00'
end_time_str='2016-07-01 23:59:00'
player_dir = 'players_Austin'
tmy_file = '724940TYA.tmy3'
slack_node = 'node_149'

#Flexible appliances
flexible_houses = 300
PV_share = 0.2
EV_share = 0.2
Batt_share = 0.2
assert PV_share >= Batt_share, 'More batteries than PV'
assert PV_share >= EV_share, 'More EVs than PV'

#Market parameters
C = 1000 #in kW
p_max = 100.0 #USD per kW
unresp_factor = 1.05
FIXED_TARIFF = False
interval = 300 #in seconds # !!!!! check that with market object #check that correctly parametrized all over the program (5min might be hard coded)
allocation_rules = ['by_price','by_award']
allocation_rule = allocation_rules[0]

#Appliance specifications
delta = 3.0 #temperature bandwidth - HVAC inactivity
price_intervals = 12 #p average calculation 
which_price = 'RT' #battery scheduling

#include System Operator
include_SO = False

#precision in bidding and clearing price
prec = 4
M = 10000 #large number
ip_address = '10.16.94.100'