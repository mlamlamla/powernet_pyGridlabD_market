#Creates the relevant databases
import mysql_functions
import pandas as pd
import mysql.connector
from datetime import datetime
import numpy as np

mydb, mycursor = mysql_functions.connect()

mycursor.execute('SET FOREIGN_KEY_CHECKS = 0')

    # Table with market buy bids for appliances -> write only (from python)
    #Appliance_name is a polymorphic association to tables market_HVAC and market_battery
try:
    mycursor.execute('CREATE TABLE buy_bids (id INT AUTO_INCREMENT PRIMARY KEY, bid_price FLOAT, bid_quantity FLOAT, timedate TIMESTAMP, appliance_name VARCHAR(255) not null)')
except Exception as e:
    print('9')
    print('Error: ', e)

    # Table with capacity restrictions
try:
    mycursor.execute('CREATE TABLE capacity_restrictions (id INT AUTO_INCREMENT PRIMARY KEY, cap_rest FLOAT, timedate TIMESTAMP)')
except Exception as e:
    print('13')
    print('Error: ', e)

    # Table with market clearing prices -> write only (from python)
try:
    mycursor.execute('CREATE TABLE clearing_pq (id INT AUTO_INCREMENT PRIMARY KEY, clearing_price FLOAT, clearing_quantity FLOAT, timedate TIMESTAMP)')
except Exception as e:
    print('10')
    print('Error: ', e)

   # Table with characteristics of each appliance. / period vals: beg, for beginning, end, for end of time period. / What is system_mode (OFF, cool, heat) / Active -> boolean
try:
    mycursor.execute('CREATE TABLE market_appliance_meter (id INT AUTO_INCREMENT PRIMARY KEY, system_mode VARCHAR(255), heating_setpoint FLOAT, cooling_setpoint FLOAT, state_of_charge FLOAT, active VARCHAR(255), period VARCHAR(255), timedate TIMESTAMP, appliance_id integer not null, FOREIGN KEY (appliance_id) REFERENCES market_appliances(id))')
except Exception as e:
    print('7')
    print('Error: ', e)

try:
    mycursor.execute('CREATE TABLE market_appliances (id INT AUTO_INCREMENT PRIMARY KEY, house_name VARCHAR(255), appliance_name VARCHAR(255), k FLOAT, T_min FLOAT, T_max FLOAT, P_heat FLOAT, P_cool FLOAT, SOC_max FLOAT, u_max FLOAT, eff FLOAT)')
except Exception as e:
    print('6')
    print('Error: ',e) 

try:
    mycursor.execute('CREATE TABLE market_HVAC (id INT AUTO_INCREMENT PRIMARY KEY, house_name VARCHAR(255), appliance_name VARCHAR(255), k FLOAT, T_min FLOAT, T_max FLOAT, P_heat FLOAT, P_cool FLOAT)')
except Exception as e:
    print('11')
    print('Error: ', e)

try:
    mycursor.execute('CREATE TABLE market_HVAC_meter (id INT AUTO_INCREMENT PRIMARY KEY, system_mode VARCHAR(255), av_power FLOAT, heating_setpoint FLOAT, cooling_setpoint FLOAT, active VARCHAR(255), timedate TIMESTAMP, appliance_id integer not null, FOREIGN KEY (appliance_id) REFERENCES market_HVAC(id))')
except Exception as e:
    print('11')
    print('Error: ', e)

try:
    mycursor.execute('CREATE TABLE market_battery (id INT AUTO_INCREMENT PRIMARY KEY, house_name VARCHAR(255), appliance_name VARCHAR(255), appliance_id integer not null, SOC_max FLOAT, u_max FLOAT, eff FLOAT)')
except Exception as e:
    print('11')
    print('Error: ', e)

try:
    mycursor.execute('CREATE TABLE market_battery_meter (id INT AUTO_INCREMENT PRIMARY KEY, SOC FLOAT, active VARCHAR(255), timedate TIMESTAMP, appliance_id integer not null)')
except Exception as e:
    print('11')
    print('Error: ', e)

try:
    # Table with characteristics of each bus
    mycursor.execute('CREATE TABLE market_bus_meter (id INT AUTO_INCREMENT PRIMARY KEY, real_power FLOAT, timedate TIMESTAMP, bus_number integer not null, FOREIGN KEY (bus_number) REFERENCES market_buses(id))')
except Exception as e:
    print('3')
    print('Error: ', e)

try:
    # Table with all buses that are participating into the market
    mycursor.execute('CREATE TABLE market_buses (id INT AUTO_INCREMENT PRIMARY KEY, bus_name VARCHAR(255))')
except Exception as e:
    print('1')
    print('Error: ', e)

    # Table with EV specification
try:
    mycursor.execute('CREATE TABLE market_EV (id INT AUTO_INCREMENT PRIMARY KEY, house_name VARCHAR(255), appliance_name VARCHAR(255), appliance_id VARCHAR(255), SOC_max FLOAT, u_max FLOAT, eff FLOAT, charging_type VARCHAR(255), k FLOAT)')
except Exception as e:
    print('8')
    print('Error: ', e)

    # Table with EV charging
try:
    mycursor.execute('CREATE TABLE market_EV_meter (id INT AUTO_INCREMENT PRIMARY KEY, connected VARCHAR(255), SOC FLOAT, active VARCHAR(255), timedate TIMESTAMP, appliance_id integer not null)')
except Exception as e:
    print('8')
    print('Error: ', e)

    # Table with characteristics of each house
try:
    #mycursor.execute('CREATE TABLE market_house_meter (id INT AUTO_INCREMENT PRIMARY KEY, total_load FLOAT, real_power FLOAT, air_temperature FLOAT, outdoor_temperature FLOAT, timedate TIMESTAMP, house_number integer not null, FOREIGN KEY (house_number) REFERENCES market_houses(id))')
    mycursor.execute('CREATE TABLE market_house_meter (id INT AUTO_INCREMENT PRIMARY KEY, total_load FLOAT, real_power FLOAT, air_temperature FLOAT, outdoor_temperature FLOAT, timedate TIMESTAMP, house_number integer not null, FOREIGN KEY (house_number) REFERENCES market_houses(id))')
except Exception as e:
    print('5')
    print('Error: ', e)
      
    # Table with all houses that are participating into the market
try:
    mycursor.execute('CREATE TABLE market_houses (id INT AUTO_INCREMENT PRIMARY KEY, house_name VARCHAR(255))')
except Exception as e:
    print('4')
    print('Error: ', e)

    # Table with market historical prices
try:
    mycursor.execute('CREATE TABLE market_prices (id INT AUTO_INCREMENT PRIMARY KEY, mean_price FLOAT, var_price FLOAT, timedate TIMESTAMP)')
except Exception as e:
    print('8')
    print('Error: ', e)

    # Table with pv specification
try:
    mycursor.execute('CREATE TABLE market_pv (id INT AUTO_INCREMENT PRIMARY KEY, house_name VARCHAR(255), appliance_name VARCHAR(255), inverter_name VARCHAR(255), appliance_id integer not null, rated_power FLOAT)')
except Exception as e:
    print('8')
    print('Error: ', e)

    # Table with pv generation
try:
    mycursor.execute('CREATE TABLE market_pv_meter (id INT AUTO_INCREMENT PRIMARY KEY, P_Out FLOAT, timedate TIMESTAMP, appliance_id integer not null)')
except Exception as e:
    print('8')
    print('Error: ', e)
    
    # Table with supply bids
try:
    mycursor.execute('CREATE TABLE supply_bids (id INT AUTO_INCREMENT PRIMARY KEY, bid_price FLOAT, bid_quantity FLOAT, timedate TIMESTAMP, gen_name VARCHAR(255) not null)')
except Exception as e:
    print('12')
    print('Error: ', e)  
    
    # Table with unresponsive loads
try:
    mycursor.execute('CREATE TABLE unresponsive_loads (id INT AUTO_INCREMENT PRIMARY KEY, unresp_load FLOAT, slack FLOAT, active_loads FLOAT, timedate TIMESTAMP)')
except Exception as e:
    print('11')
    print('Error: ', e)

#reads in real WS market price data
try:
    mycursor.execute('CREATE TABLE awarded_bids (id INT AUTO_INCREMENT PRIMARY KEY, appliance_name VARCHAR(255) not null, p_bid FLOAT, q_bid FLOAT, timedate TIMESTAMP)')
except Exception as e:
    print('5')
    print('Error: ', e)

#reads in real WS market price data
try:
    mycursor.execute('CREATE TABLE WS_market (id INT AUTO_INCREMENT PRIMARY KEY, RT FLOAT, DA FLOAT, timedate TIMESTAMP)')
except Exception as e:
    print('5')
    print('Error: ', e)

# try:
#     mycursor.execute('CREATE TABLE ieee123_b412_house_measurements (id INT AUTO_INCREMENT PRIMARY KEY, real_power float, timedate timestamp, source_id integer not null, foreign key (source_id) references ieee123_b412_meters(id))')
# except Exception as e:
#     print '2'
#     print 'Error: ', e

# try:
#     mycursor.execute('CREATE TABLE ieee123_b412_meters (id INT AUTO_INCREMENT PRIMARY KEY, real_power float, timedate timestamp, source_id integer not null, foreign key (source_id) references ieee123_b412_meters(id))')
# except Exception as e:
#     print '2'
#     print 'Error: ', e

mycursor.execute('SET FOREIGN_KEY_CHECKS = 0')  
