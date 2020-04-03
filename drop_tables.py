#Drop the databases
import pandas as pd
import mysql.connector
from datetime import datetime
import numpy as np

mydb = mysql.connector.connect(
            host='powernet-gridlabd-rt.cftqw2r7udps.us-east-1.rds.amazonaws.com',
            #service=mysql
            #port=3306
            user='gridlabd',
            port='3306',
            passwd='tfPKrZ5lOSXAVf3Y',
            database='gridlabd'
        )
mycursor = mydb.cursor(buffered = True)

table_list = ['buy_bids','capacity_restrictions','clearing_pq','clearing_prices','ieee123_b412_bus_measurements','market_appliance_meter','market_appliances']
table_list += ['market_bus_meter','market_buses','market_house_meter','market_houses','market_prices','supply_bids','unresponsive_loads']

for table in table_list:
	sql = "DROP TABLE "+table
	mycursor.execute(sql) 