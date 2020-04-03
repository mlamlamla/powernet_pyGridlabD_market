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

# #query = 'INSERT INTO ' +table_name+'(var1_name, var2_name, ...) '+' VALUES(%s,%s, ...)'
# query = 'INSERT INTO clearing_prices(prices, supply, demand, timedate) VALUES(%s, %s, %s, %s)'
# timedate = '2015-07-01 14:58:30 EDT' # This you get from the precommit function: sim_time = os.getenv("clock")
# vals = (0.4, 5, 6, timedate)
# mycursor.execute(query, vals)
# # UNCOMMENT IF REALLY WANTS TO INSERT VALUES
# mydb.commit()

date = '2015-07-01'
time_prev = '14:45:00'
time_end = '15:00:00'
begin = date + ' ' + time_prev
end = date + ' ' + time_end

query = 'SELECT * FROM clearing_prices WHERE timedate >= %(begin)s AND timedate <= %(end)s'
print query
df = pd.read_sql(query, con=mydb, params={'begin': begin, 'end': end})
print(df)