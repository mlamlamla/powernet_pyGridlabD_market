import pandas as pd
import mysql.connector
from datetime import datetime
import numpy as np

mydb = mysql.connector.connect(
            host='127.0.0.1',
            #service=mysql
            #port=3306
            user='root',
            port='3306',
            passwd='gridlabd',
            database='gridlabd'
        )

# mydb = mysql.connector.connect(
#             host='127.0.0.1',
#             #service=mysql
#             #port=3306
#             user='root',
#             port='3306',
#             passwd='gridlabd',
#             database='gridlabd'
#         )

mycursor = mydb.cursor(buffered = True)

#mycursor.execute('SHOW table status from gridlabd;')

#mycursor.execute('SELECT table_name AS `Table`, round(((data_length + index_length) / 1024 / 1024), 2) FROM information_schema.TABLES WHERE table_schema = 'gridlabd' AND table_name = 'market_appliances_meter';')

#KB
mycursor.execute('SELECT table_schema gridlabd, Round(Sum(data_length + index_length) / 1024 , 4) FROM   information_schema.tables GROUP  BY table_schema; ')

table = mycursor.fetchall() 
print table

#mydb.commit()
#mydb.close()