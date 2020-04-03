import mysql.connector
from mysql.connector import Error
from mysql.connector import errorcode

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
Delete_all_query = """truncate table clearing_prices """
mycursor.execute(Delete_all_query)
mydb.commit()
print("All Record Deleted successfully ")