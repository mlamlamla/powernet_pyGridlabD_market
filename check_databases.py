print('Database includes the following tables:')
import mysql_functions

mydb, mycursor = mysql_functions.connect()

mycursor.execute('SHOW TABLES')
tables = mycursor.fetchall()  
for x in tables:
    print(x[0])
    mycursor.execute("SHOW columns FROM "+x[0])
    print([column[0] for column in mycursor.fetchall()])

# for table_name in table_list:
# 	print table_name
# 	mycursor.execute("SHOW columns FROM "+table_name)
# 	print [column[0] for column in mycursor.fetchall()]

#df = mysql_functions.get_values('market_buses')
#print df

#df = mysql_functions.get_values('market_houses')
#print df

mycursor.execute("SHOW PROCESSLIST;")