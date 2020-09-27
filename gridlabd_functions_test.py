import datetime
from datetime import timedelta
from dateutil import parser

def on_precommit(t):
	dt_sim_time = parser.parse(gridlabd.get_global('clock')).replace(tzinfo=None)

	#Run market only every five minutes
	if not ((dt_sim_time.second == 0) and (dt_sim_time.minute % 5 == 0)):
		return t
	
	else: #interval in minutes #is not start time
		print('Start precommit: '+str(dt_sim_time))
		return t