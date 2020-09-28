import sys
assert(sys.version_info.major>2)
import gridlabd
import time
import pandas
#import pdb; pdb.set_trace()

###USER input
ind = 66
df_settings = pandas.read_csv('settings_Diss.csv',index_col=[0]) #,parse_dates=['start_time','end_time'])

##################
#Assemble correct glm files
#################
if True:
	#write global file with settings from csv file: HH_global.py
	import global_functions
	global_functions.write_global(df_settings.loc[ind],ind,'none')
	#re-write glm model for flexible devices - deactivate if no new file needs to be generated
	import glm_functions
	glm_functions.rewrite_glmfile(rewrite_houses=False,mode='TS') #Re-populates houses
	#print('Is timestep 300 in min_timestep?')
	#import pdb; pdb.set_trace()
	
#import pdb; pdb.set_trace()
#import glm_functions
#out_dir = '/docker_powernet/glm_generation_Austin'
#glm_functions.modify_k(out_dir)


#generate new HVAC settings?
#import pdb; pdb.set_trace()
##################
#Run GridlabD
#################
print('run Gridlabd')
gridlabd.command('IEEE123_BP_2bus_1min.glm')
gridlabd.command('-D')
gridlabd.command('suppress_repeat_messages=FALSE')
#gridlabd.command('--debug')
#gridlabd.command('--verbose')
gridlabd.command('--warn')
gridlabd.start('wait')
