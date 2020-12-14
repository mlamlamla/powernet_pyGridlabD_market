import pandas as pd
import basecase_fcts as bfcts

# Input
run = 'Diss' #'Paper' #'FinalReport_Jul1d'
settings_file = 'settings_Diss.csv'
folder_input = 'Diss/Diss_0090'
folder_output = 'Diss/Diss_0090_eval'
ind = 90

# Settings
df_settings = pd.read_csv(settings_file)
s_settings = df_settings.loc[ind]
interval = 300

# Evaluate
#bfcts.analyze_year(folder_input,folder_output,s_settings,interval)

# Average length HVAC ops
folder = 'Diss/Diss_0202'
df_hvac_load = pd.read_csv(folder+'/hvac_load_all.csv',skiprows=range(8))
df_hvac_load['# timestamp'] = df_hvac_load['# timestamp'].map(lambda x: str(x)[:-4])
df_hvac_load = df_hvac_load.iloc[:-1]
df_hvac_load['# timestamp'] = pd.to_datetime(df_hvac_load['# timestamp'])
df_hvac_load.set_index('# timestamp',inplace=True)	

list_consecutives = []
for col in df_hvac_load.columns:
	consecutives = 0
	for ind in df_hvac_load.index:
		if df_hvac_load[col].loc[ind] > 0.0:
			consecutives += 1
		else:
			if consecutives > 0:
				list_consecutives += [consecutives]
				consecutives = 0

print('Average HVAC operations time: '+str(sum(list_consecutives)/len(list_consecutives)))
print(min(list_consecutives))
print(max(list_consecutives))

import pdb; pdb.set_trace()