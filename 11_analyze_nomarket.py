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
bfcts.analyze_year(folder_input,folder_output,s_settings,interval)