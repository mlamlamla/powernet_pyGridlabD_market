import pandas as pd

#import pdb; pdb.set_trace()
#df_WS = pd.DataFrame(index=pd.date_range(pd.Timestamp(2016,1,1,0,0), pd.Timestamp(2016,12,31,23,55), freq="5min"),columns=['DA','RT'])
df_WS = pd.read_csv('glm_generation_Austin/Ercot_HBSouth.csv',index_col=[0],parse_dates=True)

# for sheet in range(12):
# 	print('DAM, sheet '+str(sheet))
# 	df_DAM_month = pd.read_excel('glm_generation_Austin/Ercot_DAM_2016.xlsx',sheet_name=sheet,index_col=[0],parse_dates=True)
# 	df_DAM_month = df_DAM_month.loc[df_DAM_month['Settlement Point'] == 'HB_SOUTH']
# 	df_DAM_month['Time'] = pd.to_timedelta((df_DAM_month['Hour Ending']).astype(str) + ':00', unit='h') - pd.Timedelta(hours=1)
# 	df_DAM_month.index = df_DAM_month.index + df_DAM_month['Time']
# 	df_WS['DA'].loc[df_DAM_month.index] = df_DAM_month['Settlement Point Price']
# 	df_WS.to_csv('glm_generation_Austin/Ercot_HBSouth.csv')

for sheet in range(19,19):
	print('RTP, sheet '+str(sheet))
	df_RTP_month = pd.read_excel('glm_generation_Austin/Ercot_RTP_2016.xlsx',sheet_name=sheet,index_col=[0],parse_dates=True)
	#import pdb; pdb.set_trace()
	df_RTP_month['Time'] = pd.to_timedelta((df_RTP_month['Delivery Hour']-1).astype(str) + ':'+(15*(df_RTP_month['Delivery Interval']-1)).astype(str)+':00', unit='h')
	df_RTP_month.index = df_RTP_month.index + df_RTP_month['Time']
	df_RTP_month = df_RTP_month.loc[df_RTP_month['Settlement Point Name'] == 'HB_SOUTH']
	df_WS['RT'].loc[df_RTP_month.index] = df_RTP_month['Settlement Point Price']
	df_WS.to_csv('glm_generation_Austin/Ercot_HBSouth.csv')
	
#import pdb; pdb.set_trace()
df_WS.fillna(method='ffill',inplace=True)
df_WS.to_csv('glm_generation_Austin/Ercot_HBSouth.csv')
