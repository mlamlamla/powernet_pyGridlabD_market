import pandas as pd

################
#
# Market income
#
#################

ind_unconstrained = 95

df_settings = pd.read_csv('settings_Diss.csv',index_col=[0])
start = pd.to_datetime(df_settings['start_time'].loc[ind_unconstrained]) + pd.Timedelta(days=1)
end = pd.to_datetime(df_settings['end_time'].loc[ind_unconstrained])
city = df_settings['city'].loc[ind_unconstrained]
market_file = 'Ercot_LZ_South.csv'

df_WS = pd.read_csv('glm_generation_'+city+'/'+market_file,parse_dates=[0])
df_WS.rename(columns={'Unnamed: 0':'timestamp'},inplace=True)
df_WS.set_index('timestamp',inplace=True)
df_WS = df_WS.loc[start:end]

df_prices_unconstr = pd.read_csv('Diss/Diss_'+"{:04d}".format(ind_unconstrained)+'/df_prices.csv',index_col=[0],parse_dates=True)
df_prices_unconstr = df_prices_unconstr.loc[start:end]

inds = [57,56,55,54,58,59,60,61,62,64]
inds = [105,104,103,102,101,100,99,98] + [ind_unconstrained]

df_summary = pd.DataFrame(columns=['market_income'])
for ind in inds:
	folder_WS = 'Diss/Diss_'+"{:04d}".format(ind)

	df_prices = pd.read_csv(folder_WS+'/df_prices.csv',index_col=[0],parse_dates=True)
	df_prices_1min_LEM = df_prices.copy()
	df_prices_1min_LEM = df_prices_1min_LEM.loc[start:end]

	if df_settings['PV_share'].loc[ind_unconstrained] > 0.0:
		df_PV_LEM = pd.read_csv(folder_WS+'/total_P_Out.csv',skiprows=range(8))
		df_PV_LEM['# timestamp'] = df_PV_LEM['# timestamp'].map(lambda x: str(x)[:-4])
		df_PV_LEM['# timestamp'] = pd.to_datetime(df_PV_LEM['# timestamp'])
		df_PV_LEM.set_index('# timestamp',inplace=True)
		df_PV_LEM = df_PV_LEM.loc[start:end]

		df_prices_1min_LEM['WS_supply'] = (df_prices_1min_LEM['clearing_quantity'] - df_PV_LEM.sum(axis=1)/1000.)/1000.
	else:
		df_prices_1min_LEM['WS_supply'] = df_prices_1min_LEM['clearing_quantity']

	df_prices_1min_LEM['WS_supply'].loc[df_prices_1min_LEM['WS_supply'] < 0.0] = 0.0
	#import pdb; pdb.set_trace()
	market_income = ((df_prices_1min_LEM['clearing_price'] - df_prices_unconstr['clearing_price'])*df_prices_1min_LEM['WS_supply']/12000.).sum()

	C = df_settings['line_capacity'].loc[ind]/1000
	df_summary = df_summary.append(pd.DataFrame(index=[C],columns=['market_income'],data=[[market_income]]))

df_summary_round = df_summary.round(2)
df_summary_round.sort_index(ascending=False,inplace=True)
print(df_summary_round)
table = open('Diss/df_summary_market_income.tex','w')
table.write(df_summary_round.to_latex(index=True,escape=False,na_rep=''))
table.close()