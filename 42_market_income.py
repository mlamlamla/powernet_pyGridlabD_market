import pandas as pd

################
#
# Market income
#
#################

df_summary = pd.DataFrame(columns=['market_income'])
for ind in [57,56,55,54,58,59,60,61,62,64]:
	folder_WS = 'Diss/Diss_00'+str(ind)

	df_prices = pd.read_csv(folder_WS+'/df_prices.csv',index_col=[0],parse_dates=True)
	df_prices_1min_LEM = df_prices.copy()
	df_prices_1min_LEM = df_prices_1min_LEM.loc[start:end]

	df_PV_LEM = pd.read_csv(folder_WS+'/total_P_Out.csv',skiprows=range(8))
	df_PV_LEM['# timestamp'] = df_PV_LEM['# timestamp'].map(lambda x: str(x)[:-4])
	df_PV_LEM['# timestamp'] = pd.to_datetime(df_PV_LEM['# timestamp'])
	df_PV_LEM.set_index('# timestamp',inplace=True)
	df_PV_LEM = df_PV_LEM.loc[start:end]

	df_prices_1min_LEM['WS_supply'] = (df_prices_1min_LEM['clearing_quantity'] - df_PV_LEM.sum(axis=1)/1000.)/1000.
	df_prices_1min_LEM['WS_supply'].loc[df_prices_1min_LEM['WS_supply'] < 0.0] = 0.0
	market_income = ((df_prices_1min_LEM['clearing_price'] - df_prices_1min_b['clearing_price'])*df_prices_1min_LEM['WS_supply']/12.).sum()

	C = df_settings['line_capacity'].loc[ind]/1000
	df_summary = df_summary.append(pd.DataFrame(index=[C],columns=['market_income'],data=[[market_income]]))

df_summary_round = df_summary.round(2)
df_summary_round.sort_index(ascending=False,inplace=True)
print(df_summary_round)
table = open('Diss/df_summary_market_income.tex','w')
table.write(df_summary_round.to_latex(index=True,escape=False,na_rep=''))
table.close()