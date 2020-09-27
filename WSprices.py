import pandas as pd
import matplotlib.pyplot as ppt

ind = 22

df_WS = pd.read_csv('/Users/admin/Documents/powernet/powernet_markets_Dissertation/glm_generation_Austin/Ercot_HBSouth.csv',index_col=[0],parse_dates=True)
df_prices = pd.read_csv('/Users/admin/Documents/powernet/powernet_markets_Dissertation/Diss/Diss_'+"{:04d}".format(ind)+'/df_prices.csv',index_col=[0],parse_dates=True)

start = df_prices.index[0] #pd.Timestamp(2016,4,24)
end = df_prices.index[-1] #pd.Timestamp(2016,5,1)

for date in pd.date_range(start,end):
	fig = ppt.figure(figsize=(9,3),dpi=150)   
	ax = fig.add_subplot(111)
	lns1 = ax.plot(df_WS['DA'],label='DA')
	lns2 = ax.plot(df_WS['RT'],label='RT')
	lns3 = ax.plot(df_prices['clearing_price'],label='LEM')
	ax.set_xlabel('Date')
	ax.set_xlim(date,date+pd.Timedelta(days=1))
	ax.set_ylim(-50.,100.)
	ax.set_ylabel('USD/MWh')
	lns = lns1 + lns2 + lns3
	labs = [l.get_label() for l in lns]
	L = ax.legend(lns, labs, bbox_to_anchor=(0.5, -0.5), loc='lower center', ncol=6)
	ppt.savefig('Diss/Diss_'+"{:04d}".format(ind)+'/WSprices'+str(date)+'.png', bbox_inches='tight')

	df_WS_i = df_WS.loc[date:(date+pd.Timedelta(days=1))]
	df_WS_i['diff'] = df_WS_i['DA'] - df_WS_i['RT']
	print('Mean price diff DA - RT: '+str(df_WS_i['diff'].mean()))
	print('Var price diff DA - RT: '+str(df_WS_i['diff'].var()))