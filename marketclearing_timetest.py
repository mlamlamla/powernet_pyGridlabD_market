import numpy as np
import pandas as pd
import market_functions as Mfcts
import time

N_total = 2500
buy_bids = np.random.rand(N_total,2)

df_time = pd.DataFrame(index=range(N_total),columns=['sorting_time','clearing_time'])

retail = Mfcts.Market()
retail.reset()
retail.Pmin = 0.0
retail.Pmax = 1.0
retail.Pprec = 3

for n in range(1,N_total):
	print(n)
	retail.reset()
	for i in range(n):
		retail.buy(buy_bids[i,0],buy_bids[i,1]) #(q,p)
	retail.sell(n*0.5,0.5) #(q,p)
	#t0 = time.time()
	x, df_time = retail.clear(df_time=df_time)
	#t1 = time.time()
	#df_time.at[n,'clearing_time'] = t1-t0

df_time.to_csv('marketclearing_timetest_sep.csv')

import matplotlib.pyplot as ppt

fig = ppt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(df_time['sorting_time'],label='sorting time')
ax.plot(df_time['clearing_time'],label='clearing time')
ax.legend()
ppt.savefig('marketclearing_timetest_sep.png')