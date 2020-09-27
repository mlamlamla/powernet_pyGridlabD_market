import matplotlib.pyplot as ppt
ppt.rcParams['mathtext.fontset'] = 'stix'
ppt.rcParams['font.family'] = 'STIXGeneral'
import numpy as np
import pandas as pd

#Simulate theoretical results
def get_theta_no_dispatch(theta,theta_out,beta,t):
	return beta**t*theta + (1. - beta**t)*theta_out

def discounted_utility(theta,theta_out,theta_comf,alpha,beta,delta,t=0):
	U1 = alpha*(theta_out - theta_comf)**2*np.exp(np.log(delta)*t)/np.log(delta)
	U2 = -2*alpha*(theta_out - theta_comf)*(theta_out - theta)*np.exp(np.log(delta)*t)/(np.log(delta) + np.log(beta))
	U3 = alpha*(theta_out - theta)**2*np.exp(np.log(delta)*t)/(np.log(delta) + 2*np.log(beta))
	return U1 + U2 + U3

def discounted_utlity_dispatch(theta,theta_out,theta_comf,alpha,beta,delta,t_dispatch,price,gamma,P):
	U_beforedispatch = discounted_utility(theta,theta_out,theta_comf,alpha,beta,delta)
	theta_no_dispatch = beta**(t_dispatch + 1)*theta + (1. - beta**(t_dispatch + 1))*theta_out
	U_nodispatch = discounted_utility(theta_no_dispatch,theta_out,theta_comf,alpha,beta,delta,t_dispatch + 1)
	theta_after_dispatch = beta**(t_dispatch + 1)*theta + (1. - beta**(t_dispatch + 1))*theta_out - (1-beta)*gamma*P
	U_afterdispatch = discounted_utility(theta_after_dispatch,theta_out,theta_comf,alpha,beta,delta,t_dispatch + 1)
	U = U_beforedispatch - U_nodispatch - delta**t_dispatch*price + U_afterdispatch
	return U

theta_out = 80
theta_comf= 70

gamma = 10.
P = 4
beta = 0.98

alpha = 0.02
delta = 0.97

#Sanity check: Utility descreases with theta increasing
#for theta in range(70,81):
#	print(discounted_utility(theta,theta_out,theta_comf,alpha,beta,delta))

#Utility of later dispatch
theta = 70

price_list = [0.05]*15 + [0.025]*46
delta_list = [0.995,0.99,0.98,0.97]
#delta_list = np.arange(0.9,0.996,0.001)
max_t_list = []

df_u = pd.DataFrame(index=range(60),columns=[str(d) for d in delta_list],data=0.0)

for delta in delta_list:
	U0 = discounted_utlity_dispatch(theta,theta_out,theta_comf,alpha,beta,delta,0,price_list[0],gamma,P)
	max_u = 0.0
	max_t = 0
	for t_dispatch in range(61):
		price = price_list[t_dispatch]
		perc_change = -100*(discounted_utlity_dispatch(theta,theta_out,theta_comf,alpha,beta,delta,t_dispatch,price,gamma,P)-U0)/U0
		df_u[str(delta)].loc[t_dispatch] = perc_change
		if perc_change > max_u:
			max_u = perc_change
			max_t = t_dispatch
		#print(perc_change)

	print()
	print(max_t)
	max_t_list += [max_t]

#import pdb; pdb.set_trace()
ls = ['-','--',':','-.','.']
fig = ppt.figure(figsize=(6,3),dpi=150)   
ax = fig.add_subplot(111)
lns = []
i = 0
for delta in delta_list:
	lns += ax.plot(df_u[str(delta)],label='$\\delta$ = '+str(delta),color='0.5',ls=ls[i])
	i += 1
ax.set_xlabel('Time [min]')
ax.set_ylabel('Utility change to immediate dispatch [%]')
ax.set_xlim(xmin=0,xmax=60)

ax.hlines(0,0,60,'k')

labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

#ppt.savefig('Diss/Diss_0003/temperature_offset.pdf', bbox_inches='tight')
ppt.savefig('utility_bydelta.pdf', bbox_inches='tight')
ppt.close()

df_T = pd.DataFrame(index=range(61),columns=[str(d) for d in delta_list],data=0.0)
df_T_hvac = pd.DataFrame(index=delta_list,columns=['T_at_dispatch'],data=0.0)

i = 0

for delta in delta_list:
	max_t = max_t_list[i]
	for t in range(max_t + 1):
		df_T[str(delta)].loc[t] = get_theta_no_dispatch(theta,theta_out,beta,t)
	df_T_hvac['T_at_dispatch'].loc[delta] = df_T[str(delta)].loc[t]
	theta_dis = beta*df_T[str(delta)].loc[t] + (1 - beta)*(theta_out - gamma*P)
	for t in range(max_t + 1,61):
		df_T[str(delta)].loc[t] = get_theta_no_dispatch(theta_dis,theta_out,beta,t - (max_t+1))
	i += 1

print(df_T_hvac)

fig = ppt.figure(figsize=(6,3),dpi=150)   
ax = fig.add_subplot(111)
lns = ax.plot(df_T_hvac['T_at_dispatch'],color='0.5')
ax.set_xlabel('Discounting factor $\\delta$')
ax.set_ylabel('Room temperature $\\theta_t$\n at which HVAC dispatches [defF]')
ax.set_xlim(xmin=min(delta_list),xmax=max(delta_list))
ax.text(0.985, 70.1, '$\\theta^{comf}$')
ppt.xticks(np.arange(0.9,0.995,0.02).tolist()+[0.995])

ax.hlines(theta_comf,0,60,'k',label='Comfort temperature')

# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc=0)

#ppt.savefig('Diss/Diss_0003/temperature_offset.pdf', bbox_inches='tight')
#ppt.savefig('optT_bydelta.png', bbox_inches='tight')
#ppt.savefig('optT_bydelta.pdf', bbox_inches='tight')

# fig = ppt.figure(figsize=(6,3),dpi=150)   
# ax = fig.add_subplot(111)
# lns = []
# i = 0
# for delta in delta_list:
# 	lns += ax.plot(df_T[str(delta)],label='$\\delta$ = '+str(delta),color='0.5',ls=ls[i])
# 	i += 1
# ax.set_xlabel('Time [min]')
# ax.set_ylabel('Room temperature [defF]')
# ax.set_xlim(xmin=0,xmax=60)

# lns += ax.hlines(T_comf,0,60,'k',label='Comfort temperature')

# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc=0)

# #ppt.savefig('Diss/Diss_0003/temperature_offset.pdf', bbox_inches='tight')
# ppt.savefig('optT_bydelta.pdf', bbox_inches='tight')

