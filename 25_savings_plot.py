import numpy as np
import pandas as pd
import matplotlib.pyplot as ppt
from sklearn.linear_model import LinearRegression
from numpy.polynomial.polynomial import polyfit

# Default
run = 'Diss'
ind_b = 90
folder_WS = run

df_results = pd.read_csv(run + '/' + 'weekly_welfare_changes.csv',index_col=[0])

reg = LinearRegression()
reg.fit(df_results['RR'].to_numpy().reshape(len(df_results),1),(df_results['av_uchange']).to_numpy().reshape(len(df_results),1))
reg.coef_

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(df_results['RR'],df_results['av_uchange'],marker='x',color='0.6')
#import pdb; pdb.set_trace()
lns2 = ppt.plot(df_results['RR'],reg.intercept_[0] + reg.coef_[0][0]*df_results['RR'],'-',color='0.25')
ax.set_xlabel('Average procurement cost [USD/MWh]')
#ax.set_xlim(left=0.0)
#ax.set_ylim(bottom=0.0)
ax.set_ylabel('Average household welfare change [USD]')
ppt.savefig(folder_WS + '/uchange_RR.png', bbox_inches='tight')
ppt.savefig(folder_WS + '/uchange_RR.pdf', bbox_inches='tight')

print(str(reg.intercept_[0]) + ' + ' + str(reg.coef_[0][0]) + '* RR')

# For all households / procurement cost savings

regb = LinearRegression()
regb.fit(df_results['RR'].to_numpy().reshape(len(df_results),1),(df_results['u_change']).to_numpy().reshape(len(df_results),1))
regb.coef_

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(df_results['RR'],df_results['u_change'],marker='x',color='0.6')
#import pdb; pdb.set_trace()
lns2 = ppt.plot(df_results['RR'],regb.intercept_[0] + regb.coef_[0][0]*df_results['RR'],'-',color='0.25')
ax.set_xlabel('Average procurement cost [USD/MWh]')
#ax.set_xlim(left=0.0)
#ax.set_ylim(bottom=0.0)
ax.set_ylabel('Procurement cost savings [USD]')
ppt.savefig(folder_WS + '/uchange_procc.png', bbox_inches='tight')
ppt.savefig(folder_WS + '/uchange_procc.pdf', bbox_inches='tight')

print(str(reg.intercept_[0]) + ' + ' + str(reg.coef_[0][0]) + '* RR')


###############

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(df_results['max_p'],df_results['av_uchange'],marker='x',color='0.6')
ax.set_xlabel('Maximum price [USD/MWh]')
ax.set_xlim(left=0.0)
ax.set_ylabel('Average household welfare change [USD]')
#ax.set_ylim(bottom=0.0)
ppt.savefig(folder_WS + '/uchange_maxp.png', bbox_inches='tight')
ppt.savefig(folder_WS + '/uchange_maxp.pdf', bbox_inches='tight')

reg2 = LinearRegression()
reg2.fit(df_results['max_p'].to_numpy().reshape(len(df_results),1),(df_results['av_uchange']).to_numpy().reshape(len(df_results),1))
reg2.coef_

###############

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(df_results['var_p'],df_results['av_uchange'],marker='x',color='0.6')
ax.set_xlabel('Price variance $[USD^2 / MWh^2 ]$')
ax.set_xlim(left=0.0)
#ax.set_ylim(bottom=0.0)
ax.set_ylabel('Average household welfare change [USD]')
ppt.savefig(folder_WS + '/uchange_varp.png', bbox_inches='tight')
ppt.savefig(folder_WS + '/uchange_varp.pdf', bbox_inches='tight')

reg3 = LinearRegression()
reg3.fit(df_results['var_p'].to_numpy().reshape(len(df_results),1),(df_results['av_uchange']).to_numpy().reshape(len(df_results),1))
reg3.coef_

print(df_results['var_p'].corr(df_results['RR']))

###############

reg4 = LinearRegression()
reg4.fit((np.sqrt(df_results['var_p'])).to_numpy().reshape(len(df_results),1),(df_results['av_uchange']).to_numpy().reshape(len(df_results),1))
reg4.coef_

fig = ppt.figure(figsize=(6,4),dpi=150)   
ppt.ioff()
ax = fig.add_subplot(111)
lns = ppt.scatter(np.sqrt(df_results['var_p']),df_results['av_uchange'],marker='x',color='0.6')
lns2 = ppt.plot(np.sqrt(df_results['var_p']),reg4.intercept_[0] + reg4.coef_[0][0]*np.sqrt(df_results['var_p']),'-',color='0.25')
ax.set_xlabel('Price standard error $[USD / MWh ]$')
ax.set_xlim(left=0.0)
#ax.set_ylim(bottom=0.0)
ax.set_ylabel('Average household welfare change [USD]')
ppt.savefig(folder_WS + '/uchange_stdp.png', bbox_inches='tight')
ppt.savefig(folder_WS + '/uchange_stdp.pdf', bbox_inches='tight')

print(np.sqrt(df_results['var_p']).corr(df_results['RR']))

import pdb; pdb.set_trace()
