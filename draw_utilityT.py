import matplotlib.pyplot as ppt
ppt.rcParams['mathtext.fontset'] = 'stix'
ppt.rcParams['font.family'] = 'STIXGeneral'
import numpy as np

fig = ppt.figure(figsize=(6,4),dpi=150)   
ax = fig.add_subplot(111)

theta = np.arange(65,75,0.01)

lns1 = ax.plot(theta,-(theta-70)**2,color='k')
#lns2 = ax.plot(theta,-4.*(theta-70.) + 4,color='0.5')
#lns3 = ax.plot(theta,4.*(theta-70.) + 4,color='0.5')

ax.set_xlabel('Temperature $\\theta$  ') #, horizontalalignment='right') #,x=1.)
ax.set_ylabel('Utility [USD]')
ax.set_xlim(xmin=66,xmax=74)
ax.set_ylim(ymin=-17.5,ymax=1.)

ppt.text(70,0.5,'$\\theta^{com}$', horizontalalignment='center')
ax.vlines(68,-4,0,'0.25',linestyles='dashed')
ppt.text(68,0.5,'$\\theta^{heat}$', horizontalalignment='center')
#ppt.text(66.5,-5,'$\\frac{\\lambda}{(1-\\beta)\\gamma}$',fontsize=16)
ax.vlines(72,-4,0,'0.25',linestyles='dashed')
ppt.text(72,0.5,'$\\theta^{cool}$', horizontalalignment='center')
#ppt.text(72.5,-5,'$-\\frac{\\lambda}{(1-\\beta)\\gamma}$',horizontalalignment='left',fontsize=16)

#ax.set_aspect('equal')
ax.grid(True, which='both')

# set the x-spine (see below for more info on `set_position`)
# ax.spines['left'].set_position('zero')

# # turn off the right spine/ticks
# ax.spines['right'].set_color('none')
# ax.yaxis.tick_left()

# set the y-spine
ax.spines['bottom'].set_position('zero')

# turn off the top spine/ticks
ax.spines['top'].set_color('none')
ax.xaxis.tick_top()
ax.tick_params(axis='x', color='0.75')
ax.tick_params(axis='y', color='1.0')
ax.set_xticklabels([])
ax.set_yticklabels([])

label = ax.xaxis.get_label()
x_lab_pos, y_lab_pos = label.get_position()
label.set_position([1.0, y_lab_pos])
label.set_horizontalalignment('right')

#myFmt = DateFormatter("%d")
#ax.xaxis.set_major_locator(HourLocator(interval = 1))

#ppt.savefig('Diss/Diss_0003/temperature_offset.pdf', bbox_inches='tight')
ppt.savefig('utilityT.pdf', bbox_inches='tight')