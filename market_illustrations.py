import market_functions as Mfct
import numpy as np

retail = Mfct.Market()
retail.Pmin = 0.0
retail.Pmax = 100.0
retail.Pprec = 3
retail.Qlabel = 'Quantity in [MW]'
retail.Plabel = 'Price in [USD/MW]'

#Figure 2
retail.buy(1200.0, 100.0)
p = 100.0
for i in range(100):
	p -= np.random.uniform()*4
	q = np.random.uniform()*6
	if p < 0.0:
		break
	retail.buy(q, p)
retail.sell(1250.0, 15.0)

retail.clear()

retail.plot(save_name='clear_at_intersection.png')

#Figure 10
retail.reset()

retail.buy(1400.0,100.0)
p = 100.0
for i in range(100):
	p -= np.random.uniform()*8
	q = np.random.uniform()*6
	if p < 0.0:
		break
	retail.buy(q, p)
retail.sell(1250.0, 15.0)

retail.clear()

retail.plot(save_name='clear_at_maxp.png')