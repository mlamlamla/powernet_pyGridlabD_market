import gridlabd

for i in range(5):
	print(i)
	gridlabd.command('pv_calibration.glm')
	gridlabd.command('-D')
	gridlabd.command('suppress_repeat_messages=FALSE')
	gridlabd.command('--warn')
	gridlabd.start('wait')