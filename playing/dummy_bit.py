#if dummy:
		#	n_size = dict(zip(list(map(str, range(0,len(size)))),np.exp(size)))
		#	sort_size = sorted(n_size.items(), key=operator.itemgetter(1))
		#	size_range = sorted(size)
		#	pdf_x = self.pdf_workers(self.knots)        	# calculates pdf of xs in one step
		#	n_pdf_x = dict(enumerate(pdf_x)) 			# creates a dictionary where the keys are the #obs of x
		#	wgts = np.empty(0)
		#	for pair in sort_size:
		#		index = int(pair[0])
		#		wgts  = np.hstack((wgts ,(n_pdf_x[index]/pair[1])))		#weights contain the number of firms of this size
#
#			n_wage = dict(enumerate(wage))
#			n_profit = dict(enumerate(profit))
#			wage_range = np.empty(0)
#			profit_range = np.empty(0)
#			for pair in sort_size:
#				index = int(pair[0])
#				wage_range = np.hstack((wage_range,n_wage[index]))		#sort the data by size
#				profit_range = np.hstack((profit_range,n_profit[index]))
			#cdf_theta_hat  = np.cumsum(pdf_theta_hat )			# Backing up model cdf
			#cdf_theta_hat  = cdf_theta_hat /cdf_theta_hat [-1] 	# Normilization of the model cdf