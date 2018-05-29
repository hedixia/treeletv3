import numpy as np 
from math import sqrt, log

def jacobi_rotation (M, k, l, tol=0.00000000001):
	"""
	input: numpy matrix for rotation M, two different row number k and l 
	output: cos and sin value of rotation 
	change: M is inplace changed
	"""
	
	#rotation matrix calc
	if M[k,l] + M[l,k] < tol:
		cos_val = 1
		sin_val = 0 
	else:
		b = (M[l,l] - M[k,k]) / (M[k,l] + M[l,k])
		tan_val = (1 if b>=0 else -1) / (abs(b) + sqrt(b * b + 1)) # |tan_val| < 1
		cos_val = 1 / (sqrt(tan_val * tan_val + 1)) # cos_val > 0 
		sin_val = cos_val * tan_val # |cos_val| > |sin_val|
	
	#right multiplication by jacobian matrix
	temp1 = M[k,:] * cos_val - M[l,:] * sin_val 
	temp2 = M[k,:] * sin_val + M[l,:] * cos_val
	M[k,:] = temp1 
	M[l,:] = temp2 
	
	#left multiplication by jacobian matrix transpose
	temp1 = M[:,k] * cos_val - M[:,l] * sin_val 
	temp2 = M[:,k] * sin_val + M[:,l] * cos_val
	M[:,k] = temp1 
	M[:,l] = temp2 

	return (cos_val, sin_val)