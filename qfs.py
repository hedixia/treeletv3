import random
#turn a list to a dict from val -> [entries]
def l2dl (some_list):
	temp_dict = {}
	for i in range(len(some_list)):
		temp_dict.setdefault(some_list[i], []).append(i)
	return temp_dict
	
#reverse l2dl 
def rl2dl (some_dict):
	n = 0
	for i in somedict:
		n += len(somedict[i])
	temp_list = [None] * n 
	for i in somedict:
		for j in somedict[i]:
			temp_list[j] = i 
	return temp_list
	
#sample one element from each cluster while ignoring the clusters with number less than or equal to ignore 
def dlsamp (some_dict_list, ignore=0):
	temp_samp = []
	for i in some_dict_list:
		if len(some_dict_list[i]) > ignore:
			temp_samp.append(random.sample(some_dict_list[i], 1)[0])
	return temp_samp
	
def index (error, noe1, noe2):
	return error - 1/(noe1 + noe2)
	
def compare(label_1, label_2):
	comparelist = (np.array(label_1) == np.array(label_2))
	return np.sum(comparelist)/len(label_1)
	
def argmax (L, f): #return $x\in L$ such that $f(x) = max \ f(L)$
	maxarg = None 
	maxval = 0
	for x in L:
		if maxarg == None:
			maxarg = x 
			maxval = f(x)
		if f(x) > maxval:
			maxarg = x 
			maxval = f(x)
	return maxarg
	
def lapply (L, f):
	return [f(x) for x in L]