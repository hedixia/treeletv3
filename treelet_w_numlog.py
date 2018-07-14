import numpy as np


def lnexp (pow, pm):
	if pow > 20:
		return pow
	elif pow < 0.001 and not pm:
		return np.log(pow) + pow / 2 + pow * pow / 24
	else:
		return np.log(np.exp(pow) + (1 if pm else -1))


class num_log:
	def __init__ (self, num=None, pos=True):
		self.pos = pos
		self.num = num #num=None is 0

	def fromfloat (self, other):
		self.pos = (other >= 0)
		if other == 0:
			self.num = None
		else:
			self.num = np.log(abs(num))

	@property
	def iszero (self):
		return self.num == None

	def __add__ (self, other):
		if other.iszero:
			return num_log(self.num, self.pos)
		pm_val = self.pos ^ other.pos
		if self.num < other.num:
			new_pos = other.pos
			diffexp = other.num - self.num
		else:
			new_pos = self.pos
			diffexp = self.num - other.num
		return num_log(lnexp(diffexp, pm_val), new_pos)

	def __sub__ (self, other):
		return self + (- other)

	def __mul__ (self, other):
		if other.iszero:
			return num_log()
		return num_log(self.num + other.num, self.pos ^ other.pos)

	def __truediv__ (self, other):
		if other.iszero:
			raise ZeroDivisionError
		return num_log(self.num - other.num, self.pos ^ other.pos)

	def __pow__ (self, other):  # other is of type float
		return num_log(self.num * other, self.pos)

	def __neg__ (self):
		return num_log(self.num, not self.pos)

	def __lt__ (self, other):
		if self.pos ^ other.pos:
			if other.iszero:
				return False
			elif self.iszero:
				return True
			else:
				return (self.num < other.num) ^ self.pos
		else:
			return other.pos

	def __le__ (self, other):
		return not (self > other)

	def __eq__ (self, other):
		return (self.num == other.num) ^ (self.pos == other.pos)

	def __gt__ (self, other):
		if self.pos ^ other.pos:
			if self.iszero:
				return False
			elif other.iszero:
				return True
			else:
				return (self.num > other.num) ^ self.pos
		else:
			return self.pos

	def __ge__ (self, other):
		return not (self < other)

	def __abs__(self):
		return num_log(self.num, True)

	def sqrt(self):
		if self.iszero:
			return num_log()
		else:
			if not self.pos:
				raise Warning("square root of an negative number")
			return num_log(self.num/2)

	def __invert__(self): #get multiplicative inverse
		return num_log(-self.num, self.pos)

	def __float__ (self):
		if self.iszero:
			return 0
		else:
			return np.exp(self.num) * (1 if self.pos else -1)

	def __repr__ (self):
		if self.iszero:
			return repr(0)
		return ("+exp(" if self.pos else "-exp(") + str(self.num) + ")"

	def __bool__ (self):
		return self.iszero

	def __hash__ (self):
		return hash(self.num)




def jacobi_rotation_log (M, k, l, tol=num_log(20)):
	"""
	input: numpy matrix for rotation M, two different row number k and l 
	output: cos and sin value of rotation 
	change: M is inplace changed
	"""

	# rotation matrix calc
	if M[k, l] + M[l, k] < tol:
		cos_val = num_log(0) #1
		sin_val = num_log() #0
	else:
		b = (M[l, l] - M[k, k]) / (M[k, l] + M[l, k])
		tan_val = (num_log(0) if b.pos else num_log(0, False)) / (
					abs(b) + (b * b + num_log(0)).sqrt())  # |tan_val| < 1
		cos_val = ~(tan_val * tan_val + num_log(0)).sqrt() # cos_val > 0
		sin_val = cos_val * tan_val  # |cos_val| > |sin_val|

	# right multiplication by jacobian matrix
	temp1 = M[k, :] * cos_val - M[l, :] * sin_val
	temp2 = M[k, :] * sin_val + M[l, :] * cos_val
	M[k, :] = temp1
	M[l, :] = temp2

	# left multiplication by jacobian matrix transpose
	temp1 = M[:, k] * cos_val - M[:, l] * sin_val
	temp2 = M[:, k] * sin_val + M[:, l] * cos_val
	M[:, k] = temp1
	M[:, l] = temp2

	return (cos_val, sin_val)


class treelet:
	def __init__ (self, A, psi):
		self.A = np.matrix(A)
		self.phi = lambda x, y:psi(self.A[x, y], self.A[x, x], self.A[y, y])
		self.n = self.A.shape[0]
		self.max_row = {i:0 for i in range(self.n)}
		self.transform_list = []
		self.dendrogram_list = []

	# Treelet Tree
	def tree (self):
		return [I[0:2] for I in self.transform_list]

	def fullrotate (self):
		self.rotate(self.n - 1)
		self.root = list(self.max_row)[0]

	def rotate (self, multi=False):
		if multi:
			for i in range(multi):
				self.rotate()
			self.dfrk = [self.transform_list[i][1] for i in
			             range(self.n - 1)].append(self.transform_list[-1][0])
		else:
			(p, q) = self._find()
			(cos_val, sin_val) = jacobi_rotation(self.A, p, q)
			self._record(p, q, cos_val, sin_val)
			try:
				self.dendrogram_list.append((np.log(self.A[p, q]) * 2 - np.log(
					self.A[p, p]) - np.log(self.A[q, q]), p, q))
			except ZeroDivisionError:
				self.dendrogram_list.append(None)

	def _find (self):
		if self.transform_list == []:
			self.max_row_val = {}
			for i in self.max_row:
				self._max(i)
		else:
			(l, k, cos_val, sin_val) = self.current
			for i in self.max_row:
				if i == k or i == l:
					self._max(i)
				if self.phi(self.max_row[i], i) < self.phi(k, i):
					self.max_row[i] = k
				if self.phi(self.max_row[i], i) < self.phi(l, i):
					self.max_row[i] = l
				if self.max_row[i] == k or self.max_row[i] == l:
					self._max(i)
		v = list(self.max_row_val.values())
		k = list(self.max_row_val.keys())
		i = k[v.index(max(v))]
		return (self.max_row[i], i)

	def _max (self, col_num):
		temp_max_row = 0
		max_temp = 0
		for i in self.max_row:
			if i == col_num:
				continue
			temp = self.phi(i, col_num)
			if temp >= max_temp:
				temp_max_row = i
				max_temp = temp
		self.max_row[col_num] = temp_max_row
		self.max_row_val[col_num] = max_temp

	def _pair_gen (self):
		v = list(self.max_row_val.values())
		k = list(self.max_row_val.keys())
		i = k[v.index(max(v))]
		return (self.max_row[i], i)

	def _record (self, l, k, cos_val, sin_val, drop=True):
		if self.A[l, l] < self.A[k, k]:
			self.current = (k, l, cos_val, sin_val)
		else:
			self.current = (l, k, cos_val, sin_val)
		self.transform_list.append(self.current)
		del self.max_row[self.current[1]]
		del self.max_row_val[self.current[1]]
