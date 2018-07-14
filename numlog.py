import numpy as np


def lnexp (pow, pm):
	if pow > 20:
		return pow
	elif pow < 0.001 and not pm:
		return np.log(pow) + pow / 2 + pow * pow / 24
	else:
		return np.log(np.exp(pow) + (1 if pm else -1))


class numlog:
	def __init__ (self, num=None, pos=True):
		self.pos = pos
		self.num = num  # num=None is 0

	def fromfloat (self, other):
		self.pos = (other >= 0)
		if other == 0:
			self.num = None
		else:
			self.num = np.log(abs(num))

	@property
	def iszero (self):
		return self.num is None

	def __add__ (self, other):
		if other.iszero:
			return numlog(self.num, self.pos)
		if self.iszero:
			return numlog(other.num, other.pos)
		pm_val = self.pos ^ other.pos
		if self.num < other.num:
			new_pos = other.pos
			diffexp = other.num - self.num
		else:
			new_pos = self.pos
			diffexp = self.num - other.num
		return numlog(lnexp(diffexp, pm_val), new_pos)

	def __sub__ (self, other):
		return self + (- other)

	def __mul__ (self, other):
		if other.iszero:
			return numlog()
		return numlog(self.num + other.num, self.pos ^ other.pos)

	def __truediv__ (self, other):
		if other.iszero:
			raise ZeroDivisionError
		return numlog(self.num - other.num, self.pos ^ other.pos)

	def __pow__ (self, other):  # other is of type float
		return numlog(self.num * other, self.pos)

	def __neg__ (self):
		return numlog(self.num, not self.pos)

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

	def __abs__ (self):
		return numlog(self.num, True)

	def sqrt (self):
		if self.iszero:
			return numlog()
		else:
			return numlog(self.num / 2)

	def ln (self): #This returns a float
		if self.iszero:
			raise Warning("ln of 0")
			return
		return self.num

	def __invert__ (self):  # get multiplicative inverse
		return numlog(-self.num, self.pos)

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
