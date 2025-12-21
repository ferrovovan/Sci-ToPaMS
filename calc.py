
# Скрипт для подсчётов при решении мини-контрольных

import math

def poisson_sum(end=11, λ=10):
	"""
	sum of  (λ^k / k!)
	"""
	from math import factorial, exp

	s = 0
	for k in range(1, end):
		s += (λ ** k) / factorial(k)
	return s * exp(-λ)


# Цешка из 52 по 6
# result = math.comb(52, 6)
