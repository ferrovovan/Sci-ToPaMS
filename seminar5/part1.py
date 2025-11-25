# Задание №1
# Генерация случайных величин - функциональное ядро с ООП-обертками

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Union, Tuple, List
import math
from scipy import stats


# ===============
# ЧИСТЫЕ ФУНКЦИИ - ЯДРО СИСТЕМЫ
# ===============

def bp(p: float) -> int:
	"""Распределение Бернулли"""
	return 1 if np.random.uniform(0, 1) < p else 0


def bnp(n: int, p: float) -> int:
	"""Биномиальное распределение"""
	# return sum([bp(p) for _ in range(n)])  # очень медленно
	return sum(np.random.uniform(0, 1, n) < p)


def geometric_discrete(p: float) -> float:
	"""Дискретное геометрическое распределение"""
	return np.floor(np.log(1 - np.random.uniform(0, 1)) / np.log(1 - p))


def geometric_continuous(p: float) -> float:
	"""Непрерывное геометрическое распределение"""
	return np.log(1 - np.random.uniform(0, 1)) / np.log(1 - p) - 1


def poisson(lam: float) -> float:
	"""Распределение Пуассона с параметром λ"""
	# Алгоритм Кнута для генерации Пуассона
	L = np.exp(-lam)
	k = 0
	p = 1.0
	while p > L:
		k += 1
		p *= np.random.uniform(0, 1)
	return k - 1


def uniform(a: float, b: float) -> float:
	"""Равномерное распределение на отрезке [a, b]"""
	return np.random.uniform(a, b)  # Используем встроенную функцию для точности
	# return np.random.uniform(0, 1) * (b - a) + a


def exponential(alpha: float) -> float:
	"""Экспоненциальное распределение"""
	return -np.log(1 - np.random.uniform(0, 1)) / alpha


def laplace(alpha: float, mu: float = 0) -> float:
	"""Распределение Лапласа (двойное экспоненциальное)"""
	u = np.random.uniform(0, 1)
	if u < 0.5:
		return mu - alpha * np.log(2 * u)
	else:
		return mu + alpha * np.log(2 * (1 - u))


def normal_box_muller(alpha: float, sigma: float) -> Tuple[float, float]:
	"""
	Нормальное распределение через алгоритм Бокса-Мюллера.
	Возвращает 2 значения!
	"""
	u1 = np.random.uniform(0, 1)
	u2 = np.random.uniform(0, 1)
	r = np.sqrt(-2 * np.log(u1))
	theta = 2 * np.pi * u2
	
	z1 = r * np.sin(theta)
	z2 = r * np.cos(theta)
	
	return sigma * z1 + alpha, sigma * z2 + alpha


def normal_clt(alpha: float, sigma: float, n: int = 12) -> float:
	"""
	Нормальное распределение через Центральную Предельную Теорему
	"""
	u = [np.random.uniform(0, 1) for _ in range(n)]
	eu = n * 0.5  # Матожидание суммы n U[0,1]
	du = n / 12   # Дисперсия суммы n U[0,1]
	z = (sum(u) - eu) / np.sqrt(du)
	
	return sigma * z + alpha


def cauchy(x0: float, gamma: float) -> float:
	"""Распределение Коши"""
	return x0 + gamma * np.tan(np.pi * (np.random.uniform(0, 1) - 0.5))


def custom_distribution() -> float:
	"""Пользовательское распределение с ядром f(t) = 1/t^3 I{t > 1}"""
	return 1 / np.sqrt(1 - np.random.uniform(0, 1))


# ===============
# КЛАССЫ-ОБЕРТКИ
# ===============

class Distribution(ABC):
	"""Абстрактный базовый класс для распределений"""
	
	def __init__(self, name: str, is_continuous: bool):
		self.name = name
		self.is_continuous = is_continuous
	
	@abstractmethod
	def generate(self) -> Union[float, int]:
		"""Генерирует одно случайное значение из распределения"""
		pass
	
	def generate_sample(self, n: int) -> List[Union[float, int]]:
		"""Генерирует выборку из n случайных значений"""
		return [self.generate() for _ in range(n)]

	def theoretical_pdf(self, x: np.ndarray) -> np.ndarray:
		"""Теоретическая функция плотности (для непрерывных)"""
		raise NotImplementedError("PDF not implemented for this distribution")
	    
	def theoretical_pmf(self, x: np.ndarray) -> np.ndarray:
		"""Теоретическая функция вероятности (для дискретных)"""
		raise NotImplementedError("PMF not implemented for this distribution")


class Bernoulli(Distribution):
	"""Распределение Бернулли"""
	
	def __init__(self, p: float):
		super().__init__("Bernoulli", False)
		self.p = p
	
	def generate(self) -> int:
		return bp(self.p)
	
	def theoretical_pmf(self, x: np.ndarray) -> np.ndarray:
		result = np.zeros_like(x, dtype=float)
		result[x == 0] = 1 - self.p
		result[x == 1] = self.p
		return result


class Binomial(Distribution):
	"""Биномиальное распределение"""
	
	def __init__(self, n: int, p: float):
		super().__init__("Binomial", False)
		self.n = n
		self.p = p
	
	def generate(self) -> int:
		return bnp(self.n, self.p)
	
	def theoretical_pmf(self, x: np.ndarray) -> np.ndarray:
		# Используем scipy для точного расчета биномиальных вероятностей
		return stats.binom.pmf(x, self.n, self.p)


class GeometricDiscrete(Distribution):
	"""Дискретное геометрическое распределение"""
	
	def __init__(self, p: float):
		super().__init__("Geometric Discrete", False)
		self.p = p
	
	def generate(self) -> float:
		return geometric_discrete(self.p)
	
	def theoretical_pmf(self, x: np.ndarray) -> np.ndarray:
		# Геометрическое распределение: P(X=k) = (1-p)^k * p
		k = np.floor(x)
		return (1 - self.p) ** k * self.p


class GeometricContinuous(Distribution):
	"""Непрерывное геометрическое распределение"""
	
	def __init__(self, p: float):
		super().__init__("Geometric Continuous", True)
		self.p = p
	
	def generate(self) -> float:
		return geometric_continuous(self.p)


class Poisson(Distribution):
	"""Распределение Пуассона"""
	
	def __init__(self, lam: float):
		super().__init__("Poisson", False)
		self.lam = lam
	
	def generate(self) -> float:
		return poisson(self.lam)
	
	def theoretical_pmf(self, x: np.ndarray) -> np.ndarray:
		return stats.poisson.pmf(x, self.lam)


class Uniform(Distribution):
	"""Равномерное распределение на отрезке [a, b]"""
	
	def __init__(self, a: float, b: float):
		super().__init__("Uniform", True)
		self.a = a
		self.b = b
	
	def generate(self) -> float:
		return uniform(self.a, self.b)
	
	def theoretical_pdf(self, x: np.ndarray) -> np.ndarray:
		result = np.zeros_like(x, dtype=float)
		mask = (x >= self.a) & (x <= self.b)
		result[mask] = 1.0 / (self.b - self.a)
		return result


class Exponential(Distribution):
	"""Экспоненциальное распределение"""
	
	def __init__(self, alpha: float):
		super().__init__("Exponential", True)
		self.alpha = alpha
	
	def generate(self) -> float:
		return exponential(self.alpha)

	def theoretical_pdf(self, x: np.ndarray) -> np.ndarray:
		result = np.zeros_like(x, dtype=float)
		mask = x >= 0
		result[mask] = self.alpha * np.exp(-self.alpha * x[mask])
		return result


class Laplace(Distribution):
	"""Распределение Лапласа"""
	
	def __init__(self, alpha: float, mu: float = 0):
		super().__init__("Laplace", True)
		self.alpha = alpha
		self.mu = mu
	
	def generate(self) -> float:
		return laplace(self.alpha, self.mu)
	
	def theoretical_pdf(self, x: np.ndarray) -> np.ndarray:
		return (1 / (2 * self.alpha)) * np.exp(-np.abs(x - self.mu) / self.alpha)


class NormalBoxMuller(Distribution):
	"""Нормальное распределение через алгоритм Бокса-Мюллера"""
	
	def __init__(self, alpha: float, sigma: float):
		super().__init__("Normal (Box-Muller)", True)
		self.alpha = alpha
		self.sigma = sigma
		self._next_value = None  # Для хранения второго значения
	
	def generate(self) -> float:
		if self._next_value is not None:
			value = self._next_value
			self._next_value = None
			return value
		
		z1, z2 = normal_box_muller(self.alpha, self.sigma)
		self._next_value = z2
		return z1
	
	def theoretical_pdf(self, x: np.ndarray) -> np.ndarray:
		return (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - self.alpha) / self.sigma) ** 2)


class NormalCLT(Distribution):
	"""Нормальное распределение через ЦПТ"""
	
	def __init__(self, alpha: float, sigma: float, n: int = 12):
		super().__init__("Normal (CLT)", True)
		self.alpha = alpha
		self.sigma = sigma
		self.n = n
	
	def generate(self) -> float:
		return normal_clt(self.alpha, self.sigma, self.n)
	
	def theoretical_pdf(self, x: np.ndarray) -> np.ndarray:
		return (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - self.alpha) / self.sigma) ** 2)


class Cauchy(Distribution):
	"""Распределение Коши"""
	
	def __init__(self, x0: float, gamma: float):
		super().__init__("Cauchy", True)
		self.x0 = x0
		self.gamma = gamma
	
	def generate(self) -> float:
		return cauchy(self.x0, self.gamma)
	
	def theoretical_pdf(self, x: np.ndarray) -> np.ndarray:
		return 1 / (np.pi * self.gamma * (1 + ((x - self.x0) / self.gamma) ** 2))


class CustomDistribution(Distribution):
	"""Пользовательское распределение с ядром f(t) = 1/t^3 I{t > 1}"""
	
	def __init__(self):
		super().__init__("Custom f(t) = 1/t^3 I{t > 1}", True)
	
	def generate(self) -> float:
		return custom_distribution()
	
	def theoretical_pdf(self, x: np.ndarray) -> np.ndarray:
		result = np.zeros_like(x, dtype=float)
		mask = x > 1
		result[mask] = 1 / (x[mask] ** 3)
		# Нормализуем - интеграл от 1 до ∞ от 1/t^3 dt = 1/2
		# Поэтому плотность должна быть 2/t^3
		result[mask] *= 2
		return result


# ===============
# ВИЗУАЛИЗАЦИЯ
# ===============

def plot_distributions(distributions: List[Distribution], samples: List[List[Union[float, int]]]):
	"""
	Строит гистограммы для выборок и сравнивает с теоретическими распределениями
	"""
	n_distributions = len(distributions)
	n_cols = 2
	n_rows = (n_distributions + 1) // n_cols
	
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(26, 6 * n_rows))
	axes = axes.flatten() if n_distributions > 1 else [axes]
	
	for i, (dist, sample) in enumerate(zip(distributions, samples)):
		ax = axes[i]
		sample_array = np.array(sample)
		
		# Определяем тип распределения и строим соответствующий график
		if dist.is_continuous:
			# Непрерывное распределение - гистограмма и плотность
			_plot_continuous_distribution(ax, dist, sample_array)
		else:
			# Дискретное распределение - столбчатая диаграмма и PMF
			_plot_discrete_distribution(ax, dist, sample_array)
		
		ax.set_title(f'{dist.name}', fontsize=12, pad=0)
		ax.legend(fontsize=4)
		ax.grid(True, alpha=0.3)
	
	# Скрываем пустые subplots
	for j in range(i + 1, len(axes)):
		axes[j].set_visible(False)
	
	plt.tight_layout(pad=5.0, h_pad=13.0, w_pad=0.0)
	plt.show()


def _plot_continuous_distribution(ax, dist, sample):
	"""Визуализация для непрерывных распределений"""
	sample_array = np.array(sample)
	
	hist_density = True  # Нормируем гистограмму к плотности
	if dist.name == "Cauchy":
		# Для Коши обрезаем выбросы для лучшего отображения
		#q_low, q_high = np.quantile(sample_array, [0.05, 0.95])
		#sample_filtered = sample_array[(sample_array >= q_low) & (sample_array <= q_high)]
		#n_bins = 20
		#custom_hist_range = (q_low, q_high)
		
		x0 = getattr(dist, 'x0', 0)
		gamma = getattr(dist, 'gamma', 1)
		n_bins = 50
		hist_range = (x0 - 5*gamma, x0 + 5*gamma)  # Фиксированный диапазон ±5γ
	elif dist.name == "Custom f(t) = 1/t^3 I{t > 1}":
		# Для кастомного распределения ограничиваем диапазон
		n_bins = 50
		hist_range = (0, 7)  # Фиксированный диапазон от 1 до 7
	else:
		# Гистограмма выборки
		n_bins = min(50, len(sample) // 10)  # Автоматический подбор числа бинов
		hist_range = None
		
	
	ax.hist(sample, bins=n_bins, density=hist_density, alpha=0.7, 
			color='skyblue', edgecolor='black', label='Выборка', range=hist_range)
	
	# Теоретическая плотность
	if hasattr(dist, 'theoretical_pdf'):
		if dist.name == "Cauchy":
			x_plot = np.linspace(hist_range[0], hist_range[1], 1000)
		elif dist.name == "Custom f(t) = 1/t^3 I{t > 1}":
			x_plot = np.linspace(hist_range[0], hist_range[1], 1000)
		else:
			x_min, x_max = np.min(sample), np.max(sample)
			x_range = x_max - x_min
			x_plot = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 1000)
		
		try:
			pdf_values = dist.theoretical_pdf(x_plot)
			ax.plot(x_plot, pdf_values, 'r-', linewidth=2, label='Теоретическая PDF')
		except (NotImplementedError, ValueError):
			pass


def _plot_discrete_distribution(ax, dist, sample):
	"""Визуализация для дискретных распределений"""
	# Эмпирические частоты
	unique, counts = np.unique(sample, return_counts=True)
	empirical_probs = counts / len(sample)
	
	# Специальная обработка для Бернулли
	if dist.name == "Bernoulli":
		ax.bar(unique, empirical_probs, alpha=0.7, color='lightgreen', 
			edgecolor='black', label='Выборка', width=0.3)
	else:
		ax.bar(unique, empirical_probs, alpha=0.7, color='lightgreen', 
			edgecolor='black', label='Выборка', width=0.8)
	
	# Теоретические вероятности
	if hasattr(dist, 'theoretical_pmf'):
		# Берем диапазон значений для теоретического PMF
		if dist.name == "Bernoulli":
			x_theoretical = np.array([0, 1])
		else:
			x_min, x_max = int(np.min(sample)), int(np.max(sample))
			x_theoretical = np.arange(max(0, x_min - 2), x_max + 3)  # +3 для красивого отображения
		
		try:
			pmf_values = dist.theoretical_pmf(x_theoretical)
			ax.plot(x_theoretical, pmf_values, 'ro-', linewidth=2, 
				markersize=4, label='Теоретическая PMF')
		except (NotImplementedError, ValueError):
			pass


def analyze_distributions(distributions: List[Distribution], samples: List[List[Union[float, int]]]):
	"""
	Полный анализ: генерация + визуализация + статистики
	"""
	print("=" * 30)
	print("АНАЛИЗ РАСПРЕДЕЛЕНИЙ")
	print("=" * 30)
	
	# Выводим основные статистики
	for i, (dist, sample) in enumerate(zip(distributions, samples)):
		print(f"\n{dist.name}:")
		print(f"  Размер выборки: {len(sample)}")
		print(f"  Среднее: {np.mean(sample):.4f}")
		print(f"  Стандартное отклонение: {np.std(sample):.4f}")
		if hasattr(dist, 'is_continuous') and dist.is_continuous:
			print(f"  Тип: непрерывное")
		else:
			print(f"  Тип: дискретное")
	
	# Строим графики
	plot_distributions(distributions, samples)


# ===============
# ДЕМОНСТРАЦИЯ РАБОТЫ
# ===============

def main():
	"""Основная функция для демонстрации работы"""
	
	# Создаем экземпляры распределений
	distributions = [
		Bernoulli(0.7),
		Binomial(1000, 0.7),
		GeometricDiscrete(0.7),
		Poisson(7),
		Uniform(0, 1),
		Exponential(0.8),
		Laplace(0.8),
		NormalBoxMuller(5, 2),
		Cauchy(0.5, 0.2),
		CustomDistribution()
	]
	
	# Генерируем выборки
	N = 100000
	samples = []
	
	print("Generating random samples...\n")
	for i, dist in enumerate(distributions):
		print(f"Generating {dist.name}...")
		
		sample = dist.generate_sample(N)
		samples.append(sample)
		
		# Прогресс
		#percent = (i + 1) / len(distributions) * 100
		#print(f"\r\033[KProgress: {percent:.1f}%\n", end='', flush=True)
	
	print("\nRandom generation completed!\n")
	
	# Анализ и визуализация
	analyze_distributions(distributions, samples)
	
	return distributions, samples


if __name__ == "__main__":
	distributions, samples = main()

