# Шаг 1: Импорт библиотек
import numpy as np
import argparse
from typing import Callable, Tuple


# Шаг 2: Генерация выборки
def generate_sample(size: int, scale: float = 1.0) -> np.ndarray:
	"""
	Генерирует выборку экспоненциального распределения.
	"""
	return np.random.exponential(scale=scale, size=size)

# Шаг 3: Функции для расчёта статистик
def mad(data: np.ndarray) -> float:
	"""
	Расчёт среднеквадратичного отклонения от медианы (MAD).
	"""
	return np.median(np.abs(data - np.median(data)))

def sample_variance(data: np.ndarray) -> float:
	"""
	Расчёт выборочной дисперсии.
	"""
	return np.var(data, ddof=1)

# Шаг 4: Бутстрап для доверительных интервалов
def bootstrap_confidence_interval(
		data: np.ndarray,
		statistic: Callable[[np.ndarray], float],
		num_samples: int = 1000,
		confidence_level: float = 0.95
	) -> Tuple[float, float]:
	"""
	Оценивает доверительный интервал для заданной статистики методом бутстрапа.
	"""
	bootstrap_statistics = []
	for _ in range(num_samples):
		sample = np.random.choice(data, size=len(data), replace=True)
		bootstrap_statistics.append(statistic(sample))
	
	lower_percentile = (1.0 - confidence_level) / 2.0
	upper_percentile = 1.0 - lower_percentile
	return (np.percentile(bootstrap_statistics, 100 * lower_percentile),
			np.percentile(bootstrap_statistics, 100 * upper_percentile))


# Шаг 5: Основной анализ с выводом результатов
def main(sample_size: int, confidence_level: float, num_bootstrap_samples):

	# Генерация выборки
	data = generate_sample(size=sample_size)

	# Оценка доверительных интервалов
	mad_confidence_interval = bootstrap_confidence_interval(data, mad, num_samples=num_bootstrap_samples, confidence_level=confidence_level)
	variance_confidence_interval = bootstrap_confidence_interval(data, sample_variance, num_samples=num_bootstrap_samples, confidence_level=confidence_level)
	
	mad_interval_length = mad_confidence_interval[1] - mad_confidence_interval[0]
	variance_interval_length = variance_confidence_interval[1] - variance_confidence_interval[0]

	# Вывод результатов
	print(f"95% доверительный интервал для MAD: {mad_confidence_interval}")
	print(f"95% доверительный интервал для выборочной дисперсии: {variance_confidence_interval}")

	# Сравнение результатов
	if mad_interval_length < variance_interval_length:
		print("Доверительный интервал для MAD короче, чем для дисперсии.")
	elif mad_interval_length > variance_interval_length:
		print("Доверительный интервал для дисперсии короче, чем для MAD.")
	else:
		print("Длины доверительных интервалов равны.")



if __name__ == "__main__":
	# Обработка аргументов командной строки.
	parser = argparse.ArgumentParser(description="Бутстрап-анализ для расчёта доверительных интервалов MAD и дисперсии.")

	parser.add_argument('--sample-size', type=int, default=10000, help="Размер выборки (по умолчанию 10000)")
	parser.add_argument('--confidence-level', type=float, default=0.95, help="Уровень доверия для интервалов (по умолчанию 0.95)")
	parser.add_argument('--num-bootstrap-samples', type=int, default=1000, help="Количество бутстрап-выборок (по умолчанию 1000)")
	args = parser.parse_args()
	#
	main(args.sample_size, args.confidence_level, args.num_bootstrap_samples)

	# Сравнение доверительных интервалов:
	# Поскольку дисперсия более чувствительна к выбросам,
	# её доверительный интервал, скорее всего, будет шире,
	#  чем у MAD, который, благодаря медиации,
	# оказывается более устойчивым к таким отклонениям.

