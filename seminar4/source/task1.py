import numpy as np
from plot_histogram_func import plot_histogram
from parse_args_func import get_args


# Основной алгоритм
def poisson_bernoulli_simulation(n: int, λ: float, N: int) -> (list, list):
	p: float = min(λ / n, 1)
	successes_per_trial = []
	distances_between_successes = []

	for _ in range(N):
		# Испытания Бернулли для каждого из отрезков
		trials = np.random.binomial(1, p, n)
		num_successes = np.sum(trials)
		successes_per_trial.append(num_successes)

		# Расчёт расстояний между успехами
		success_indices = np.where(trials == 1)[0]
		if len(success_indices) >= 2:  # успехов хотя бы 2
			# Вычисление разностей между последовательными индексами успехов.
			distances: np.ndarray = np.diff(success_indices)
			# Нормализация расстояний по количеству наблюдений (n).
			distances_normalized: np.ndarray = distances / n
			# Добавление в массив интервалов
			distances_between_successes.extend(distances_normalized)

	return successes_per_trial, distances_between_successes



if __name__ == "__main__":
	args = get_args("n", "lambda_param", "N", description="Схема Бернулли для Пуассоновского процесса")

	# Генерация данных
	successes, distances = poisson_bernoulli_simulation(args.n, args.lambda_param, args.N)

	# Гистограммы
	plot_histogram(successes, "Частота успехов в каждом испытании")
	if distances:  # Убедимся, что есть расстояния
		plot_histogram(distances, "Частота расстояний между успехами")
	else:
		print("Нет расстояний между успехами для отображения.")

