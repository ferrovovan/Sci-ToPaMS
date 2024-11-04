import numpy as np
from plot_histogram_func import plot_histogram
from parse_args_func import get_args


def poisson_exponential_simulation(lambda_param: float, N: int) -> list:
	successes_per_trial = []

	for _ in range(N):
		x = 0  # Начальная точка
		successes = 0

		# Генерируем интервалы между успехами, пока x <= 1
		while x <= 1:
			# Генерация расстояния между успехами по показательному распределению
			x += np.random.exponential(1 / lambda_param)  # генерирует случайное значение на основе показательного распределения с параметром `1 / lambda_param`
			if x <= 1:
				successes += 1
		
		# Сохраняем число успехов для текущего испытания
		successes_per_trial.append(successes)

	return successes_per_trial


if __name__ == "__main__":
	args = get_args("lambda_param", "N", description="Пуассоновский процесс из Показательных распределений")

	# Генерация данных с использованием показательного распределения
	successes_exp = poisson_exponential_simulation(args.lambda_param, args.N)

	# Построение гистограммы для показательного распределения
	plot_histogram(successes_exp, "Гистограмма количества успехов (показательное распределение)")

