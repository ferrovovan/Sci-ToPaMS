import numpy as np
from plot_histogram_func import plot_histogram
from parse_args_func import get_args


def poisson_process_distances(lambda_param: float, N: int) -> list:
	all_distances = []

	for _ in range(N): # проводим N итераций алгоритма
		# Генерируем число успехов на интервале [0, 1] согласно Пуассоновскому распределению
		num_successes = np.random.poisson(lambda_param)
		
		# Если успехи есть, генерируем их позиции на интервале [0, 1]
		if num_successes > 1:
			points = np.sort(np.random.uniform(0, 1, num_successes))
			distances = np.diff(points)  # Расстояния между соседними точками
			all_distances.extend(distances)

	return all_distances


if __name__ == "__main__":
	args = get_args("lambda_param", "N", description="Показательное распределение из Пуассоновского процесса")
	
	# Генерация расстояний
	distances = poisson_process_distances(args.lambda_param, args.N)

	# Построение гистограммы
	if distances:
		plot_histogram(distances, "Гистограмма расстояний между успехами")
	else:
		print("Нет расстояний между успехами для отображения.")

