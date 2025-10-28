import random
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os


def generate_grid(size: int, probability: float) -> list:
	"""Генерирует квадратное поле с заданной вероятностью проходимости."""
	return [
		[1 if random.random() < probability else 0 for _ in range(size)]
		for _ in range(size)
		]

def show_grid(grid, visited, cell_size=30, name=""):
	"""Отображает grid как изображение, используя чёрные и белые квадраты."""
	size = len(grid)
	img_weight = size * cell_size
	img = Image.new("RGB", (img_weight, img_weight), "white")
	pixels = img.load()

	# цвета
	green = (0, 255, 0)
	gray = (128, 128, 128)
	black = (0 ,0, 0)

	for i in range(size):
		for j in range(size):
			if grid[i][j] == 0:
				color = black
			elif visited and visited[i][j]:
				color = green
			else:
				color = gray
			for x in range(cell_size):
				for y in range(cell_size):
					pixels[j * cell_size + x, i * cell_size + y] = color

	if name != "":
		print(name)
		name += ".png"
		img.save(name)

def is_path(grid, visited, x, y):
	"""Проверяет, существует ли путь с помощью DFS."""
	size = len(grid)
	
	# Проверка на границы и проходимость
	if x < 0 or x >= size or y < 0 or y >= size or not grid[x][y] or visited[x][y]:
		return False
	
	# Помечаем текущую клетку как посещенную
	visited[x][y] = True

	# Если достигли нижней границы
	if x == size - 1:
		return True

	# Проверяем соседние клетки
	if (is_path(grid, visited, x + 1, y) or       # вниз
		is_path(grid, visited, x - 1, y) or   # вверх
		is_path(grid, visited, x, y + 1) or   # вправо
		is_path(grid, visited, x, y - 1)):    # влево
		return True

	return False

def check_percolation(grid, name="") -> bool:
	"""Проверяет, существует ли путь от верхней границы к нижней."""
	size = len(grid)
	
	for j in range(size):
		if grid[0][j] == 1:  # Проверяем проходимые клетки в верхней границе
			visited = [[False] * size for _ in range(size)]
			if is_path(grid, visited, 0, j):
				if name != "":
					show_grid(grid, visited, name=name)
				return True

	if name != "":
		show_grid(grid, visited=False, name=name)
	return False

def estimate_percolation(size, prob, experiments, path=""):
	"""Оценка вероятности перколяции."""
	successes = 0
	for i in range(experiments):
		grid = generate_grid(size, prob)

		if path != "":
			name = path + f"{prob}_num{i}"
		else:
			name = ""

		if check_percolation(grid, name=name):
			successes += 1
	return successes / experiments


def main():
	parser = argparse.ArgumentParser(description='Оценка вероятности перколяции.')
	parser.add_argument('--size', type=int, default=20, help='Размер поля (NxN)')
	parser.add_argument('--experiments', type=int, default=100, help='Количество экспериментов')
	parser.add_argument('--prob_start', type=float, default=0.0, help='Начальная вероятность')
	parser.add_argument('--prob_end', type=float, default=1.0, help='Конечная вероятность')
	parser.add_argument('--prob_step', type=float, default=0.1, help='Шаг вероятности')
	parser.add_argument('--path', type=str, default="", help='Куда сохранять изображения')
	
	args = parser.parse_args()

	# вероятности
	steps = int((args.prob_end - args.prob_start) / args.prob_step) + 1
	float_probabilities = [
		args.prob_start + step * args.prob_step
		for step in range(steps)
		]
	probabilities = [round(i, 2) for i in float_probabilities]
	percolation_probabilities = []

	if args.path != "":
		os.makedirs(args.path, exist_ok=True)
	for prob in probabilities:
		p = estimate_percolation(args.size, prob, args.experiments, path=args.path)
		percolation_probabilities.append(p)

	# Построение графика
	plt.plot(probabilities, percolation_probabilities, marker='o')
	plt.xlabel('Вероятность проходимости')
	plt.ylabel('Вероятность перколяции')
	plt.title('Зависимость вероятности перколяции от вероятности проходимости')
	plt.grid()
	plt.savefig('percolation_probability_plot.png')
	plt.show()

if __name__ == '__main__':
	main()

