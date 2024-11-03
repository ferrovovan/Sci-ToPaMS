import numpy as np
import matplotlib.pyplot as plt
from math import log2, ceil


# правило Стёржеса
def sturges_rule(data: list) -> int:
	return 1 + int(log2(len(data)))

# правило Фридмана-Диакониса
def freedman_diaconis_rule(data: list) -> int:
	q75, q25 = np.percentile(data, [75, 25])
	iqr = q75 - q25
	bin_width = 2 * iqr / (len(data) ** (1/3))
	return ceil((max(data) - min(data)) / bin_width)



# Построение гистограммы по 
def plot_histogram(data: list, title: str) -> None:
	n = len(data)
	m = sturges_rule(data)
	data_min, data_max = min(data), max(data)
	bin_width = (data_max - data_min) / m

	# Гистограмма вручную по формуле для высоты столбца
	bin_edges = np.linspace(data_min, data_max, m + 1)
	bin_counts, _ = np.histogram(data, bins=bin_edges)
	heights = (bin_counts / n) / bin_width

	# Построение графика
	plt.bar(bin_edges[:-1], heights, width=bin_width, align='edge', alpha=0.7)
	plt.title(title)
	plt.xlabel("Значения")
	plt.ylabel("Частота")
	plt.show()

