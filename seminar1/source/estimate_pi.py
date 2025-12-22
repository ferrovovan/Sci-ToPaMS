import time
import matplotlib.pyplot as plt
import argparse


# Линейный конгруэнтный генератор
class LCG:
	def __init__(self, seed=None, a=6364136223846793005, b=1442695040888963407, M=2**64):
		if seed is None:
			seed = int(time.time_ns())
		self.state = seed
		self.a = a
		self.b = b
		self.M = M

	def rand(self):
		"""Возвращает число в диапазоне [0,1)."""
		self.state = (self.a * self.state + self.b) % self.M
		return self.state / self.M


def estimate_pi(num_points: int, rng: LCG) -> tuple[float]:
	inside_circle = 0
	pi_estimates = []

	for i in range(1, num_points + 1):
		x = rng.rand()
		y = rng.rand()

		distance = x**2 + y**2
		if distance <= 1:
			inside_circle += 1

		pi_estimate = (inside_circle / i) * 4
		pi_estimates.append(pi_estimate)

	return tuple(pi_estimates)


def plot_estimate_pi(pi_estimates: tuple[float]):
	num_points = len(pi_estimates)
	plt.plot(range(1, num_points + 1), pi_estimates)
	plt.axhline(y=3.141592653589793, color='r', linestyle='--', label='π (точное)')
	plt.xlabel('Количество точек')
	plt.ylabel('Приближенное значение π')
	plt.title(f'Оценка числа π методом Монте-Карло ({num_points} точек, свой ГСЧ)')
	plt.legend()
	plt.savefig('pi_estimate_plot.png')  # Сохранение графика в файл
	plt.show()


def plot_CLT(pi_estimates: tuple[float]):
	import numpy as np
	"""
	Визуализация ЦПТ: показывает, как уменьшается ошибка оценки π
	с увеличением количества точек.
	
	Демонстрирует, что увеличение выборки в 10 раз уменьшает ошибку  в √10 ≈ 3.16 раз
	"""
	num_points = len(pi_estimates)
	
	# Вычисляем ошибки (разницу между приближенным и точным значением π)
	true_pi = 3.141592653589793
	errors = [abs(estimate - true_pi) for estimate in pi_estimates]
	
	# Преобразуем логарифмически для лучшей визуализации
	log_errors = np.log10(errors)
	log_points = np.log10(range(1, num_points + 1))
	
	# Создаем график
	plt.figure(figsize=(12, 8))
	
	# 1. График ошибок в логарифмическом масштабе
	plt.subplot(2, 1, 1)
	plt.plot(range(1, num_points + 1), errors, 'b-', alpha=0.7, linewidth=0.5)
	plt.xlabel('Количество точек (N)')
	plt.ylabel('Абсолютная ошибка |π - π_est|')
	plt.title('Сходимость оценки π по методу Монте-Карло\n(относительная разность)')
	plt.grid(True, alpha=0.3)
	
	# Добавляем горизонтальные линии для разных порядков точности
	accuracy_levels = [1e-1, 1e-2]
	for level in accuracy_levels:
		plt.axhline(y=level, color='gray', linestyle='--', alpha=0.5)
		plt.text(num_points * 0.98, level, f' {level}', 
				verticalalignment='bottom', fontsize=8)
	
	# 2. Логарифмический график для демонстрации ЦПТ
	plt.subplot(2, 1, 2)
	plt.scatter(log_points, log_errors, s=1, alpha=0.5, c='red', label='Ошибка оценки')
	
	# Линейная регрессия для логарифмированных данных
	# Согласно ЦПТ, ошибка должна уменьшаться как ~1/√N, т.е. log(error) ~ -0.5*log(N)
	coefficients = np.polyfit(log_points, log_errors, 1)
	regression_line = np.polyval(coefficients, log_points)
	plt.plot(log_points, regression_line, 'k-', linewidth=2, 
			label=f'Аппроксимация: наклон = {coefficients[0]:.3f}')
	
	plt.xlabel('log₁₀(N) (количество точек)')
	plt.ylabel('log₁₀(Ошибки)')
	plt.title('Центральная предельная теорема: ошибка ~ 1/√N')
	plt.grid(True, alpha=0.3)
	plt.legend()
	
	# 3. Демонстрация "увеличение в 10 раз → точность на 1 знак больше"
	plt.figtext(0.02, 0.02, 
		"ЦПТ предсказывает: Увеличение выборки в 10 раз уменьшает ошибку в ~√10 ≈ 3.16 раз\n"
		"Это соответствует увеличению точности примерно на 0.5 десятичного знака\n"
		"Для увеличения точности на 1 знак (в 10 раз) нужно увеличить выборку в 100 раз",
		fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
	
	# 4. Подтверждение теоретической зависимости
	print("\n" + "="*60)
	print("АНАЛИЗ ЦПТ ДЛЯ МЕТОДА МОНТЕ-КАРЛО:")
	print("="*60)
	
	# Выбираем контрольные точки для демонстрации
	check_points = [10, 100, 1000, 10000, min(100000, num_points)]
	if num_points >= 1000000:
		check_points.append(1000000)
	
	print("\nДемонстрация сходимости:")
	print("-"*60)
	print(f"{'N':>10} {'π_est':>15} {'Ошибка':>15} {'log10(ошибки)':>15}")
	print("-"*60)
	
	for N in check_points:
		if N <= num_points:
			error = errors[N-1]
			print(f"{N:>10} {pi_estimates[N-1]:>15.10f} {error:>15.10f} {np.log10(error):>15.3f}")
	
	# Теоретическое обоснование
	print("\n" + "-"*60)
	print("ТЕОРЕТИЧЕСКОЕ ОБОСНОВАНИЕ:")
	print("-"*60)
	print("1. Для метода Монте-Карло ошибка убывает как 1/√N (согласно ЦПТ)")
	print("2. Увеличение выборки в 10 раз: ошибка уменьшается в √10 ≈ 3.16 раз")
	print("3. log₁₀(3.16) ≈ 0.5 → точность увеличивается на ~0.5 десятичного знака")
	
	# Экспериментальная проверка
	if num_points >= 1000:
		print("\n" + "-"*60)
		print("ЭКСПЕРИМЕНТАЛЬНАЯ ПРОВЕРКА:")
		print("-"*60)
		
		# Сравниваем ошибки при разных размерах выборки
		ratios = []
		for i in range(1, 5):
			N1 = 10**i
			N2 = 10**(i+1)
			if N2 <= num_points:
				ratio = errors[N1-1] / errors[N2-1]
				ratios.append(ratio)
				print(f"N={N1} → N={N2}: ошибка уменьшилась в {ratio:.2f} раз")
		
		if ratios:
			avg_ratio = np.mean(ratios)
			print(f"\nСреднее уменьшение ошибки при увеличении N в 10 раз: {avg_ratio:.2f}")
			print(f"Теоретическое значение (√10): {np.sqrt(10):.2f}")
			print(f"Расхождение: {abs(avg_ratio - np.sqrt(10))/np.sqrt(10)*100:.1f}%")
	
	plt.tight_layout()
	plt.savefig('clt_analysis.png', dpi=150)
	plt.show()
	

def main():
	# Используем argparse для получения количества точек из командной строки
	parser = argparse.ArgumentParser(description='Оценка числа π методом Монте-Карло.')
	parser.add_argument('--points', type=int, default=10000, help='Количество точек для оценки π')
	args = parser.parse_args()

	# Создаём генератор
	rng = LCG()
	# Количество точек берется из аргумента командной строки
	num_points = args.points
	pi_estimates: tuple[float] = estimate_pi(num_points, rng)

	# Построение графика
	plot_estimate_pi(pi_estimates)
	# plot_CLT(pi_estimates)


if __name__ == '__main__':
	main()
