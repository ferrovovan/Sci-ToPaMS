import random
import matplotlib.pyplot as plt
import argparse

def estimate_pi(num_points):
    inside_circle = 0
    pi_estimates = []

    for i in range(1, num_points + 1):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)

        distance = x**2 + y**2
        if distance <= 1:
            inside_circle += 1

        pi_estimate = (inside_circle / i) * 4
        pi_estimates.append(pi_estimate)

    return pi_estimates

def main():
    # Используем argparse для получения количества точек из командной строки
    parser = argparse.ArgumentParser(description='Оценка числа π методом Монте-Карло.')
    parser.add_argument('--points', type=int, default=10000, help='Количество точек для оценки π')
    args = parser.parse_args()

    # Количество точек берется из аргумента командной строки
    num_points = args.points
    pi_estimates = estimate_pi(num_points)

    # Построение графика
    plt.plot(range(1, num_points + 1), pi_estimates)
    plt.axhline(y=3.141592653589793, color='r', linestyle='--', label='π (точное)')
    plt.xlabel('Количество точек')
    plt.ylabel('Приближенное значение π')
    plt.title(f'Оценка числа π методом Монте-Карло ({num_points} точек)')
    plt.legend()
    plt.savefig('pi_estimate_plot.png')  # Сохранение графика в файл
    plt.show()

if __name__ == '__main__':
    main()

