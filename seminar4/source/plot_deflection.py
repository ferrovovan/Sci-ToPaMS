import numpy as np
import matplotlib.pyplot as plt

from parse_args_func import get_args
from task1 import poisson_bernoulli_simulation
from task2 import poisson_process_distances
from task3 import poisson_exponential_simulation

from math import factorial  # fix: numpy 2.0 


def calculate_theoretical_exponential(lambda_param, num_bins, data_range):
    """Вычисляет теоретическую кривую для экспоненциального распределения."""
    x_values = np.linspace(data_range[0], data_range[1], num_bins)
    y_values = lambda_param * np.exp(-lambda_param * x_values)
    return x_values, y_values


def calculate_theoretical_poisson(lambda_param, max_k):
    """Вычисляет теоретическую вероятность для распределения Пуассона."""
    k_values = np.arange(0, max_k + 1)
    # factorial_vectorized = np.vectorize(np.math.factorial)  # old: numpy <2.0
    factorial_vectorized = np.vectorize(factorial)  # new: numpy >2.0
    y_values = (np.exp(-lambda_param) * lambda_param ** k_values) / factorial_vectorized(k_values)
    return k_values, y_values


def plot_exponential_comparison(process_distances, exponential_distances, lambda_param):
    """Построение гистограмм для экспоненциального распределения."""
    plt.figure(figsize=(20, 12))
    plt.title("Сравнение: Показательное распределение")

    # Построение двух гистограмм на одном графике
    plt.hist(process_distances, bins="sturges", alpha=0.6, color="blue", label="Схема Бернулли", density=True)
    plt.hist(exponential_distances, bins="sturges", alpha=0.6, color="green", label="Показательное (экспоненциальная симуляция)", density=True)

    # Теоретическое показательное распределение
    x_exp, y_exp = calculate_theoretical_exponential(lambda_param, num_bins=50, data_range=(0, np.max(exponential_distances)))
    plt.plot(x_exp, y_exp, label="Теоретическое показательное", color="black", linestyle="--")

    plt.legend()
    plt.xlabel("Значения")
    plt.ylabel("Плотность вероятности")
    plt.show()


def plot_poisson_comparison(process_success_counts, exponential_success_counts, lambda_param):
    """Построение гистограмм для распределения Пуассона."""
    plt.figure(figsize=(20, 12))
    plt.title("Сравнение: Распределение Пуассона")

    # Построение двух гистограмм на одном графике
    plt.hist(process_success_counts, bins="sturges", alpha=0.6, color="red", label="Схема Бернулли", density=True)
    plt.hist(exponential_success_counts, bins="sturges", alpha=0.6, color="orange", label="Показательное распределение", density=True)

    # Теоретическое распределение Пуассона
    k_pois, y_pois = calculate_theoretical_poisson(lambda_param, max_k=int(np.max(process_success_counts)))
    plt.plot(k_pois, y_pois, label="Теоретическое Пуассоновское", color="black", linestyle="--")

    plt.legend()
    plt.xlabel("Количество событий")
    plt.ylabel("Вероятность")
    plt.show()


if __name__ == "__main__":
    # Обработка параметров
    args = get_args("n", "lambda_param", "N", description="Сравнение гистограмм из разных алгоритмов")

    # Параметры
    n = args.n
    lambda_param = args.lambda_param
    N = args.N

    # Генерация данных по заданиям 1, 2 и 3
    trials_success_counts_1, distances_1 = poisson_bernoulli_simulation(n, lambda_param, N)
    distances_2 = poisson_process_distances(lambda_param, N)
    trials_success_counts_3 = poisson_exponential_simulation(lambda_param, N)

    # Построение гистограмм для экспоненциального распределения
    plot_exponential_comparison(distances_1, distances_2, lambda_param)

    # Построение гистограмм для распределения Пуассона
    plot_poisson_comparison(trials_success_counts_1, trials_success_counts_3, lambda_param)

