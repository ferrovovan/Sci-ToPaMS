import pandas as pd
import numpy as np


# для сообщений
SEPARATE_LINE_LEN = 60

# ============================
# Задание №1: Подготовка данных
# ============================

# Загрузка данных
df = pd.read_csv("House_Prices_2025.csv")
# В сравнении с 2024, убрано поле crime_rate, добавлено поле Sold.
#  >>> df.keys()
#  Index(['price', 'resid_area', 'air_qual', 'room_num', 'age',
#  'dist1', 'dist2', 'dist3', 'dist4',
#  'teachers', 'poor_prop', 'airport',
#  'n_hos_beds', 'n_hot_rooms', 'waterbody',
#  'rainfall', 'bus_ter', 'parks', 'Sold'],
#        dtype='object')

# 1.0 Выделение целевой переменной
target = pd.DataFrame(df["price"])  # y - что оцениваем
df.drop("price", axis=1, inplace=True)  # выбрасываем из рабочей таблицы

# "убрать бесполезный столбец, чтобы не портил матрицу X".
df.drop("bus_ter", axis=1, inplace=True)  # Везде "YES"

# 1.2: Кодирование категориальной переменной (one-hot encoding)
df["waterbody"] = df["waterbody"].fillna("None")
waterbody_encoded = pd.get_dummies(
	df["waterbody"],
	prefix="waterbody",
	drop_first=True  # чтобы избежать ловушки фиктивных переменных
)
df = pd.concat([df.drop("waterbody", axis=1), waterbody_encoded], axis=1)

# Кодирование бинарной переменной "airport"
df["airport"] = df["airport"].map({"YES": 1, "NO": 0})

# Приведение к числовому типу
df = df.astype(np.float64)

# Обработка пропусков для "n_hos_beds"
df["n_hos_beds"] = df["n_hos_beds"].fillna(df["n_hos_beds"].mean())  # заполняем средним значением

# 1.1: Удаление сильно коррелированных переменных
correlation_matrix = df.corr()
high_correlation = correlation_matrix[np.abs(correlation_matrix) > 0.9]
upper_triangle = high_correlation.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]

if len(to_drop) > 0:
	df = df.drop(columns=to_drop)


# ============================
# Вспомогательные функции для линейной регрессии
# ============================

def prepare_XY(X, Y):
	"""Подготовка матрицы признаков с добавлением столбца единиц для β0."""
	X = np.asarray(X)
	# дополняем первым столбцом из единиц — ради β₀.
	ones_column = np.ones((X.shape[0], 1))
	X = np.hstack((ones_column, X))  

	Y = np.asarray(Y)
	return X, Y


def fit(inpX, inpY):
	"""Задание №2: Оценка коэффициентов, ошибок и дисперсии ошибок."""
	X = np.asarray(inpX)
	y = np.asarray(inpY).reshape(-1, 1)
	
	# Нормальное уравнение производной по β.
	# β̂ = (XᵀX)⁻¹ Xᵀ y
	beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
	# beta_hat = np.expand_dims(beta_, 0).T  # приведение к 'столбцу'
	# ŷi = β̂0 + β̂1 xi1 + β̂2 xi2 + ⋅⋅⋅ + β̂k xik  ; предсказания
	y_hat = X @ beta_hat
	# ε̂i = yi − ŷi  ; вектор ошибок
	# eps_hat = (y - y_hat)[0]
	eps_hat = (y - y_hat)

	# X.shape[0] = n,
	# X.shape[1] = k + 1  (первый столбец добавили в prepare_XY()
	# σ̂² = (∑ ε̂ᵢ²) / (n − k − 1)
	n, p = X.shape
	# sigma_hat = np.sum(eps_ ** 2) / (n - p)  # оценка σ²
	sigma_hat = (eps_hat.T @ eps_hat)[0, 0] / (n - p)  # оценка σ²

	# return beta_[0], eps_, sigma_
	return beta_hat, eps_hat, sigma_hat


def pred(inpX, beta, eps):
	"""Предсказание значений."""
	return inpX @ beta + eps


# ============================
# Задание №3: Тест на значимость и backward elimination
# ============================
import scipy.stats as st


def test_beta_significance(X, beta_hat, sigma_hat, s):
	"""
	Проверка гипотез:
		H0: β_s = 0
		H1: β_s ≠ 0

	Возвращает:
		t_stat, p_value
	"""
	cov_beta = sigma_hat * np.linalg.inv(X.T @ X)
	var_beta_s = cov_beta[s, s]

	t_stat = beta_hat[s, 0] / np.sqrt(var_beta_s)
	p_value = 2 * st.norm.cdf(-abs(t_stat))

	return t_stat, p_value


def all_p_values(X, beta_hat, sigma_hat):
	"""
	Возвращает массив p-value для коэффициентов β1,...,βk (кроме β0)
	"""
	p_values = []
	for s in range(1, X.shape[1]):  # пропускаем β0
		_, p = test_beta_significance(X, beta_hat, sigma_hat, s)
		p_values.append(p)
	return np.array(p_values)


def backward_elimination(df, target, alpha=0.05):
	"""
	Пошаговый отбор значимых переменных
	"""
	deleted_columns = []
	X, y = prepare_XY(df, target)

	while True:
		print("\n" + "=" * SEPARATE_LINE_LEN)

		beta_hat, eps_hat, sigma_hat = fit(X, y)
		print("Оценка дисперсии ошибок σ̂²:", sigma_hat)

		p_values = all_p_values(X, beta_hat, sigma_hat)

		print("p-values:")
		for name, p in zip(df.columns, p_values):
			print(f"{name:20s}: {p:.6f}")

		if np.all(p_values < alpha):
			print("Все переменные значимы. Остановка.")
			break

		idx = np.argmax(p_values)
		col_to_drop = df.columns[idx]

		print(f"Удаляем переменную: {col_to_drop}")
		deleted_columns.append(col_to_drop)

		df = df.drop(columns=[col_to_drop])
		X, y = prepare_XY(df, target)

	return df, deleted_columns


df, deleted_columns = backward_elimination(df, target)
print("Удалённые признаки:", deleted_columns)


# ============================
# Задание №4
# ============================
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist


def beta_params_from_mean_std(mean, std):
	"""
	Преобразует матожидание и стандартное отклонение в параметры alpha и beta
	бета-распределения.
	
	Формулы:
	alpha = mean * nu
	beta = (1 - mean) * nu
	где nu = (mean*(1-mean)/std^2) - 1
	"""
	
	if not (0 < mean < 1):
		raise ValueError("mean должно быть в (0,1)")

	max_std = np.sqrt(mean * (1 - mean))
	if std <= 0 or std >= max_std:
		raise ValueError(
			f"std должно быть в (0, {max_std:.4f}) для mean={mean}"
		)

	var = std ** 2
	k = mean * (1 - mean) / var - 1
	alpha = mean * k
	beta = (1 - mean) * k
	return alpha, beta


def posterior_update(alpha, beta, successes, failures):
	"""
	Апостериор для Бернулли + Beta
	"""
	return alpha + successes, beta + failures


# Загружаем "waterbody" заново (без one-hot)
temp_df = pd.read_csv("House_Prices_2025.csv")
temp_df["waterbody"] = temp_df["waterbody"].fillna("None")
df["waterbody"] = temp_df["waterbody"].values


# ============================
# ВВОД ПОЛЬЗОВАТЕЛЯ
# ============================

prior_mean = float(input("Введите априорное среднее p (0..1): "))
prior_std = float(input("Введите априорное стандартное отклонение: "))

alpha0, beta0 = beta_params_from_mean_std(prior_mean, prior_std)

print(f"Априор: alpha={alpha0:.3f}, beta={beta0:.3f}")


groups = df.groupby("waterbody")

posterior_params = {}

# Апостериоры по каждой группе
for name, g in groups:
	S = g["Sold"].sum()
	F = len(g) - S

	alpha_post, beta_post = posterior_update(alpha0, beta0, S, F)

	mean_post = alpha_post / (alpha_post + beta_post)
	std_post = np.sqrt(
		alpha_post * beta_post /
		((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1))
	)

	posterior_params[name] = (alpha_post, beta_post)

	print(f"\nГруппа: {name}")
	print(f"  Апостериорное среднее: {mean_post:.4f}")
	print(f"  Апостериорное std:     {std_post:.4f}")

# Последовательные байесовские обновления
print("\nПоследовательное объединение групп:")

alpha_seq, beta_seq = alpha0, beta0

for name, g in groups:
	S = g["Sold"].sum()
	F = len(g) - S

	alpha_seq, beta_seq = posterior_update(alpha_seq, beta_seq, S, F)

	mean_seq = alpha_seq / (alpha_seq + beta_seq)
	std_seq = np.sqrt(
		alpha_seq * beta_seq /
		((alpha_seq + beta_seq)**2 * (alpha_seq + beta_seq + 1))
	)

	print(f"{name:15s}: mean={mean_seq:.4f}, std={std_seq:.4f}")

# Графики априора и апостериоров
x = np.linspace(0, 1, 500)

plt.figure(figsize=(10, 6))
plt.plot(x, beta_dist.pdf(x, alpha0, beta0), label="Prior", linewidth=2)

for name, (a, b) in posterior_params.items():
	plt.plot(x, beta_dist.pdf(x, a, b), label=f"Posterior: {name}")

plt.xlabel("p")
plt.ylabel("Плотность")
plt.legend()
plt.grid(True)
plt.show()
