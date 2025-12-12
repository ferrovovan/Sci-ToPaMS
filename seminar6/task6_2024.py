import pandas as pd
import numpy as np
import scipy.stats as st


SEPARATE_LINE_LEN = 60

# ============================
# Задание №1: Подготовка данных
# ============================

# Загрузка данных
df = pd.read_csv("House_Price_2024.csv")

# Выделение целевой переменной
target = pd.DataFrame(df["price"])  # y - что оцениваем
df.drop("price", axis=1, inplace=True)  # выбрасываем из рабочей таблицы
# "убрать бесполезный столбец, чтобы не портил матрицу X".
df.drop("bus_ter", axis=1, inplace=True)  # Везде "YES"

# 1.2: Кодирование категориальной переменной (one-hot encoding)
waterbody_encoded = pd.get_dummies(df["waterbody"], "waterboody")
df.drop("waterbody", axis=1, inplace=True)
df = pd.concat([df, waterbody_encoded], axis=1)

# Кодирование бинарной переменной "airport"
df["airport"] = df["airport"].map({"YES": 1, "NO": 0})

# Приведение к числовому типу
df = df.astype(np.float64)

# Обработка пропусков
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
	X = inpX
	y = inpY
	
	# Нормальное уравнение производной по β.
	# β̂ = (XᵀX)⁻¹ Xᵀ y
	beta_ = np.linalg.inv(X.T @ X) @ X.T @ y
	beta_ = np.expand_dims(beta_, 0).T
	eps_ = (y - X @ beta_)[0]

	# X.shape[0] = n,
	# X.shape[1] = k + 1  (первый столбец добавили в prepare_XY()
	# σ̂² = (∑ ε̂ᵢ²) / (n − k − 1)
	sigma_ = np.sum(eps_ ** 2) / (X.shape[0] - X.shape[1])  # оценка σ²

	# return beta_[0], eps_, sigma_
	return beta_[0], sigma_


def pred(inpX, beta, eps):
	"""Предсказание значений."""
	return inpX @ beta + eps


# ============================
# Задание №3: Тест на значимость и backward elimination
# ============================

deleted_columns = []
myX, myY = prepare_XY(df, target)

while True:
	print("\n", "=" * SEPARATE_LINE_LEN)
	
	# Оценка модели
	# b, e, s = fit(myX, myY)
	b, s = fit(myX, myY)
	print("Оценка дисперсии ошибок:", s)

	# Матрица ковариации оценок коэффициентов
	var_beta = np.diag(s * np.linalg.inv(myX.T @ myX))
	print("Дисперсии коэффициентов:", var_beta)

	# Стандартизация коэффициентов (опционально, для интерпретации)
	std_X = np.std(myX[:,1:], axis=0, ddof=1)
	std_Y = np.std(myY, ddof=1)
	beta_hat = b[1:].T[0]
	standardized_betas = beta_hat * (std_X / std_Y)
	print("Стандартизованные коэффициенты:", standardized_betas)

	# Расчет t-статистик и p-значений
	std_beta_hat = np.sqrt(var_beta[1:])
	# t = β̂ₛ / sqrt(Var(β̂ₛ))
	# чем больше по модулю t, тем дальше β̂ₛ от нуля, тем «значимее» признак
	t_statistics = beta_hat / std_beta_hat

	# степени свободы = n − p.
	df_ = myX.shape[0] - myX.shape[1]  # степени свободы
	p_values = 2 * st.t.cdf(-np.abs(t_statistics), df_)

	# Можно красиво печатать таблицей,
	# поставив в заголовке df.columns
	print("t-статистика:", t_statistics)
	print("p-значения:", p_values)

	# Проверка условия остановки (все p-values < 0.05)
	if not np.all(p_values < 0.05):
		idx = np.argmax(p_values)
		print(f"Удаляем {df.columns[idx]}")
		deleted_columns.append(df.columns[idx])
		df = df.drop(df.columns[idx], axis=1)
		myX, myY = prepare_XY(df, target)
	else:
		break


# ============================
# Сохранение результата
# ============================
for i in range(3):
	print("-" * SEPARATE_LINE_LEN)
print("Deleted:", deleted_columns)
df = pd.concat([target, df], axis=1)  # Обратно приклеиваем цены
print(df)
df.to_csv('New_House_Price_2024.csv', index=False)
