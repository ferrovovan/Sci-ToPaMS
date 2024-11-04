import argparse

def get_args(*args, description: str = ""):
	parser = argparse.ArgumentParser(description=description)
	for arg in args:
		if arg == "lambda_param":  # λ
			parser.add_argument("-lambda_param", type=float, default=5.0, help="параметр λ (по умолчанию 5.0)")
		elif arg == "n":
			parser.add_argument("-n", type=int, default=100, help="параметр n (по умолчанию 100)")
		elif arg == "N":
			parser.add_argument("-N", type=int, default=1000, help="количество итераций (по умолчанию 1000)")

	parsed_args = parser.parse_args()
	# Валидация значений параметров
	for arg_name, value in vars(parsed_args).items():
		if value <= 0:
			parser.error(f"Параметр {arg_name} должен быть > 0")
	return parsed_args

