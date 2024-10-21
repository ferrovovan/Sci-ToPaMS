#!/bin/bash

# Запуск режима тренировки и валидации
#python3 main.py --train
#python3 main.py --validate

dataset_path=""
# Список писем для проверки
emails=(
#	"$dataset_path/valide/ham/0019.2000-06-09.lokay.ham.txt"
#	"$dataset_path/valide/spam/0025.2001-08-01.SA_and_HP.spam.txt"
	"./test.txt"
)

# Перебор каждого письма в списке и запуск main.py с соответствующим аргументом
for email in "${emails[@]}"; do
	echo -e "\n Проверка письма: $email"
	python3 main.py --email "$email"
done

