#!/bin/bash

# Функция для вывода текста зелёным цветом
echo_green() {
	local text="$1"
	echo -e "\n\033[0;32m$text\033[0m"  # Вывод текста с зелёным цветом
}

# Переходим в директорию source и запускаем python-задания
cd "$(dirname "$0")/source" || exit 1



## Примеры использования

echo_green "Чтобы закрыть окно matplotlib нажмите 'q' или 'Ctrl + w'."
RUN_TASK_1=true
RUN_TASK_2=true
RUN_TASK_3=true


# Задание 1
if $RUN_TASK_1; then
	echo_green "Задание 1."
	python3 task1.py

fi

# Задание 2
if $RUN_TASK_2; then
	echo_green "Задание 2."
	python3 task2.py
fi

# Задание 3
if $RUN_TASK_3; then
	echo_green "Задание 3."
	python3 task3.py
fi
