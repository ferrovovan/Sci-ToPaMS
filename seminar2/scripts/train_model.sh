#!/bin/bash

# Скрипт для тренировки модели

echo "Запуск тренировки модели..."
python3 source/main.py --train --db_name="naive_bayes.db.csv" --dataset_path="database/"

echo "Тренировка завершена. База данных сохранена в naive_bayes.db.csv"

