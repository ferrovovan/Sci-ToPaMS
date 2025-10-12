#!/bin/bash

# Скрипт для валидации модели


echo "Запуск валидации модели..."
python3 source/main.py --validate --db_name="naive_bayes.db.csv" --dataset_path="database/"

echo "Валидация завершена."
