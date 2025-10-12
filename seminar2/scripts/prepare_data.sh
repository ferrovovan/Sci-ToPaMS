#!/bin/bash

# Скрипт для подготовки данных

echo "Распаковка базы данных..."
if [ -f "enron-spam_database.tar.xz" ]; then
    if [ ! -d "database" ]; then
        mkdir -p database
        tar -xf enron-spam_database.tar.xz -C database
        echo "База данных распакована в папку database/"
    else
        echo "Папка database уже существует, пропускаем распаковку"
    fi
else
    echo "Файл enron-spam_database.tar.xz не найден!"
    exit 1
fi

: << EOF
# echo "Распределение писем для валидации..."
# python3 utils/distribute_emails.py

# echo "Данные подготовлены."
# echo "Теперь можно запустить тренировку: ./train_model.sh"
EOF

echo "Распределите 20% писем для валидации!"
