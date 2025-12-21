#!/bin/bash

# Скрипт для проверки модели на нескольких письмах

# Функция для обработки одного письма
process_email() {
    local email_path="$1"
    echo "Обработка письма: $email_path"
    python3 source/main.py --email "$email_path" --db_name="naive_bayes.db.csv" --dataset_path="database/"
    echo "Готово: $email_path"
    echo "----------------------------------------"
}

# Массив с путями к проверяемым письмам
declare -a email_paths=(
    # ham
    #"database/valide/ham/0005.1999-12-12.kaminski.ham.txt"  # TN
    #"database/valide/ham/0003.1999-12-14.farmer.ham.txt"  # TN
    # spam
    #"database/valide/spam/0004.2004-08-01.BG.spam.txt"  # FN
    #"database/valide/spam/0011.2001-06-29.SA_and_HP.spam.txt"  # FN
    #"database/valide/spam/0022.2004-08-03.BG.spam.txt"  # TP
    "database/valide/spam/0108.2004-08-09.BG.spam.txt"  # TP
    #"database/valide/spam/0120.2002-05-10.SA_and_HP.spam.txt"  # TP
)

echo "Запуск предоставления модели на ${#email_paths[@]} письме(ах)..."

success_count=0
error_count=0


# Цикл по массиву писем
for email_path in "${email_paths[@]}"; do
    if [[ -f "$email_path" ]]; then
        if process_email "$email_path"; then
            ((success_count++))
        else
            ((error_count++))
        fi
    else
        echo "Предупреждение: файл не найден - $email_path"
        echo "----------------------------------------"
        ((error_count++))
    fi
done

echo "========================================"
echo "Итоги выборочной валидации:"
echo "Успешно обработано: $success_count"
echo "С ошибками: $error_count"
echo "Всего обработано: ${#email_paths[@]}"
echo "Валидация завершена."
