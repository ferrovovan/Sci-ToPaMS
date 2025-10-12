# Python задание 2: наивный байесовский классификатор

**Зависимости:**
* numpy

## Математическая основа
Этот раздел расписан (пока - нет) в `docs/math_explanation.md`


## Как использовать
1. Распаковать **набор писем** особого вида, как `enron-spam_database.tar.xz`. Например
```bash
tar xf enron-spam_database.tar.xz
```
ИЛИ
```bash
sh scripts/prepare_data.sh
```

2. Переименовать папку с данными (стандарт - `database`).
```bash
mv  enron-spam_database  database
```

3. Выделить из него (набора писем) **валидационную группу**.  
Я не знаю как это делать правильно. Можно попробовать так:
```bash
python3 distribute_emails_dev/simple_and_weird_distribute_emails.py
```


4. Запуск тренировки.
```bash
python3 source/main.py --train --dataset_path="database/"
```
ИЛИ
```bash
scripts/train_model.sh
```

5. Узнать итоги обучения
```bash
python3 source/main.py --validate --dataset_path="database/"
```
ИЛИ
```
sh scripts/validate_model.sh
```


