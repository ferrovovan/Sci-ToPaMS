# Семинар 2: наивный байесовский классификатор

## Как использовать
1. Распаковать набор писем (например, `enron-spam_database.tar.xz`)
2. Выделить из него валидационную группу с помощью `utils/distribute_emails.py`
3. Переименовать папку с данными (стандарт - `dataset`).
4. Запустить тренировку
```bash
python3 source/main.py --train --dataset_path ./dataset
```
5. Узнать результат
```bash
python3 source/main.py --validate --dataset_path ./dataset
```
  
- Также смотри `utils/main.sh`
