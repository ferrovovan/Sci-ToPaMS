import argparse
from email_loader import FileEmailLoader
from text_processor import TextProcessor
from naive_bayes_db import NaiveBayesDB
from data_manager import DataManager
from os.path import exists

def train(db, data_manager, email_loader, text_processor, db_name, dataset_path: str):
    for i in range(1, 7):
        emails = email_loader.load_emails(f'{dataset_path}/enron{i}')
        for email, is_spam in emails:
            words = text_processor.preprocess(email)
            db.train(words, is_spam)
    data_manager.save_to_csv(db, db_name)
    print(f"Тренировка завершена. База данных сохранена в {db_name}.")

def validate(db, data_manager, email_loader, text_processor, db_name, dataset_path: str):
    total = 0
    right = 0
    emails = email_loader.load_emails(f'{dataset_path}/valide')
    total += len(emails)
    for email, is_spam in emails:
        words = text_processor.preprocess(email)
        if db.classify(words) == is_spam:
            right += 1
    accuracy = right / total
    print(f"Результат проверки: {accuracy * 100:.2f}% точности.")

def custom_test(data_manager: DataManager, db: NaiveBayesDB, email_path: str):
    """Печатает конкретное письмо и выводит результат проверки."""
    try:
        with open(email_path, 'r', encoding='utf-8') as f:
            email_content = f.readlines()  # Читаем файл построчно

        # Предобработка письма
        text_processor = TextProcessor()
        words = text_processor.preprocess(''.join(email_content))  # Преобразуем список строк обратно в строку

        # Классификация
        is_spam = db.classify(words)

        # Определяем тип письма
        email_type = "спам" if ("spam" in email_path) else "обычное"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        print(f"Это {BOLD}{email_type}{RESET} письмо.\n")

        # Печатаем только первые 7 строк письма
        print("Содержимое письма:")
        for line in email_content[:7]:  # Берем только первые 7 строк
            print(line.strip())  # Используем strip(), чтобы убрать лишние пробелы и символы новой строки

        print(f"Результат проверки: {BOLD}{'Спам' if is_spam else 'Не спам'}{RESET}\n")

    except Exception as e:
        print(f"Ошибка при обработке письма: {e}")


import argparse

def main():
    parser = argparse.ArgumentParser(description='Наивный байесовский классификатор для писем')
    
    # Создаем группу взаимно исключающих аргументов
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='Режим тренировки')
    group.add_argument('--validate', action='store_true', help='Режим проверки (валидации)')
    group.add_argument('--email', type=str, help='Путь к письму для проверки')
    
    parser.add_argument('--db_name', type=str, default='naive_bayes.db.csv', help='Имя файла базы данных')
    parser.add_argument('--dataset_path', type=str, default='../dataset', help='Путь к набору данных')
    
    args = parser.parse_args()

    # Инициализация компонентов
    email_loader = FileEmailLoader()
    text_processor = TextProcessor()
    data_manager = DataManager()
    
    # Если база данных существует, загружаем её
    if exists(args.db_name):
        db = data_manager.load_from_csv(args.db_name)
    else:
        db = NaiveBayesDB()

    # Выбор режима работы
    if args.train:
        train(db, data_manager, email_loader, text_processor, args.db_name, args.dataset_path)
    elif args.validate:
        validate(db, data_manager, email_loader, text_processor, args.db_name, args.dataset_path)
    elif args.email:
        custom_test(data_manager, db, args.email)
    else:
        print("Пожалуйста, выберите режим: --train или --validate или --email")

if __name__ == "__main__":
    main()

