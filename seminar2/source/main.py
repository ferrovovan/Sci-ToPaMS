import argparse
from email_loader import FileEmailLoader
from text_processor import TextProcessor
from model_evaluator import ModelEvaluator
from naive_bayes_db import NaiveBayesDB
from data_manager import DataManager
from os.path import exists, isdir


def train(db, data_manager, email_loader, text_processor, db_name, dataset_path: str):
	"""Обучение модели"""
	for i in range(1, 7):
		emails = email_loader.load_emails(f'{dataset_path}/enron{i}')
		for email, is_spam in emails:
			words = text_processor.preprocess(email)
			db.train_on_single_mail(words, is_spam)

	data_manager.save_to_csv(db, db_name)
	print(f"Тренировка завершена. База данных сохранена в {db_name}.")


def validate(db, data_manager, email_loader, text_processor, db_name, dataset_path: str):
	"""Валидация модели с выводом метрик"""
	results = ModelEvaluator.validate_model(db, email_loader, text_processor, dataset_path)
	
	print("\nРезультаты валидации:")
	print(f"Точность (Accuracy): {results['accuracy']:.3f}")
	print(f"Чувствительность (Sensitivity): {results['sensitivity']:.3f}")
	print(f"Специфичность (Specificity): {results['specificity']:.3f}")
	print(f"Коэффициент Байеса LR+: {results['lr_plus']:.3f}")
	print(f"Коэффициент Байеса LR-: {results['lr_minus']:.3f}")
	
	cm = results['confusion_matrix']
	print(f"\nМатрица ошибок:")
	print(f"True Positive: {cm['tp']}")
	print(f"True Negative: {cm['tn']}")
	print(f"False Positive: {cm['fp']}")
	print(f"False Negative: {cm['fn']}")
	
	return results


def single_mail_test(data_manager: DataManager, db: NaiveBayesDB, email_path: str):
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



#import argparse  # вот теперь понадобится парсер аргументов

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
	
	### Проверка корректности аргументов
	# Если база данных существует, загружаем её
	if exists(args.db_name):
		db = data_manager.load_from_csv(args.db_name)
	else:
		db = NaiveBayesDB()
	if not args.db_name.endswith(".csv"):
		parser.error("Имя базы данных (--db_name) должно оканчиваться на .csv")
	if not isdir(args.dataset_path):
		print(args.dataset_path)
		parser.error(f"Указанный путь к набору данных (--dataset_path) не существует: {args.dataset_path}")

	# Выбор режима работы
	if args.train:
		train(db, data_manager, email_loader, text_processor, args.db_name, args.dataset_path)
	elif args.validate:
		validate(db, data_manager, email_loader, text_processor, args.db_name, args.dataset_path)
	elif args.email:
		single_mail_test(data_manager, db, args.email)
	else:
		print("Пожалуйста, выберите режим: --train или --validate или --email")


if __name__ == "__main__":
	main()

