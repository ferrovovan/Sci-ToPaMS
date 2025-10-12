import csv
from naive_bayes_db import NaiveBayesDB

class DataManager:
	"""
	Интерфейс для работы с внешней системой (файловой системой, базой данных и т.д.)
	"""
	def save_to_csv(self, db: NaiveBayesDB, filepath: str) -> None:
		"""Сохранение базы данных в CSV файл"""
		with open(filepath, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['word', 'spam_count', 'all_count'])
			for word, counts in db.known_words.items():
				writer.writerow([word, counts[0], counts[1]])  # 'spam_count', 'total_count'

	def load_from_csv(self, filepath: str) -> NaiveBayesDB:
		"""Загрузка базы данных из CSV файла"""
		db = NaiveBayesDB()
		with open(filepath, 'r') as csvfile:
			reader = csv.reader(csvfile)
			next(reader)  # Пропускаем заголовок
			for row in reader:
				word, spam_count, all_count = row
				db.known_words[word] = [int(spam_count), int(all_count)]
		return db

