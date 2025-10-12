import os
from typing import List, Tuple


class EmailLoader:
	"""Интерфейс для загрузки писем"""
	def load_emails(self, base_directory: str):
		"""
		Загружает письма из указанной директории.
		Должен возвращать список кортежей (текст письма, является ли спамом).
		"""
		raise NotImplementedError("Метод должен быть реализован в подклассах")

class FileEmailLoader(EmailLoader):
	"""Класс для загрузки писем из файловой структуры каталогов.

	Ожидается, что структура будет такой:
		 └── base_directory
		     ├── enron1
		     │   ├── ham
		     │   └── spam
		     ├── enron2
		     │   ├── ham
		     │   └── spam 
		     │
		     └── enronN
		         ├── ham 
		         └── spam
	"""
	
	@staticmethod
	def _read_file(file_path: str) -> str:
		"""Считывает текст письма из файла."""
		with open(file_path, "r", encoding="ASCII", errors="ignore") as f:
			return f.read()

	def _find_category_dirs(self, base_directory: str, category: str) -> List[str]:
		"""Находит все подкаталоги с данным названием (ham или spam)."""
		result = []
		for root, dirs, _ in os.walk(base_directory):
			if os.path.basename(root) == category:
				result.append(root)
		return result

	def _load_category(self, dir_path: str, category: str, count: int) -> List[Tuple[str, bool]]:
		"""Загружает все письма из одной категории (ham или spam)."""
		emails = []
		is_spam = (category == "spam")
		for filename in os.listdir(dir_path):
			if count == 0:
				break
			file_path = os.path.join(dir_path, filename)
			if os.path.isfile(file_path):
				emails.append((self._read_file(file_path), is_spam))
				count -= 1
		return emails


	def load_emails(self, base_directory: str, count: int = -1) -> List[Tuple[str, bool]]:
		emails = []
		for category in ("ham", "spam"):
			category_dir = self._find_category_dirs(base_directory, category)
			for dir_path in category_dir:
				emails.extend(self._load_category(dir_path, category, count))
				if count > 0 and len(emails) >= count:
					return emails[:count]
		return emails

