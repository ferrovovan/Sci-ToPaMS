#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import shutil
from pathlib import Path
from typing import List, Dict


class MailCategory:
	"""Класс для работы с подкатегориями писем (ham/spam)."""

	def __init__(self, path: Path):
		self.path = path
		self._count_cache = None

	def count(self) -> int:
		"""Подсчитать количество писем (лениво)."""
		if self._count_cache is None:
			self._count_cache = sum(1 for _ in self.path.glob("*") if _.is_file())
		return self._count_cache

	def pick_random(self, n: int) -> List[Path]:
		"""Выбрать случайные письма."""
		all_files = [f for f in self.path.glob("*") if f.is_file()]
		return random.sample(all_files, n)

	def move_files(self, files: List[Path], dest_category: "MailCategory"):
		"""Переместить выбранные письма в другую категорию."""
		for f in files:
			shutil.move(str(f), dest_category.path / f.name)
		# сбросить кэш
		self._count_cache = None
		dest_category._count_cache = None


class EnronFolder:
	"""Описывает одну папку (enron1, enron2, ... или valide)."""

	def __init__(self, path: Path):
		self.path = path
		self.name = path.name
		self.ham = MailCategory(path / "ham")
		self.spam = MailCategory(path / "spam")
		self.summary = path / "Summary.txt"

	def total_emails(self) -> int:
		"""Общее количество писем в папке."""
		return self.ham.count() + self.spam.count()

	def category(self, kind: str) -> MailCategory:
		"""Возвратить категорию по имени."""
		if kind == "ham":
			return self.ham
		elif kind == "spam":
			return self.spam
		raise ValueError(f"Неизвестный тип категории: {kind}")

	def __repr__(self):
		return f"<EnronFolder {self.name}: {self.total_emails()} писем>"


class MailDatabase:
	"""Композиция всех папок и их распределитель."""

	def __init__(self, base_dir: Path):
		self.base_dir = base_dir
		self.folders: Dict[str, EnronFolder] = {}
		self._load_folders()

	def _load_folders(self):
		"""Загрузить все подкаталоги как EnronFolder."""
		for item in self.base_dir.iterdir():
			if item.is_dir():
				self.folders[item.name] = EnronFolder(item)

	def total_emails(self) -> int:
		"""Общее количество писем по всей базе."""
		return sum(f.total_emails() for f in self.folders.values() if f.name != "valide")

	def distribute(self, valid_rate: float):
		"""Распределить часть данных в папку valide."""
		valide = self.folders.get("valide")
		if not valide:
			raise RuntimeError("Папка 'valide' не найдена!")

		total = self.total_emails()
		target_count = int(total * valid_rate)
		print(f"→ Переносим {target_count} писем ({valid_rate*100:.1f}%) в {valide.name}")

		# пропорционально от каждой папки
		for folder in self.folders.values():
			if folder.name == "valide":
				continue
			part = int(folder.total_emails() * valid_rate)
			for kind in ["ham", "spam"]:
				cat = folder.category(kind)
				dest = valide.category(kind)
				n = int(cat.count() * valid_rate)
				files = cat.pick_random(n)
				cat.move_files(files, dest)


def main():
	import argparse
	parser = argparse.ArgumentParser(description="Распределитель писем по базе")
	parser.add_argument("--valid", type=float, required=True,
						help="Процент писем для валидации (например, 15 означает 15%)")
	args = parser.parse_args()

	db = MailDatabase(Path("database"))
	db.distribute(valid_rate=args.valid / 100.0)


if __name__ == "__main__":
	main()

