import string

class TextProcessor:
	@staticmethod
	def preprocess(email: str) -> set:
		"""Очистка текста и приведение к множеству уникальных слов"""
		translator = str.maketrans('', '', string.punctuation)
		cleaned_email = email.translate(translator)
		words = cleaned_email.lower().split()
		return set(words)

