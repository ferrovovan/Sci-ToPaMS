class NaiveBayesDB:
	"""
	Внутренняя логика классификатора наивного Байеса
	"""
	def __init__(self):
		self.known_words: dict[str, dict] = {}
		self.total_samples = 0
		self.spam_samples = 0

	def train_on_single_mail(self, words: set, is_spam: bool) -> None:
		"""Тренировка на одном письме"""
		self.total_samples += 1
		if is_spam:
			self.spam_samples += 1

		for word in words:
			if word not in self.known_words:
				self.known_words[word] = [0, 0]

			if is_spam:
				self.known_words[word][0] += 1 # 'spam_count'
			self.known_words[word][1] += 1        # 'total_count'

	def get_spam_all_count(self, words: set):  #  -> Tuple[int, int]
		"""Возвращает spam_count и all_count для набора слов"""
		spam_count: int = 0
		all_count: int  = 0
		
		for word in words:
			if word in self.known_words:
				spam_count += self.known_words[word][0]  # 'spam_count'
				all_count  += self.known_words[word][1]  # 'total_count'
			else:
				# сглаживание Лапласа
				spam_count += 1
				all_count  += 2

		return spam_count, all_count

	def classify(self, words: set) -> bool:
		"""Классификация содержимого письма"""
		spam_count, all_count = self.get_spam_all_count(words)
		probability = spam_count / all_count
		return probability > 0.5

	def get_prior_probability(self) -> float:
		"""Возвращает априорную вероятность спама"""
		return self.spam_samples / self.total_samples if self.total_samples > 0 else 0.5

