class NaiveBayesDB:
    """
    Внутренняя логика классификатора наивного Байеса
    """
    def __init__(self):
        self.known_words = {}

    def train(self, words: set, is_spam: bool) -> None:
        """Тренировка на письме"""
        for word in words:
            if word not in self.known_words:
                self.known_words[word] = [0, 0]
            if is_spam:
                self.known_words[word][0] += 1  # spam_count
            self.known_words[word][1] += 1  # total_count

    def classify(self, words: set) -> bool:
        """Классификация письма на основе Naive Bayes"""
        spam_count: int = 0
        all_count: int  = 0
        for word in tuple(words):
            if word in self.known_words:
                spam_count += self.known_words[word][0]
                all_count  += self.known_words[word][1]
            else:
                # сглаживание Лапласа
                spam_count += 1
                all_count  += 2

        probability: float = spam_count / all_count
        # print(probability)
        if probability > 0.5:
            return True
        return False

