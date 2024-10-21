import os

class EmailLoader:
    """Интерфейс для загрузки писем"""
    def load_emails(self, base_directory: str):
        """
        Загружает письма из указанной директории.
        Должен возвращать список кортежей (текст письма, является ли спамом).
        """
        raise NotImplementedError("Этот метод должен быть реализован в подклассах")

class FileEmailLoader(EmailLoader):
    """Класс для загрузки писем из файловой структуры каталогов"""
    def load_emails(self, base_directory: str, count: int = -1):
        """
        Загружает письма из структуры каталогов.
        
        Ожидается, что структура будет такой:
        └── base_directory
            ├── enron1
            │   ├── ham
            │   └── spam
            ├── enron2
            │   ├── ham
            │   └── spam
            └── enronN
                ├── ham
                └── spam

        Аргументы:
        base_directory (str): Корневая директория с подкаталогами ham и spam.
        
        Возвращает:
        list of tuples: [(текст письма, является ли спамом), ...]
        """
        emails = []
        categories: tuple = ('ham', 'spam')
        
        # Проход по директориям enron1, enron2 и т.д.
        for root, dirs, files in os.walk(base_directory):
            # Проверка, находится ли текущий путь в категории 'ham' или 'spam'
            for category in categories:
                if category in root:  # Если в пути есть папка ham или spam
                    is_spam = (category == 'spam')
                    # Чтение всех файлов (писем) в каталоге ham или spam
                    for file in files:
                        if count == 0:
                            break
                        file_path = os.path.join(root, file)
                        if os.path.isfile(file_path):  # Проверка, что это файл
                            with open(file_path, 'r', encoding='ASCII', errors='ignore') as f:
                                email_content = f.read()
                                emails.append((email_content, is_spam))
                            count -= 1
        return emails

