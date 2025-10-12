import os
import shutil
import random


def count_emails(directory):
    """Подсчёт количества писем по категориям в заданной директории."""
    ham_count = 0
    spam_count = 0
    for root, dirs, files in os.walk(directory):
        if 'ham' in root:
            ham_count += len(files)
        elif 'spam' in root:
            spam_count += len(files)
    return ham_count, spam_count


def collect_all_emails(training_dirs):
    """Сбор всех писем из обучающих директорий."""
    ham_files = []
    spam_files = []

    for directory in training_dirs:
        for root, dirs, files in os.walk(directory):
            if 'ham' in root:
                ham_files.extend(os.path.join(root, file) for file in files)
            elif 'spam' in root:
                spam_files.extend(os.path.join(root, file) for file in files)

    return ham_files, spam_files


def distribute_emails(ham_files, spam_files, valide_dir, training_dirs, target_ratio: float):
    """Распределение писем по тренировочным и валидационной папке."""
    random.shuffle(ham_files)
    random.shuffle(spam_files)

    ham_valide_count = int(len(ham_files) * target_ratio)
    spam_valide_count = int(len(spam_files) * target_ratio)

    print(f'Для валидации будет перемещено {ham_valide_count} обычных писем и {spam_valide_count} спам писем')

    def move_files(files, destination_dir, category, count):
        """Перемещение файлов в целевую директорию с обработкой существующих файлов."""
        valide_path = os.path.join(destination_dir, category)
        os.makedirs(valide_path, exist_ok=True)
        for i in range(count):
            source_file = files[i]
            destination_file = os.path.join(valide_path, os.path.basename(source_file))

            # Если файл уже существует, переименовываем его
            if os.path.exists(destination_file):
                base, ext = os.path.splitext(destination_file)
                counter = 1
                while os.path.exists(destination_file):
                    destination_file = f"{base}_{counter}{ext}"
                    counter += 1

            shutil.move(source_file, destination_file)

    # Перемещение писем в валидационную папку
    move_files(ham_files, valide_dir, 'ham', ham_valide_count)
    move_files(spam_files, valide_dir, 'spam', spam_valide_count)

    print(f'Перемещено {ham_valide_count} обычных писем и {spam_valide_count} спам писем в {valide_dir}')

    # Теперь перемещение из валидационной папки обратно в тренировочные
    ham_valide_files = [os.path.join(valide_dir, 'ham', f) for f in os.listdir(os.path.join(valide_dir, 'ham'))]
    spam_valide_files = [os.path.join(valide_dir, 'spam', f) for f in os.listdir(os.path.join(valide_dir, 'spam'))]

    # Перемещаем из валидации в тренировочные по 80% обратно
    move_files(ham_valide_files, random.choice(training_dirs), 'ham', len(ham_valide_files) - ham_valide_count)
    move_files(spam_valide_files, random.choice(training_dirs), 'spam', len(spam_valide_files) - spam_valide_count)


def main():
    base_directory = "database"
    training_dirs = [os.path.join(base_directory, f'enron{i}') for i in range(1, 7)]
    valide_dir = os.path.join(base_directory, 'valide')
    
    if not os.path.exists(base_directory):
    	print("Ошибка: нужна папка database!")
    	sys.exit(1)

    # Подсчёт писем в тренировочных директориях
    total_ham = 0
    total_spam = 0
    for training_dir in training_dirs:
        ham_count, spam_count = count_emails(training_dir)
        total_ham += ham_count
        total_spam += spam_count
        print(f'{training_dir}: {ham_count} обычных, {spam_count} спам писем')

    # Подсчёт писем в валидационной директории
    valide_ham, valide_spam = count_emails(valide_dir)
    print(f'{valide_dir}: {valide_ham} обычных, {valide_spam} спам писем')

    # Сбор всех писем из обучающих директорий
    ham_files, spam_files = collect_all_emails(training_dirs)

    # Распределение писем (20% в valide, 80% в тренировочные папки)
    distribute_emails(ham_files, spam_files, valide_dir, training_dirs, target_ratio=0.2)

if __name__ == "__main__":
    main()

