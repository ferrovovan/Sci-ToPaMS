from typing import List, Tuple
import numpy as np

class ModelEvaluator:
	"""Класс для оценки модели классификации"""
	
	@staticmethod
	def calculate_metrics(y_true: List[bool], y_pred: List[bool], 
						posterior_probs: List[float]) -> dict:
		"""
		Вычисляет метрики качества классификации
		
		Args:
			y_true: истинные метки
			y_pred: предсказанные метки
			posterior_probs: апостериорные вероятности
			
		Returns:
			Словарь с метриками
		"""
		tp = np.sum([(pred and true) for pred, true in zip(y_pred, y_true)])
		tn = np.sum([(not pred and not true) for pred, true in zip(y_pred, y_true)])
		fp = np.sum([(pred and not true) for pred, true in zip(y_pred, y_true)])
		fn = np.sum([(not pred and true) for pred, true in zip(y_pred, y_true)])
		
		accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
		sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
		specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
		
		# Коэффициент Байеса (Likelihood Ratio)
		lr_plus = sensitivity / (1 - specificity) if specificity != 1 else float('inf')
		lr_minus = (1 - sensitivity) / specificity if specificity != 0 else float('inf')
		
		return {
			'accuracy': accuracy,
			'sensitivity': sensitivity,
			'specificity': specificity,
			'lr_plus': lr_plus,
			'lr_minus': lr_minus,
			'confusion_matrix': {
				'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
			}
		}
	
	@staticmethod
	def validate_model(db, email_loader, text_processor, dataset_path: str) -> dict:
		"""
		Полная валидация модели с расчетом всех метрик
		
		Returns:
			Словарь с результатами валидации
		"""
		y_true = []
		y_pred = []
		posterior_probs = []
		
		emails = email_loader.load_emails(f'{dataset_path}/valide')
		
		for email, is_spam in emails:
			words = text_processor.preprocess(email)
			spam_count, all_count = db.get_spam_all_count(words)
			probability = spam_count / all_count
			
			y_true.append(is_spam)
			y_pred.append(probability > 0.5)
			posterior_probs.append(probability)
		
		return ModelEvaluator.calculate_metrics(y_true, y_pred, posterior_probs)

