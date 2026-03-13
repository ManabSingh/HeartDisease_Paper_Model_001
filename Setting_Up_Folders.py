import os


def create_project_structure(project_root: str = "heart-disease-cross-domain-xai") -> None:
	file_paths = [
		"data/raw/cleveland/cleveland.csv",
		"data/raw/statlog/statlog.csv",
		"data/processed/harmonized/cleveland_harmonized.csv",
		"data/processed/harmonized/statlog_harmonized.csv",
		"data/processed/cleaned/cleveland_clean.csv",
		"data/processed/cleaned/statlog_clean.csv",
		"data/processed/balanced/cleveland_smoteenn.csv",
		"notebooks/01_data_exploration.ipynb",
		"notebooks/02_preprocessing.ipynb",
		"notebooks/03_feature_engineering.ipynb",
		"notebooks/04_model_training.ipynb",
		"notebooks/05_stacking_ensemble.ipynb",
		"notebooks/06_internal_evaluation.ipynb",
		"notebooks/07_shap_analysis_source.ipynb",
		"notebooks/08_cross_domain_testing.ipynb",
		"notebooks/09_shap_analysis_target.ipynb",
		"src/data_acquisition/download_datasets.py",
		"src/data_acquisition/feature_harmonization.py",
		"src/preprocessing/missing_value_imputation.py",
		"src/preprocessing/outlier_removal.py",
		"src/preprocessing/encoding.py",
		"src/preprocessing/scaling.py",
		"src/preprocessing/smoteenn_balancing.py",
		"src/models/train_xgboost.py",
		"src/models/train_lightgbm.py",
		"src/models/train_random_forest.py",
		"src/models/stacking_model.py",
		"src/optimization/optuna_tuning.py",
		"src/optimization/gridsearch_tuning.py",
		"src/evaluation/metrics.py",
		"src/evaluation/cross_validation.py",
		"src/explainability/shap_source_analysis.py",
		"src/explainability/shap_target_analysis.py",
		"src/explainability/feature_shift_analysis.py",
		"src/cross_domain/statlog_testing.py",
		"models/trained_models/xgboost.pkl",
		"models/trained_models/lightgbm.pkl",
		"models/trained_models/random_forest.pkl",
		"models/trained_models/stacked_model.pkl",
		"results/metrics/cleveland_metrics.csv",
		"results/metrics/statlog_metrics.csv",
		"results/shap_plots/cleveland_summary.png",
		"results/shap_plots/statlog_summary.png",
		"results/feature_shift/attribution_comparison.csv",
		"figures/model_architecture.png",
		"figures/shap_source_plot.png",
		"figures/shap_target_plot.png",
		"reports/research_paper/manuscript.docx",
		"reports/presentation/project_ppt.pptx",
		"requirements.txt",
		"README.md",
		"main.py",
	]

	for relative_file_path in file_paths:
		full_path = os.path.join(project_root, relative_file_path)
		parent_dir = os.path.dirname(full_path)

		if parent_dir:
			os.makedirs(parent_dir, exist_ok=True)

		with open(full_path, "a", encoding="utf-8"):
			pass

	print(f"Project structure created at: {os.path.abspath(project_root)}")


if __name__ == "__main__":
	create_project_structure()