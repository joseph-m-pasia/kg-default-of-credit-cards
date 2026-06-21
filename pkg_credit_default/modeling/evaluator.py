typing import Dict
import numpy as np

from pkg_credit_default.utils.logger      import logger
from pkg_credit_default.modeling.plotter  import plot_metric_comparison

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate the trained model on the test set and log the results.
    Assuming a binary classification problem, this function computes accuracy, precision, recall, F1 score, and ROC AUC score.
    Input:
        model:      the trained model to evaluate
        X_test:     the test features
        y_test:     the test labels
    """
    logger.info("Evaluating model on the test set...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")
    logger.info(f"Test ROC AUC Score: {roc_auc:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

def select_champion_model(score_results: list, metric: str = "f1_score", plot_metrics: bool = False) -> tuple:

    """
    Select the champion model based on the specified metric and optionally plot the comparison.
    Input:
        score_results: list of tuples, e.g. [('random_forest', 0.82), ('logistic_regression', 0.80)]
        metric:        the specific metric to compare (e.g. "accuracy", "precision", "recall", "f1_score", "roc_auc")
        plot_metrics:  boolean indicating whether to plot the metric comparison across models
    """

    logger.info(f"Selecting champion model based on {metric}...")
    for model_name, score in score_results:
        logger.info(f"{model_name} - {metric}: {score:.4f}")
    
    if plot_metrics:
        logger.info(f"Plotting {metric} comparison across models...")
        model_scores = [(model, scores) for model, scores in score_results]
        plot_metric_comparison(model_scores, metric)

    champion_model = max(score_results, key=lambda x: x[1])
    logger.info(f"Champion model based on {metric}: {champion_model[0]} with {metric} = {champion_model[1]:.4f}")

    return champion_model



#========= EXAMPLE USAGE =========
if __name__ == "__main__":
    """ 
    This is just an example of how to use the evaluate_model function.
    In practice, you would call this function after training your model and splitting your data into train/test sets.
    """

    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Create a synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    test = evaluate_model(model, X_test, y_test)
    print(test)

    # Select champion model example
    score_results = [("random_forest", 0.82), 
                     ("logistic_regression", 0.80), 
                     ("xgb_regressor", 0.85), 
                     ("knn", 0.78), 
                     ("lightgbm", 0.83)]

    select_champion_model(score_results, metric="f1_score", plot_metrics=False)  
