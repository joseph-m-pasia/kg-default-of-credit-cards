from pkg_credit_default.utils.logger            import logger
from pkg_credit_default.modeling.trainer        import train_model

import matplotlib.pyplot as plt

def plot_metric_comparison(score_results, metric):
    
    """ 
    Plot a bar chart comparing the specified metric across different models.
    Input:
        score_results: a list of tuples [(model, score), ...] for the specific metric
        metric:        the specific metric to compare (e.g. "accuracy")
    """
    model_names = [model for model, score in score_results]
    metric_scores = [score for model, score in score_results]
    
    plt.figure(figsize=(5, 3))
    bars = plt.bar(model_names, metric_scores, color='skyblue')
    plt.xlabel('Model Type')
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
    plt.ylim(0, 1)
    
    # Add score labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')
    
    plt.show()