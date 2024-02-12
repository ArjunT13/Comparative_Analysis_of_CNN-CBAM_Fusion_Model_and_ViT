from roc_utils import *
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import sem
from sklearn.metrics import roc_auc_score
 

def ROC_with_CI_outcomes(true_variables, pred_variables, outcomes, model):
    names = list(mcolors.CSS4_COLORS)
    for i in range(len(outcomes)):
        plt.figure()
        plot_roc_bootstrap(X=pred_variables[i], y=true_variables[i], pos_label=True,
                   n_bootstrap=1500,
                   random_state=10,
                   stratified=True,
                   show_boots=False,
                   label=outcomes[i], color=names[i+15])
        plt.savefig("ROC_Outcomes/"+model+"/ROC_"+outcomes[i]+".png", dpi = 300, bbox_inches = 'tight')

        print(outcomes[i]+" - Original ROC area: {:0.3f}".format(roc_auc_score(true_variables[i], pred_variables[i])))
        n_bootstraps = 1500
        rng_seed = 42  # control reproducibility
        bootstrapped_scores = []

        rng = np.random.RandomState(rng_seed)
        for x in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, len(pred_variables[i]), len(pred_variables[i]))
            selected_true_variables = [true_variables[i][j] for j in indices]
            selected_pred_variables = [pred_variables[i][j] for j in indices]
            if len(np.unique(selected_true_variables)) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue

            score = roc_auc_score(selected_true_variables, selected_pred_variables)
            bootstrapped_scores.append(score)
        
        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()

        # Computing the lower and upper bound of the 90% confidence interval
        # You can change the bounds percentiles to 0.025 and 0.975 to get
        # a 95% confidence interval instead.
        confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
        print(outcomes[i]+" - Confidence interval (90%) for the score: [{:0.3f} - {:0.3}]".format(
            confidence_lower, confidence_upper))
        confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
        print(outcomes[i]+" - Confidence interval (95%) for the score: [{:0.3f} - {:0.3}]".format(
            confidence_lower, confidence_upper))


def read_file(filename):
    testY = []
    y_pred1 = []
    with open(filename, 'r') as file:
        for line in file:
            item1, item2 = map(str.strip, line.split(', '))
            item2 = float(item2[1:-1])
            y_pred1.append(item2)
            testY.append(float(item1))
    return testY, y_pred1


true_variables = []
pred_variables = []

filename = 'vit_chf.txt'
testY, y_pred1 = read_file(filename)
true_variables.append(testY)
pred_variables.append(y_pred1)
print("File read")

filename1 = 'vit_mortality.txt'
testY, y_pred2 = read_file(filename1)
true_variables.append(testY)
pred_variables.append(y_pred2)
print("File read1")

filename2 = 'vit_stroke.txt'
testY, y_pred3 = read_file(filename2)
true_variables.append(testY)
pred_variables.append(y_pred3)
print("File read2")

outcomes = ["CHF", "Mortality", "Stroke"]

ROC_with_CI_outcomes(true_variables, pred_variables, outcomes, "ViT")
print("Done")