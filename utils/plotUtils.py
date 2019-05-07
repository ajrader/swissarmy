import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn import metrics


# function to plot feature_importances for RF
def plotFI(forest, featureNames=[]):  # ,autoscale=True,headroom=0.05):
    """
    forest is the model to be graphed.
    featureNames is the list of features to be displayed

    """
    # if autoscale:
    #    x_scale = forest.feature_importances_.max()+ headroom
    # else:
    #    x_scale = 1
    # define the type of model
    model_type = type(forest)

    featureImportances = forest.feature_importances_
    # sort the importances from biggest to least
    indices = np.argsort(featureImportances)[::-1]
    estimators = forest.estimators_
    # calculate the variance over the forest
    if isinstance(estimators, list):
        std = np.std([tree.feature_importances_ for tree in estimators], axis=0)
    elif isinstance(estimators, np.ndarray):
        feature_importances = []
        for estimator in estimators:
            for tree in estimator:
                feature_importances.append(tree.feature_importances_)
        std = np.std(feature_importances, axis=0)

    # print summary statement
    nfeatures = len(featureImportances)
    print("Number of Features: %d" % (nfeatures))
    print("Number of Trees: %d" % (len(estimators)))

    # print featureNames
    if len(featureNames) == 0:
        featureNames = map(str, indices)

    fN2 = [featureNames[a] for a in indices]
    print("Feature ranking:")

    for f in range(len(indices)):
        print("%d. feature %d=%s (%f)" % (f + 1, indices[f], featureNames[indices[f]], featureImportances[indices[f]]))

    # Plot the feature importances of the forest
    # define a cutoff in terms of feature_importance
    if nfeatures <= 30:
        kfeatures = nfeatures  # keep all if smaller than 30
    else:
        kfeatures = 30

    kindices = indices[:kfeatures]
    plt.title("Feature importances")
    plt.barh(range(len(kindices)), featureImportances[kindices],
             color="steelblue", xerr=std[kindices], align="center", ecolor='k')  # ,lw=2)
    # results.plot(kind="barh", figsize=(width,len(results)/4), xlim=(0,x_scale))
    plt.yticks(range(len(kindices)), fN2)
    # grid(True)
    c1 = 'value'
    c2 = 'std'
    tdata = np.vstack([featureImportances[indices], std[indices]])
    df = pd.DataFrame(data=tdata.T, index=fN2, columns=[c1, c2])
    return df



def plot_conf_matrix(y_true, y_pred, normed=True, title_str=None, **kwargs):
    my_c = metrics.confusion_matrix(y_true, y_pred)

    print(metrics.matthews_corrcoef(y_true, y_pred))
    if title_str is None:
        title_str = 'Confusion Matrix'
    if normed:
        cm_normalized = my_c.astype('float') / my_c.sum(axis=1)[:, np.newaxis]
        my_c = cm_normalized
        plt.title('Normalized '+title_str)
    else:
        plt.title(title_str)

    sns.heatmap(my_c, annot=True, fmt='', cmap='Blues')
    plt.ylabel('True')
    # plt.yticks
    plt.xlabel('Assigned')
    plt.show()

    return


def plot_roc_curve(target_test, target_predicted_proba):
    fpr, tpr, thresholds = metrics.roc_curve(target_test, target_predicted_proba[:, 1])

    roc_auc = metrics.auc(fpr, tpr)
    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specificity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    return roc_auc
