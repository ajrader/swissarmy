import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn import metrics


# function to plot feature_importances for RF
def plotFI(forest, featureNames=[], max_features=30):  # ,autoscale=True,headroom=0.05):
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
        featureNames = list(map(str, indices))

    fN2 = [featureNames[a] for a in indices]
    print("Feature ranking:")

    for f in range(len(indices)):
        print("%d. feature %d=%s (%f)" % (f + 1, indices[f], featureNames[indices[f]], featureImportances[indices[f]]))

    # Plot the feature importances of the forest
    # define a cutoff in terms of feature_importance
    if nfeatures <= max_features:
        kfeatures = nfeatures  # keep all if smaller than 30
    else:
        kfeatures = max_features

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
        title_str = 'Normalized '+title_str
        cb_label = 'fraction'
        fmt_ann = "{x:.3f}"
    else:
        fmt_ann = "{x:d}"
        cb_label = 'count'

    true_classes = [str(a) for a in range(my_c.shape[0])]
    pred_classes = [str(a) for a in range(my_c.shape[1])]
    #akws = {'ha': 'center', 'va': 'center_baseline'}
    #sns.heatmap(my_c, annot=True, fmt='d', annot_kws=akws, cmap='Blues')
    fig, ax = plt.subplots()

    im, cbar = heatmap(my_c, true_classes, pred_classes, ax=ax,
                       cmap="Blues", cbarlabel=cb_label)
    texts = annotate_heatmap(im, valfmt=fmt_ann)

    fig.tight_layout()

    plt.ylabel('True')
    # plt.yticks
    plt.xlabel('Assigned')
    plt.title('Normalized ' + title_str)
    plt.show()


    return

def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

        # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    #ax.tick_params(top=True, bottom=False,
    #               labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #         rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.grid(which='major', b=False, axis='both')

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_roc_curve(target_test, target_predicted_proba):
    try:
        y_pred = target_predicted_proba[:, 1]
    except IndexError:
        y_pred = target_predicted_proba

    fpr, tpr, thresholds = metrics.roc_curve(target_test, y_pred)

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


def plot_precision_recall_curve(target_test, target_predicted_proba, color='b'):
    try:
        y_pred = target_predicted_proba[:, 1]
    except IndexError:
        y_pred = target_predicted_proba

    precision, recall, thresh = metrics.precision_recall_curve(target_test, y_pred)
    #step_kwargs = ({'step': 'post'}
    #               if 'step' in signature(plt.fill_between).parameters
    #               else {})
    plt.step(recall, precision, color=color, alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color=color, step='post' ) #'**step_kwargs)
    average_precision = metrics.average_precision_score(target_test, y_pred)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.3f}'.format(average_precision))
    return average_precision