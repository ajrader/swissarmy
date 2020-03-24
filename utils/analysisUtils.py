from scipy import stats

def fast_linear_regress(x,y):
    slope, intercept, rvanue, pval, stderr = stats.linregress(x, y)
