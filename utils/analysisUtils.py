from scipy import stats

def fast_linear_regress(x,y):
    slope, intercept, rvanue = stats.linregress(x,y)
