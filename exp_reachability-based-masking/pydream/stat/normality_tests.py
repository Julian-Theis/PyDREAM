"""
Code originates from: https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/

"""

from scipy.stats import shapiro, normaltest, anderson

"""
Shapiro-Wilk Test of Normality
The Shapiro-Wilk Test is more appropriate for small sample sizes (< 50 samples), but can also handle sample sizes as large as 2000.
The Shapiro-Wilk test is used as a numerical means of assessing normality.
"""
def run_shapiro_wilk_normality_test(data, alpha=0.05, print_results=True):
    stat, p = shapiro(data)
    if print_results:
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0) at significance level ', alpha)
        else:
            print('Sample does not look Gaussian (reject H0) at significance level ', alpha)
    return stat, p

def run_dagostino_pearson_test(data, alpha, print_results=True):
    stat, p = normaltest(data)
    if print_results:
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0) at significance level ', alpha)
        else:
            print('Sample does not look Gaussian (reject H0) at significance level ', alpha)
    return stat, p

def run_anderson_darling(data, print_results=True):
    result = anderson(data)
    print('Statistic: %.3f' % result.statistic)
    if print_results:
        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            if result.statistic < result.critical_values[i]:
                print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
            else:
                print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
    return result


