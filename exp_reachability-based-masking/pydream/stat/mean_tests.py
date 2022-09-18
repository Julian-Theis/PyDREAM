from scipy.stats import wilcoxon,ttest_rel

def run_wilcoxon_signed_rank_test(sample1, sample2, alpha=0.05, print_results=True):
    stat, p = wilcoxon([a_i - b_i for a_i, b_i in zip(sample1, sample2)])
    if print_results:
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        if p > alpha:
            print('Samples look similar  (fail to reject H0) at significance level ', alpha)
        else:
            print('Samples do not look similar (reject H0) at significance level ', alpha)
    return stat, p

def run_paired_t_test(sample1, sample2, alpha=0.05, print_results=True):
    stat, p = ttest_rel(sample1, sample2)
    if print_results:
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        if p > alpha:
            print('Samples look similar (fail to reject H0) at significance level ', alpha)
        else:
            print('Samples do not look similar (reject H0) at significance level ', alpha)
    return stat, p