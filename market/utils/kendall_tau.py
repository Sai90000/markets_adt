import scipy.stats as stats

def kendall_tau_from_scores(pred, target):
    pred_ranks = stats.rankdata(pred)
    target_ranks = stats.rankdata(target)
    tau, p_value = stats.kendalltau(pred_ranks, target_ranks)
    return tau, p_value