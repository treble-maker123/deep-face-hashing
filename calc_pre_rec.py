from sklearn.metrics import precision_recall_curve
from pdb import set_trace

def calc_pre_rec(hamm_dist, gt, radius):
    '''
    Calculates the precision-recall curve values.
    '''
    # distance within radius counts as 0
    dist = hamm_dist * (hamm_dist > radius)
    # normalize the distance values, so the smaller distance, the closer to 1
    max_val = dist.max()
    scores = ((max_val - dist) / max_val) ** 2
    scores[scores != scores] = 1
    # calculate the curves
    pre_curve, rec_curve, _ = precision_recall_curve(gt.ravel(), scores.ravel())

    # pred == 1 is what the model believes to be the person
    pred = (dist == 0).astype("int8")
    # true positives
    tp = (pred * gt).sum(axis=0)
    # false positives
    fp = (pred * (gt == 0)).astype("int8").sum(axis=0)
    # false negative
    fn = ((pred == 0) * gt).astype("int8").sum(axis=0)
    # recall
    rec = tp / (tp + fn)
    rec[rec != rec] = 0
    # precision
    pre = tp / (tp + fp)
    pre[pre != pre] = 0
    # harmonic mean
    hmean = 2 * (pre * rec) / (pre + rec)
    hmean[hmean != hmean] = 0

    return pre.mean(), rec.mean(), hmean.mean(), pre_curve, rec_curve
