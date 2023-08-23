import torch

def goodness(activations, reduction="sum"):
    assert reduction in ["sum", "mean", None], "reduction must be either 'mean' or 'sum'"
    score = activations.square()
    if reduction == "sum":
        score = score.sum(dim=1)
    elif reduction == "mean":
        score = score.mean(dim=1)

    return score


def goodness_loss(pos_actvs, neg_actvs, threshold=2.0, mode="maximise"):
    pos_goodnesses = goodness(pos_actvs)
    neg_goodnesses = goodness(neg_actvs)
    diff_goodnesses = (pos_goodnesses.mean() - neg_goodnesses.mean()).abs() # For logging purposes

    if mode == "maximise":
        pos_goodnesses = -pos_goodnesses
    elif mode == "minimise":
        neg_goodnesses = -neg_goodnesses
    
    loss = torch.cat([pos_goodnesses, neg_goodnesses], dim=0).mean()
    return loss, diff_goodnesses


def prob_loss(pos_actvs, neg_actvs, threshold=2.0, mode="maximise"):
    pos_probs = torch.sigmoid(goodness(pos_actvs) - threshold)
    neg_probs = torch.sigmoid(goodness(neg_actvs) - threshold)
    diff_probs = (pos_probs.mean() - neg_probs.mean()).abs() # For logging purposes

    if mode == "maximise":
        pos_probs = 1 - pos_probs
    elif mode == "minimise":
        neg_probs = 1 - neg_probs

    loss = torch.cat([pos_probs, neg_probs], dim=0).mean()
    return loss, diff_probs


def log_loss(pos_actvs, neg_actvs, threshold=2.0, mode="maximise"):
    
    pos_logits = goodness(pos_actvs, "mean") - threshold
    neg_logits = goodness(neg_actvs, "mean") - threshold
    diff_logits = (pos_logits.mean() - neg_logits.mean()).abs() # For logging purposes

    if mode == "maximise":
        pos_logits = -pos_logits
    elif mode == "minimise":
        neg_logits = -neg_logits
    
    all_logits = torch.cat([pos_logits, neg_logits], dim=0)
    exp_logits = torch.nan_to_num(torch.exp(all_logits))
    loss = torch.log(1 + exp_logits).mean()
    return loss, diff_logits