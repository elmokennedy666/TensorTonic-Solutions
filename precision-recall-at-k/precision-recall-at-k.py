def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here

    intersection = len(set(recommended[:k]).intersection(set(relevant)))
    
    pr = intersection / k
    rec = intersection / len(relevant)
    return [pr, rec]