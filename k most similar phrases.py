from typing import List, Tuple, Union

def top_k_similar_sentences(query: str, k: int = 5, return_scores: bool = True, print_results: bool = False) -> Union[List[Tuple[str, float]], List[str]]:
    """
    Return the top-k most similar sentences to `query`.
    - If return_scores is True, returns a list of (sentence, score) sorted by descending score.
    - If return_scores is False, returns a list of sentences only.
    - If print_results is True, prints the results (score then sentence).
    """
    q = embed_sentence(query)  # numpy array, shape (H,)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        q_norm = 1e-12
    q = q / q_norm

    # cosine similarities (emb_matrix_normed pre-normalized)
    sims = emb_matrix_normed @ q  # shape (N,)

    N = sims.shape[0]
    if k <= 0:
        return [] if not return_scores else []
    k = min(k, N)

    if k == 1:
        best_idx = int(np.argmax(sims))
        result = [(sentences[best_idx], float(sims[best_idx]))]
    else:
        # fast top-k selection, then sort those k by descending score
        topk_idx = np.argpartition(-sims, k-1)[:k]          # unsorted top-k indices
        topk_sorted_idx = topk_idx[np.argsort(-sims[topk_idx])]  # sorted by descending similarity
        result = [(sentences[i], float(sims[i])) for i in topk_sorted_idx]

    if print_results:
        for sent, score in result:
            print(f"{score:.6f}\t{sent}")

    if return_scores:
        return result
    else:
        return [sent for sent, _ in result]