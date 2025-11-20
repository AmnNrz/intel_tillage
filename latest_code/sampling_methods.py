import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, StratifiedKFold
from time import time
from tqdm import trange
from collections import defaultdict
from datetime import datetime
import os, re, csv

import dpp_sampler as sampler
import sampling_utils as utils

# Defines a method for training on the entire training set.
def full_model(data, labels, test_data, test_labels, model, logger):
    logger.info("Full")
    model, test_dict = utils.utils.train_and_test(data, labels, test_data, test_labels, model, logger)
    logger.info(f"Accuracy of the model on the test images: {round(test_dict['test_accuracy'], 4)}")
    return test_dict

def random_sampling(data, labels, test_data, test_labels, model, logger, batch_size=10, sub_sample_count=200):
    """
    Train models on progressively larger random subsets:
    step, 2*step, ..., sub_sample_count samples.
    """
    logger.info(f"\033[93mUniform sampling\033[0m")
    results = {}
    sub_ids = np.random.permutation(labels.shape[0])  # one fixed random order
    for k in range(batch_size, sub_sample_count + 1, batch_size):
        #logger.info(f'\033[93mUniform with {k} samples \033[0m')
        train_ids = sub_ids[:k]
        model, test_dict = utils.utils.train_and_test(data.iloc[train_ids], labels[train_ids], test_data, test_labels, model, logger)
        logger.info(f"\033[92mUniform\033[0m with {k} samples : Accuracy of the model on the test images: {round(test_dict['test_accuracy'], 4)}")
        results[k] = test_dict
    return results

def passive_dpp(data, labels, test_data, test_labels, model, logger, batch_size=10, sub_sample_count=200, steps=200, alpha=4):
    """
    Progressive Passive DPP (nested):
    add `step` points per round until `sub_sample_count`.
    Uses cond_ids to guarantee nesting.
    """
    logger.info(f"\033[93mPassive DPP\033[0m")
    N = labels.shape[0]
    train_ids = np.array([], dtype=int)
    results = {}

    # Build batch sizes: step, step, ..., remainder
    batches = [batch_size] * (sub_sample_count // batch_size)
    rem = sub_sample_count % batch_size
    if rem:
        batches.append(rem)

    for i, bsz in enumerate(batches, start=1):
        # diversity-only (passive): gamma=0.0, uniform weights
        new_ids = sampler.sample_ids_mc(
            data.values,
            np.ones(N),
            k=bsz,
            alpha=alpha,
            gamma=0.0,           # <-- passive
            cond_ids=train_ids,  # <-- ensures nesting
            steps=steps
        )
        train_ids = np.append(train_ids, new_ids).astype(int)

        # evaluate at current size
        with trange(1, desc=f"Passive DPP k={len(train_ids)}", leave=False) as _:
            model, test_dict = utils.utils.train_and_test(
                data.iloc[train_ids], labels[train_ids], test_data, test_labels, model, logger
            )
        logger.info(f"\033[92mPassive DPP\033[0m with {len(train_ids)} samples : Accuracy of the model on the test images: {round(test_dict['test_accuracy'], 4)}")
        results[len(train_ids)] = test_dict
    return results


# Defines a method for Active DPP (steps = 1000) and Active Greedy-DPP (steps = 0), where steps is the 
# number of Monte Carlo steps.

def active_dpp(data, labels, test_data, test_labels, model, logger, batch_size=10, sub_sample_count=200, steps=200, gamma=5, alpha=4, num_models=10, eps=1e-8):
    if steps == 0:
        logger.info(f"\033[93mActive Greedy-DPP\033[0m")
    else:
        logger.info(f"\033[93mActive DPP\033[0m")

    N = labels.shape[0]
    train_ids = np.array([], dtype=int)

    # progressive batches (e.g., 10,10,... until sub_sample_count)
    batches = [batch_size] * (sub_sample_count // batch_size)
    rem = sub_sample_count % batch_size
    if rem:
        batches.append(rem)

    # initialize global scores (will overwrite for remaining each round)
    scores = np.full(N, eps, dtype=np.float64)
    results = {}
    for i, bsz in enumerate(batches, start=1):
        # --- selection step (exploitation + exploration) ---
        k_exploit = int(np.ceil(2.0 * bsz / 3))
        # exploitation (uncertainty/diversity with gamma>0)
        new_ids = sampler.sample_ids_mc(
            data.values, scores, k=k_exploit, alpha=alpha, gamma=gamma,
            cond_ids=train_ids, steps=steps
        )
        train_ids = np.append(train_ids, new_ids).astype(int)

        # exploration (pure diversity; gamma=0)
        new_ids = sampler.sample_ids_mc(
            data.values, scores, k=bsz - k_exploit, alpha=alpha, gamma=0.0,
            cond_ids=train_ids, steps=steps
        )
        train_ids = np.append(train_ids, new_ids).astype(int)

        # --- compute uncertainty on remaining pool via an ensemble ---
        remaining_ids = np.setdiff1d(np.arange(N), train_ids, assume_unique=False)
        if remaining_ids.size == 0:
            break

        # accumulate probabilities on remaining only
        probs_accum = None
        for j in range(num_models):
            m, _ = utils.train_and_test(
                data.iloc[train_ids], labels[train_ids],
                data.iloc[remaining_ids], labels[remaining_ids], 
                model, logger, print_results=False
            )
            p = m.predict_proba(data.iloc[remaining_ids])
            if probs_accum is None:
                probs_accum = np.zeros_like(p, dtype=np.float64)
            probs_accum += p

        # average ensemble, compute entropy on remaining, update scores
        out = probs_accum / num_models
        out = np.clip(out, eps, 1.0)
        ent = -(out * np.log(out)).sum(axis=1)

        scores[:] = eps               # tiny default everywhere
        scores[remaining_ids] = ent + eps  # higher = more uncertain
        model, test_dict = utils.train_and_test(data.iloc[train_ids], labels[train_ids], test_data, test_labels, model, logger)
        results[len(train_ids)] = test_dict
        logger.info(f"\033[92mActive DPP\033[0m with {len(train_ids)} samples : Accuracy of the model on the test images: {round(test_dict['test_accuracy'], 4)}")
    return results

def qbc_dpp(data, labels, test_data, test_labels, model, logger, batch_size=10, sub_sample_count=200, steps=200, gamma=5, alpha=4, num_models=10, eps=1e-8, rng_seed=None):
    """
    Progressive Query by Committee (nested):
      - Each round selects a full batch via DPP using current uncertainty 'scores' (gamma>0 encourages diversity).
      - Uncertainty is estimated by an ensemble trained on the current labeled set; entropy on remaining pool only.
      - Returns a dict: {k: test_dict_at_k} for plotting sample-efficiency curves.
    """
    logger.info(f"\033[93mQuery by Committee with DPP\033[0m")

    if rng_seed is not None:
        np.random.seed(rng_seed)

    N = labels.shape[0]
    max_k = min(sub_sample_count, N)
    if max_k < sub_sample_count:
        logger.info(f"[warn] sub_sample_count={sub_sample_count} > N={N}. Using {max_k}.")

    # Build batch sizes: batch_size, ..., remainder
    batches = [batch_size] * (max_k // batch_size)
    rem = max_k % batch_size
    if rem:
        batches.append(rem)

    train_ids = np.array([], dtype=int)
    scores = np.full(N, eps, dtype=np.float64)  # overwritten on remaining each round
    results = {}

    for i, bsz in enumerate(batches, start=1):
        # --- selection: pick full batch using current scores (uncertainty) + DPP diversity ---
        new_ids = sampler.sample_ids_mc(
            data.values, scores, k=bsz, alpha=alpha, gamma=gamma,
            cond_ids=train_ids, steps=steps
        )
        train_ids = np.append(train_ids, new_ids).astype(int)
        train_ids = np.unique(train_ids)  # ensure uniqueness

        # --- compute uncertainty on remaining pool via committee/ensemble ---
        picked = set(train_ids.tolist())
        remaining_ids = np.array([idx for idx in range(N) if idx not in picked], dtype=int)
        if remaining_ids.size == 0:
            # evaluate once at this final k and exit
            model, test_dict = utils.train_and_test(
                data.iloc[train_ids], labels[train_ids], test_data, test_labels, model, logger
            )
            results[len(train_ids)] = test_dict
            logger.info(f"\033[93mQBC-DPP\033[0m with {len(train_ids)} samples : "
                  f"Accuracy: {round(test_dict['test_accuracy'], 4)}")
            break

        probs_accum = None
        for j in range(num_models):
            m, _ = utils.train_and_test(
                data.iloc[train_ids], labels[train_ids],
                data.iloc[remaining_ids], labels[remaining_ids], model, logger,
                print_results=False
            )
            p = m.predict_proba(data.iloc[remaining_ids])
            if probs_accum is None:
                probs_accum = np.zeros_like(p, dtype=np.float64)
            probs_accum += p

        # Average committee predictions, compute entropy on remaining only
        out = probs_accum / num_models
        out = np.clip(out, eps, 1.0)
        ent = -(out * np.log(out)).sum(axis=1)

        # Update scores: high entropy => more uncertain (more likely to be selected next)
        scores[:] = eps
        scores[remaining_ids] = ent + eps

        # --- evaluate on held-out test set at current training size ---
        model, test_dict = utils.train_and_test(
            data.iloc[train_ids], labels[train_ids], test_data, test_labels, model, logger
        )
        results[len(train_ids)] = test_dict
        
        logger.info(f"\033[92mQBC-DPP\033[0m with {len(train_ids)} samples : "
              f"Accuracy: {round(test_dict['test_accuracy'], 4)}")

    # Return the full learning curve (k -> metrics). Final point is results[max_k] if reached.
    return results

def qbc_pure(data, labels, test_data, test_labels, model, logger, batch_size=10, sub_sample_count=200, num_models=10, eps=1e-8, warm_start=None, rng_seed=None):
    """
    Pure Query-by-Committee (no DPP, no diversity kernel).

    Flow per round:
      1) Train a committee (ensemble of `num_models`) on the current labeled set.
      2) Predict class probabilities on the remaining pool.
      3) Compute committee disagreement via predictive entropy.
      4) Select the top-`batch_size` highest-entropy points (pure QBC).
      5) Add them to the labeled set (nested), evaluate on held-out test set.

    Returns:
      results: dict mapping k (#labeled) -> test_dict (your metrics)
    """
    logger.info(f"\033[93mQuery by Committee\033[0m")
    
    if rng_seed is not None:
        np.random.seed(rng_seed)

    N = labels.shape[0]
    max_k = min(sub_sample_count, N)
    if max_k < sub_sample_count:
        logger.info(f"[warn] sub_sample_count={sub_sample_count} > N={N}. Using {max_k}.")

    # Warm start size: to train the first committee before scoring uncertainty
    if warm_start is None:
        warm_start = batch_size
    warm_start = min(warm_start, max_k)

    # Build batch sizes: warm_start once, then repeat batch_size until max_k
    remaining_quota = max_k
    batches = []
    if warm_start > 0:
        batches.append(warm_start)
        remaining_quota -= warm_start
    while remaining_quota > 0:
        take = min(batch_size, remaining_quota)
        batches.append(take)
        remaining_quota -= take

    results = {}
    train_ids = np.array([], dtype=int)

    # --- Round 0: warm start (uniform random) ---
    if batches:
        k0 = batches[0]
        # sample without replacement
        init_ids = np.random.permutation(N)[:k0]
        train_ids = np.array(init_ids, dtype=int)

        # Evaluate after warm start
        _, test_dict = utils.train_and_test(
            data.iloc[train_ids], labels[train_ids], test_data, test_labels, model, logger
        )
        results[len(train_ids)] = test_dict
        logger.info(
            f"\033[92mQBC\033[0m warm-start with {len(train_ids)} samples : "
            f"Accuracy: {round(test_dict['test_accuracy'], 4)}"
        )

    # --- Progressive QBC rounds (pure top-k by committee disagreement) ---
    # remaining batches after warm start
    for i, bsz in enumerate(batches[1:], start=1):
        # Build remaining pool
        picked_mask = np.zeros(N, dtype=bool)
        picked_mask[train_ids] = True
        remaining_ids = np.nonzero(~picked_mask)[0]
        if remaining_ids.size == 0:
            break

        # Committee predictions on remaining pool
        probs_accum = None
        for _ in range(num_models):
            # Train a model on current labeled set
            m, _ = utils.train_and_test(
                data.iloc[train_ids], labels[train_ids],
                data.iloc[remaining_ids], labels[remaining_ids], model, logger,
                print_results=False
            )
            # Predict on remaining only
            p = m.predict_proba(data.iloc[remaining_ids])
            if probs_accum is None:
                probs_accum = np.zeros_like(p, dtype=np.float64)
            probs_accum += p

        # Average committee, compute entropy on remaining
        out = probs_accum / num_models
        out = np.clip(out, eps, 1.0)              # numerical stability
        ent = -(out * np.log(out)).sum(axis=1)    # predictive entropy

        # Select top-bsz by entropy (pure QBC selection)
        if bsz >= ent.size:
            chosen_rel = np.arange(ent.size)      # take all if fewer remain than bsz
        else:
            # argpartition is O(n) for top-k
            chosen_rel = np.argpartition(ent, -bsz)[-bsz:]
            # (optional) sort chosen by entropy descending for determinism in logs
            chosen_rel = chosen_rel[np.argsort(ent[chosen_rel])[::-1]]

        new_ids = remaining_ids[chosen_rel]

        # Append while preserving order and removing dupes
        if train_ids.size == 0:
            train_ids = new_ids
        else:
            seen = set(train_ids.tolist())
            train_ids = np.concatenate(
                [train_ids, new_ids[[i not in seen and not seen.add(i) for i in new_ids]]]
            )

        # Evaluate at current size on held-out test set
        _, test_dict = utils.train_and_test(
            data.iloc[train_ids], labels[train_ids], test_data, test_labels, model, logger
        )
        results[len(train_ids)] = test_dict
        logger.info(
            f"\033[92mQBC\033[0m with {len(train_ids)} samples : "
            f"Accuracy: {round(test_dict['test_accuracy'], 4)}"
        )

        # Early exit if we’ve reached max_k
        if len(train_ids) >= max_k:
            break

    return results

def curriculum_guided_active_dpp(data, labels, test_data, test_labels, model, logger, batch_size=10, sub_sample_count=200, steps=200,
    gamma=5,
    alpha=4,
    num_models=10,
    eps=1e-8,
    beta=1.0,
    start_q_low=0.20,
    band_w=0.30,
    end_q_low=0.60,
    prefilter_mult=5,
    rng_seed=None,
    show_tqdm=False  # <- set True if you want to see per-model bars
):
    logger.info(f"\033[93mCurriculum Guided Active DPP\033[0m")
    TAG = "\033[92mCGAL-DPP\033[0m"  # match your Active DPP print style

    if rng_seed is not None:
        np.random.seed(rng_seed)

    N = labels.shape[0]
    max_k = min(sub_sample_count, N)
    if max_k < sub_sample_count:
        logger.info(f"[warn] sub_sample_count={sub_sample_count} > N={N}. Using {max_k}.")

    # batch plan
    batches = [batch_size] * (max_k // batch_size)
    rem = max_k % batch_size
    if rem:
        batches.append(rem)

    # helpers
    def entropy(p, axis=1):
        p = np.clip(p, eps, 1.0)
        return -(p * np.log(p)).sum(axis=axis)

    def gaussian_bump(x, mu, sigma):
        z = (x - mu) / (sigma + eps)
        return np.exp(-0.5 * z * z) + eps

    def schedule_q_low(t, T):
        return start_q_low + (end_q_low - start_q_low) * (t / max(T - 1, 1))

    train_ids = np.array([], dtype=int)
    results = {}

    # ---- warm start: passive DPP for diversity ----
    warm_k = min(batch_size, max_k)
    init_ids = sampler.sample_ids_mc(
        data.values, np.ones(N), k=warm_k, alpha=alpha, gamma=0.0,
        cond_ids=[], steps=steps
    )
    train_ids = np.asarray(init_ids, dtype=int)

    # eval (Active DPP format)
    _, test_dict = utils.train_and_test(
        data.iloc[train_ids], labels[train_ids], test_data, test_labels, model, logger
    )
    logger.info(f"{TAG} with {len(train_ids)} samples : "
          f"Accuracy of the model on the test images: {test_dict['test_accuracy']:.4f}")
    results[len(train_ids)] = test_dict

    # ---- progressive rounds ----
    T = len(batches) - 1  # we already used one batch for warm start
    for t, bsz in enumerate(batches[1:], start=1):
        # remaining pool
        picked_mask = np.zeros(N, dtype=bool)
        picked_mask[train_ids] = True
        remaining_ids = np.nonzero(~picked_mask)[0]
        if remaining_ids.size == 0:
            break

        # committee predictions on remaining pool
        probs_accum = None
        iterator = range(num_models)
        if show_tqdm:
            from tqdm import trange
            iterator = trange(num_models, desc=f"Round {t} of {T}", unit="model", leave=False)
        for _ in iterator:
            m, _ = utils.train_and_test(
                data.iloc[train_ids], labels[train_ids],
                data.iloc[remaining_ids], labels[remaining_ids], model, logger,
                print_results=False
            )
            p = m.predict_proba(data.iloc[remaining_ids])
            if probs_accum is None:
                probs_accum = np.zeros_like(p, dtype=np.float64)
            probs_accum += p

        P = np.clip(probs_accum / num_models, eps, 1.0)
        maxp = P.max(axis=1)
        d = 1.0 - maxp          # difficulty
        u = entropy(P)          # uncertainty

        # curriculum band
        q_low = schedule_q_low(t - 1, T if T > 0 else 1)
        q_high = min(q_low + band_w, 0.99)
        lo, hi = np.quantile(d, [q_low, q_high]) if remaining_ids.size > 1 else (d.min(), d.max())
        band_mask = (d >= lo) & (d <= hi)
        candidate_rel = np.nonzero(band_mask)[0]

        # widen if needed
        if candidate_rel.size < bsz:
            widen = 0.05
            lq, hq = q_low, q_high
            while candidate_rel.size < bsz and (lq > 0.0 or hq < 1.0):
                lq = max(0.0, lq - widen)
                hq = min(1.0, hq + widen)
                lo, hi = np.quantile(d, [lq, hq])
                band_mask = (d >= lo) & (d <= hi)
                candidate_rel = np.nonzero(band_mask)[0]
        if candidate_rel.size == 0:
            candidate_rel = np.arange(remaining_ids.size)

        candidates = remaining_ids[candidate_rel]

        # composite quality q(x) = u(x)^beta * curriculum_bump(d)
        mu = (lo + hi) / 2.0
        sigma = max((hi - lo) / 2.0, 1e-3)
        w_curr = gaussian_bump(d[candidate_rel], mu=mu, sigma=sigma)
        q = np.clip((u[candidate_rel] ** beta) * w_curr, eps, None)

        # optional prefilter
        if prefilter_mult and candidate_rel.size > bsz:
            M = min(prefilter_mult * bsz, candidate_rel.size)
            topM_rel = np.argpartition(q, -M)[-M:]
            topM_rel = topM_rel[np.argsort(q[topM_rel])[::-1]]
            pool_rel = topM_rel
        else:
            pool_rel = np.arange(candidate_rel.size)

        pool_ids = candidates[pool_rel]
        pool_X = data.values[pool_ids]
        pool_scores = q[pool_rel]

        # DPP selection over the pool (**cond_ids must be empty, pool is subset**)
        new_rel = sampler.sample_ids_mc(
            pool_X, pool_scores, k=bsz, alpha=alpha, gamma=gamma,
            cond_ids=[], steps=steps
        )
        new_ids = pool_ids[np.asarray(new_rel, dtype=int)]

        # append uniquely
        if train_ids.size == 0:
            train_ids = np.asarray(new_ids, dtype=int)
        else:
            seen = set(train_ids.tolist())
            to_add = [i for i in new_ids if (i not in seen and not seen.add(i))]
            if to_add:
                train_ids = np.concatenate([train_ids, np.asarray(to_add, dtype=int)])

        # evaluate (Active DPP print style)
        _, test_dict = utils.train_and_test(
            data.iloc[train_ids], labels[train_ids], test_data, test_labels, model, logger
        )
        logger.info(f"{TAG} with {len(train_ids)} samples : "
              f"Accuracy of the model on the test images: {test_dict['test_accuracy']:.4f}")
        results[len(train_ids)] = test_dict

        if len(train_ids) >= max_k:
            break

    return results
