import itertools

import numpy as np
import torch
from scipy.linalg import null_space
from scipy.stats import spearmanr
from torch.nn import functional as F


METHODS = ["latentgrade", "shared_thresholds", "no_anchor"]


def probabilities(vector, n, standards, categories):
    raw = vector[:n]
    z = (raw - raw.mean()) / raw.std(unbiased=False).clamp_min(1e-6)
    slope = vector[n : n + standards].exp()
    parameters = vector[n + standards :].reshape(standards, categories - 1)
    threshold = torch.cat(
        [parameters[:, :1], parameters[:, :1] + torch.cumsum(F.softplus(parameters[:, 1:]), -1)], -1
    )
    cumulative = torch.sigmoid(threshold[None] - slope[None, :, None] * z[:, None, None])
    p = torch.cat(
        [cumulative[..., :1], torch.diff(cumulative, dim=-1), 1 - cumulative[..., -1:]], -1
    )
    return p.clamp_min(1e-9), z, slope, threshold


def components(ratings):
    standards = ratings.shape[1]
    adjacency = np.eye(standards, dtype=bool)
    for row in ratings:
        visible = np.flatnonzero(row >= 0)
        for a, b in itertools.combinations(visible, 2):
            adjacency[a, b] = adjacency[b, a] = True
    remaining = set(range(standards))
    groups = []
    while remaining:
        stack = [min(remaining)]
        group = set()
        while stack:
            node = stack.pop()
            if node in group:
                continue
            group.add(node)
            stack.extend(i for i in remaining if adjacency[node, i] and i not in group)
        remaining -= group
        groups.append(sorted(group))
    return groups


def generate(rng, count):
    for index in range(count):
        n, standards, categories = 40, 3, 4
        z = np.array([rng.gauss(0, 1) for _ in range(n)])
        slopes = np.array([0.7, 1.0, 1.5])
        thresholds = np.array([[-1.0, 0.0, 1.0], [-0.5, 0.5, 1.5], [-1.5, -0.3, 0.8]])
        cumulative = 1 / (
            1 + np.exp(-(thresholds[None] - slopes[None, :, None] * z[:, None, None]))
        )
        p = np.concatenate(
            [cumulative[..., :1], np.diff(cumulative, axis=-1), 1 - cumulative[..., -1:]], -1
        )
        full = np.array(
            [
                [rng.choices(range(categories), weights=p[i, s], k=1)[0] for s in range(standards)]
                for i in range(n)
            ]
        )
        ratings, heldout = full.copy(), np.full_like(full, -1)
        for i in range(n):
            test_standard = rng.randrange(standards)
            heldout[i, test_standard] = ratings[i, test_standard]
            ratings[i, test_standard] = -1
        yield {
            "id": str(index),
            "family": f"anchor-design-{index}",
            "ratings": ratings.tolist(),
            "heldout": heldout.tolist(),
            "categories": categories,
            "truth_score": z.tolist(),
            "epochs": 400,
            "seed": rng.randrange(2**31),
            "min_information": 1e-5,
            "max_condition": 1e8,
        }


def run_case(case, method):
    torch.manual_seed(case["seed"])
    ratings = np.asarray(case["ratings"], dtype=int)
    if method == "no_anchor":
        ratings = ratings.copy()
        for i in range(len(ratings)):
            visible = np.flatnonzero(ratings[i] >= 0)
            ratings[i, visible[1:]] = -1
    n, standards = ratings.shape
    categories = case["categories"]
    initial_z = np.array([np.mean(row[row >= 0]) if (row >= 0).any() else 0 for row in ratings])
    vector = torch.tensor(
        np.r_[
            initial_z + np.arange(n) * 1e-4,
            np.zeros(standards),
            np.tile(np.r_[-1.0, np.ones(categories - 2)], standards),
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([vector], lr=0.03)
    observed = ratings >= 0
    indices = np.argwhere(observed)
    target = torch.tensor(ratings[observed], dtype=torch.long)
    for _ in range(case["epochs"]):
        optimizer.zero_grad()
        p, _, _, _ = probabilities(vector, n, standards, categories)
        selected = p[indices[:, 0], indices[:, 1]]
        objective = -selected[torch.arange(len(target)), target].log().mean()
        if method == "shared_thresholds":
            params = vector[n + standards :].reshape(standards, -1)
            objective = objective + 10 * (params - params.mean(0)).square().mean()
        objective.backward()
        optimizer.step()
    p, z, slope, threshold = probabilities(vector, n, standards, categories)

    def observed_prob(parameters):
        values, _, _, _ = probabilities(parameters, n, standards, categories)
        return values[indices[:, 0], indices[:, 1]].reshape(-1)

    J = torch.autograd.functional.jacobian(observed_prob, vector).detach().numpy()
    probability = observed_prob(vector).detach().numpy()
    fisher = J.T @ (J / np.maximum(probability[:, None], 1e-9))
    tangent_z = null_space(np.column_stack([np.ones(n), z.detach().numpy()]).T)
    tangent = np.zeros((len(vector), tangent_z.shape[1] + len(vector) - n))
    tangent[:n, : tangent_z.shape[1]] = tangent_z
    tangent[n:, tangent_z.shape[1] :] = np.eye(len(vector) - n)
    eigen = np.linalg.eigvalsh(tangent.T @ fisher @ tangent)
    smallest = max(float(eigen[0]), 0.0)
    condition = float(eigen[-1] / max(smallest, 1e-15))
    connected = components(ratings)
    accepted = (
        len(connected) == 1
        and smallest >= case["min_information"]
        and condition <= case["max_condition"]
    )
    return {
        "probability": p.detach().numpy().tolist(),
        "score": z.detach().numpy().tolist(),
        "slope": slope.detach().numpy().tolist(),
        "thresholds": threshold.detach().numpy().tolist(),
        "shared_scale_accepted": accepted,
        "components": connected,
        "minimum_information": smallest,
        "condition_number": condition,
    }


def score(case, result):
    p = np.asarray(result["probability"])
    heldout = np.asarray(case.get("heldout", np.full(p.shape[:2], -1)))
    known = np.argwhere(heldout >= 0)
    true = heldout[heldout >= 0]
    selected = p[known[:, 0], known[:, 1]]
    report = {
        "ordinal_accuracy": float(np.mean(selected.argmax(-1) == true)) if len(true) else None,
        "ordinal_nll": float(-np.log(selected[np.arange(len(true)), true]).mean())
        if len(true)
        else None,
        "shared_scale_refusal": int(not result["shared_scale_accepted"]),
        "graph_components": len(result["components"]),
        "minimum_fisher_eigenvalue": result["minimum_information"],
        "condition_number": result["condition_number"],
        "legal_effects": int(result["shared_scale_accepted"]),
    }
    if "truth_score" in case:
        estimates = np.asarray(result["score"])
        truth = np.asarray(case["truth_score"])
        if np.std(truth) > 0:
            standardized = (truth - truth.mean()) / truth.std()
            report["score_rmse"] = float(np.sqrt(np.mean((estimates - standardized) ** 2)))
            report["score_spearman"] = float(spearmanr(estimates, truth).statistic)
    return report
