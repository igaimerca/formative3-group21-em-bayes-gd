# Formative 3 - Probability Distributions, Bayesian Probability, and Gradient Descent

Formative 3 Group 21 submission - Aime Igirimpuhwe ([@igaimerca](https://github.com/igaimerca))

## Submission deliverables

| Required | Link |
|---|---|
| Jupyter Notebook with all implementations (EM, Bayes, gradient descent - each with plots) | [Part 1](https://github.com/igaimerca/formative3-group21-em-bayes-gd/blob/main/notebooks/part1_em_heights.ipynb) &middot; [Part 2](https://github.com/igaimerca/formative3-group21-em-bayes-gd/blob/main/notebooks/part2_bayes_sentiment.ipynb) &middot; [Part 4](https://github.com/igaimerca/formative3-group21-em-bayes-gd/blob/main/notebooks/part4_gradient_descent.ipynb) |
| A neat PDF showing handwritten manual calculations of Part 3 | [part3_manual_gradient_descent.pdf](https://github.com/igaimerca/formative3-group21-em-bayes-gd/blob/main/docs/part3_manual_gradient_descent.pdf) |
| A PDF showing contributions | [Task allocation / contribution sheet](<https://github.com/igaimerca/formative3-group21-em-bayes-gd/blob/main/docs/BSE Group Assignments _ Task Sheet_Formative 3 _Cohort 2 Group 21 - 1.pdf>) |

## Quick links

- [Part 3 - handwritten manual gradient descent derivation (PDF)](https://github.com/igaimerca/formative3-group21-em-bayes-gd/blob/main/docs/part3_manual_gradient_descent.pdf)
- [Part 1 - EM/GMM notebook](https://github.com/igaimerca/formative3-group21-em-bayes-gd/blob/main/notebooks/part1_em_heights.ipynb)
- [Part 2 - Bayesian sentiment notebook](https://github.com/igaimerca/formative3-group21-em-bayes-gd/blob/main/notebooks/part2_bayes_sentiment.ipynb)
- [Part 4 - Gradient descent notebook](https://github.com/igaimerca/formative3-group21-em-bayes-gd/blob/main/notebooks/part4_gradient_descent.ipynb)
- [Task allocation / contribution sheet (PDF)](<https://github.com/igaimerca/formative3-group21-em-bayes-gd/blob/main/docs/BSE Group Assignments _ Task Sheet_Formative 3 _Cohort 2 Group 21 - 1.pdf>)

## What's in here

| Part | Deliverable | Where |
|---|---|---|
| 1 - EM for a Gaussian Mixture | Notebook: from-scratch EM on height data, tracking table, live classification demo | [`notebooks/part1_em_heights.ipynb`](notebooks/part1_em_heights.ipynb) |
| 2 - Bayesian Probability | Notebook: keyword-based sentiment via hand-rolled Bayes' Theorem | [`notebooks/part2_bayes_sentiment.ipynb`](notebooks/part2_bayes_sentiment.ipynb) |
| 3 - Manual Gradient Descent | Full worked derivation (chain rule, no skipped steps) | [`docs/part3_manual_gradient_descent.pdf`](docs/part3_manual_gradient_descent.pdf) |
| 4 - Gradient Descent in Code | Notebook: same problem in Python, SciPy-checked gradient, convergence plots | [`notebooks/part4_gradient_descent.ipynb`](notebooks/part4_gradient_descent.ipynb) |

Shared logic lives in `src/` and is imported by the notebooks rather than duplicated inside them:

- `src/em_gmm.py` - E-step / M-step / log-likelihood / posterior classification, built on raw numpy arithmetic (no `sklearn.mixture`)
- `src/bayes_sentiment.py` - word-count based Bayes' Theorem breakdown, plain Python/CSV (no ML libraries)
- `src/gradient_descent.py` - batch gradient descent with both a hand-derived analytic gradient and a SciPy (`approx_fprime`) numerical gradient, used to cross-check each other

## Datasets

- `data/GaltonFamilies.csv` - Galton's parent/child height records (Part 1). Used as: every child's height (`childHeight`) as the "Children" population, and one height per family from the `father` column as the "Pros" (taller adult reference) population - there's no basketball-player dataset available, so father heights are the closest already-available stand-in for a taller population.
- `data/imdb_reviews_sample.csv` - a balanced 6,000-review sample (2,992 positive / 3,008 negative) drawn from the 50k-review IMDb dataset (Part 2). Sampled down from the full 66 MB file since keyword frequency statistics don't need all 50k rows and a smaller file keeps the repo light.

## Part 1 highlights

- Argues why a hard split at the global mean is worse than EM's soft, iterative assignment (see the notebook's markdown cells for the full reasoning).
- Tracking table for iterations 0 (init), 1, and 2: μ1, μ2, σ1², σ2², π1, π2, log-likelihood.
- Live demo classifying new heights with explicit P(Child) vs. P(Pro) posteriors.
- A validation check against the true `gender` labels (which EM never sees) shows the discovered clusters track sex more than age - documented honestly in the notebook rather than glossed over, since many "children" in this dataset are already adult-height offspring.

## Part 2 highlights

- Keywords: **excellent, wonderful, superb** (positive) and **terrible, awful, waste** (negative).
- Only P(Positive | keyword) is computed, per the assignment's "pick one direction" instruction.
- Prior / Likelihood / Marginal / Posterior shown per keyword in a table, all derived from raw document counts.

## Part 3 / 4 - Gradient Descent

Model: ŷ = m1·x1 + m2·x2 + b, fit with X = [(1,3), (4,10)], y = (5, 6), m_init = [-1, 2], b_init = 1, learning rate = 0.01.

- Part 3 shows **one full manual update** (group size = 1, per the assignment's "iterations = number of group members" rule), with every chain-rule step written out - no skipped arithmetic.
- Part 4 reruns the same setup in code for 150 iterations, cross-checks the hand-derived gradient against SciPy's numerical gradient (they agree to 5 decimal places), and plots (a) m1/m2/b over iterations and (b) cost over iterations.

## Running it locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib pandas jupyter
jupyter notebook notebooks/
```
