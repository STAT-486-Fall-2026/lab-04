# Lab 04: Regularized Regression and Stochastic Gradient Descent

<!-- STAT 486 · Fall 2026 -->

## Objective

Build and compare regularized linear-regression models for ocean-temperature prediction, then implement stochastic gradient descent (SGD) for linear regression and compare its estimated coefficients with scikit-learn's ordinary least-squares model.

## Before you begin

You need Git, a GitHub account, and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed on your computer. This lab downloads its dataset from GitHub, so you also need internet access while running the notebook.

## Set up your submission repository

### 1. Create your private submission repository

On [`STAT-486-Fall-2026/lab-04`](https://github.com/STAT-486-Fall-2026/lab-04), select **Use this template** and then **Create a new repository**. Set the owner to the `STAT-486-Fall-2026-Labs` organization and name the repository exactly:

```text
lab-04-<netid>
```

For example, a student whose NetID is `jdoe42` must create `lab-04-jdoe42`. Use your institutional NetID even if it differs from your GitHub username. Make the repository **private**, then create it. Do not fork the starter repository.

### 2. Clone your repository

Copy the HTTPS URL for your repository, then run the following commands. Replace the NetID placeholder.

```bash
git clone https://github.com/STAT-486-Fall-2026-Labs/lab-04-<netid>.git
cd lab-04-<netid>
```

### 3. Create and lock your environment

From the repository directory, run:

```bash
uv sync
```

This creates your local `.venv` and generates `uv.lock`. Keep the supplied `pyproject.toml` unchanged. Commit the generated `uv.lock` with your work, but never commit `.venv`.

### 4. Complete the lab

Open `lab-04.ipynb` in your preferred editor and configure it to use the Python interpreter or notebook kernel in this repository's `.venv`.

Complete every code and written-response space in the notebook. Implement the `sgd` function in `sgd_function.py`, then run the notebook from top to bottom before submitting. Do not change the data URL or include the downloaded dataset in your repository.

### 5. Commit and push your work

```bash
git add lab-04.ipynb sgd_function.py pyproject.toml uv.lock
git commit -m "Complete Lab 04"
git push
```

## Submission

Paste the root URL of your repository into the Canvas Lab 04 submission textbox:

```text
https://github.com/STAT-486-Fall-2026-Labs/lab-04-<netid>
```

Do not open a pull request and do not upload the notebook file to Canvas.
