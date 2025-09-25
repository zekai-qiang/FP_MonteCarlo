# A Monte Carlo Simulator for UKFPO Outcomes

## 1. Introduction

This is a Monte Carlo-based method designed to model the probabilistic outcomes of the UKFPO allocation process. The algorithm attempts to simulate the interaction between applicant preferences and capacity constraints and estimate the likelihood of allocation to specific foundation schools given input preference list.

Dataset FP_ratios.xlsx was parsed from first preference ratios published at https://foundationprogramme.nhs.uk/wp-content/uploads/sites/2/2025/07/UKFP-First-preference-ratios-2023-–-2025.pdf

Download or update it and add to your excel_file_path in run_simulator.py.

## 2. Mathematical Framework

### 2.1 Problem Definition

Let **S** = {s₁, s₂, ..., sₙ} represent the set of n foundation schools, and **A** = {a₁, a₂, ..., aₘ} represent the set of m applicants. Each school sᵢ has a capacity cᵢ representing the number of available training positions. Each applicant aⱼ submits a preference list Pⱼ = (pⱼ₁, pⱼ₂, ..., pⱼₖ) where pⱼₗ ∈ S represents the ℓ-th most preferred school for applicant aⱼ.

The allocation function φ: A → S ∪ {∅} maps each applicant to either a foundation school or to the null set (representing non-allocation), subject to the capacity constraint:

|{aⱼ ∈ A : φ(aⱼ) = sᵢ}| ≤ cᵢ ∀ sᵢ ∈ S

### 2.2 Preference Generation Model

The algorithm generates synthetic preference lists for all applicants based on observed first-choice preference distributions and school similarity metrics. For each applicant aⱼ, the first choice is sampled from a discrete distribution:

P(pⱼ₁ = sᵢ) = fᵢ / Σₖ fₖ

where fᵢ represents the observed number of first-choice preferences for school sᵢ.

### 2.3 School Similarity Function

The similarity between schools is modeled using their first-preference ratios after pre-allocation. For schools sᵢ and sⱼ, the similarity weight is defined as:

w(sᵢ, sⱼ) = 1 / (1 + |rᵢ - rⱼ|)

where rᵢ represents the first-preference ratio for school sᵢ, calculated as:

rᵢ = fᵢ / cᵢ

This similarity function assumes that schools with similar competitiveness (ratio of first choices to capacity) are more likely to appear together in preference lists. Change this if you would like to.

### 2.4 Preference List Generation Algorithm

For each applicant aⱼ with first choice pⱼ₁ = sₖ, subsequent preferences are generated using a weighted sampling approach that combines school similarity with random variation:

For position ℓ > 1, the weight for school sᵢ is calculated as:

Wᵢ,ⱼ,ℓ = w(sₖ, sᵢ) × α + Rᵢ,ⱼ,ℓ × (1 - α)

where:
- w(sₖ, sᵢ) is the similarity weight between the first choice sₖ and school sᵢ
- α ∈ [0,1] is the preference correlation parameter controlling the strength of similarity-based preferences
- Rᵢ,ⱼ,ℓ ~ U(0.5, 1.5) is a random factor drawn from a uniform distribution

The probability of selecting school sᵢ for position ℓ is then:

P(pⱼℓ = sᵢ | pⱼ₁ = sₖ, pⱼ₁, ..., pⱼ₍ℓ₋₁₎) = Wᵢ,ⱼ,ℓ / Σᵤ∉{pⱼ₁,...,pⱼ₍ℓ₋₁₎} Wᵤ,ⱼ,ℓ

where the summation excludes schools already selected in previous positions.

## 3. Allocation Mechanism

### 3.1 Two-Phase Allocation Process

**Phase 1: First Choice Allocation**
All applicants are processed in random order for their first-choice preferences only. For each applicant aⱼ with first choice pⱼ₁ = sᵢ:
- If cᵢ > 0, allocate aⱼ to sᵢ and set cᵢ ← cᵢ - 1
- Otherwise, defer aⱼ to Phase 2

**Phase 2: Remaining Preference Allocation**
Unallocated applicants are processed again in the same random order. For each unallocated applicant aⱼ:
- Iterate through preference list Pⱼ = (pⱼ₁, pⱼ₂, ..., pⱼₖ)
- For the first school sᵢ ∈ Pⱼ with cᵢ > 0, allocate aⱼ to sᵢ and set cᵢ ← cᵢ - 1
- If no school in Pⱼ has available capacity, aⱼ remains unallocated

### 3.2 Randomization Strategy

The random ordering of applicants π: {1, 2, ..., m} → {1, 2, ..., m} is generated using a uniform random permutation for each simulation iteration. This ensures that:

P(π(aⱼ) = k) = 1/m ∀ aⱼ ∈ A, ∀ k ∈ {1, 2, ..., m}

## 4. Statistical Estimation Framework

### 4.1 Monte Carlo Estimation

The algorithm estimates allocation probabilities using Monte Carlo simulation with N independent iterations. For a target applicant with preference list P* = (p₁*, p₂*, ..., pₖ*), let Xᵢ⁽ⁿ⁾ be the indicator variable for iteration n:

Xᵢ⁽ⁿ⁾ = {1 if target applicant is allocated to school sᵢ in iteration n
        {0 otherwise

The estimated probability of allocation to school sᵢ is:

P̂ᵢ = (1/N) Σₙ₌₁ᴺ Xᵢ⁽ⁿ⁾

### 4.2 Confidence Intervals

The standard error for each probability estimate follows from the binomial distribution:

SE(P̂ᵢ) = √[P̂ᵢ(1 - P̂ᵢ)/N]

The 95% confidence interval for the true allocation probability is:

P̂ᵢ ± 1.96 × SE(P̂ᵢ)

### 4.3 Position-Based Analysis

The algorithm also estimates the probability of receiving an allocation at each preference position. Let Yₗ⁽ⁿ⁾ be the indicator variable for receiving the ℓ-th choice in iteration n:

Yₗ⁽ⁿ⁾ = {1 if target applicant is allocated to pₗ* in iteration n
        {0 otherwise

The estimated probability of receiving the ℓ-th choice is:

Q̂ₗ = (1/N) Σₙ₌₁ᴺ Yₗ⁽ⁿ⁾

## 5. Model Validation and Assumptions

### 5.1 Important Assumptions

1. **Applicant preferences are independent**: The model assumes that applicant preferences are conditionally independent given the first choice and school similarity structure.

2. **Preference correlations are similarity-based**: The assumption that schools with similar first-preference ratios are more likely to appear together in preference lists reflects the hypothesis that applicants consider school competitiveness when ranking alternatives.

3. **Processing order is sufficiently random**: The model assumes that the actual allocation system introduces sufficient randomness that can be approximated by random serial dictatorship.

4. **School characteristics are stationary**: The model assumes that school characteristics (capacity, competitiveness) remain stable within the allocation cycle being simulated.

### 5.2 Sensitivity Analysis

The preference correlation parameter α allows for sensitivity analysis of model assumptions. Values of α approaching 1 create highly correlated preferences based on school similarity, while values approaching 0 generate more random preference structures.

## 6. Computational Implementation

### 6.1 Algorithmic Complexity

The computational complexity of a single simulation iteration is O(m × k × log n), where m is the number of applicants, k is the average preference list length, and n is the number of schools. The logarithmic factor arises from the weighted sampling procedure for preference generation.

### 6.2 Convergence Criteria

The Monte Carlo estimation converges to the true allocation probabilities as N → ∞ by the Strong Law of Large Numbers. In practice, convergence is assessed by monitoring the stability of probability estimates across simulation batches and ensuring that confidence interval widths are acceptably narrow for decision-making purposes.

## 7. Forecasting Extensions

### 7.1 Temporal Trend Modeling

For forecasting future allocation years, the model incorporates linear extrapolation of first-preference ratios:

r̂ᵢ⁽ᵗ⁺¹⁾ = rᵢ⁽ᵗ⁾ + (rᵢ⁽ᵗ⁾ - rᵢ⁽ᵗ⁻¹⁾)

where rᵢ⁽ᵗ⁾ represents the first-preference ratio for school sᵢ in year t.

### 7.2 Demand Adjustment

The forecasted number of first-choice applications is calculated as:

f̂ᵢ⁽ᵗ⁺¹⁾ = max(50, min(5000, ⌊r̂ᵢ⁽ᵗ⁺¹⁾ × cᵢ⁽ᵗ⁺¹⁾⌋))

where cᵢ⁽ᵗ⁺¹⁾ is the known capacity for school sᵢ in year t+1, and the bounds ensure realistic demand levels.
