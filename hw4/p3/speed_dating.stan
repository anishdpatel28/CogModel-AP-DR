data {
  int<lower=0> N;           // Number of observations in training data
  int<lower=1> K;           // Number of predictors (3 in our case)
  matrix[N, K] X;           // Standardized predictor matrix
  array[N] int<lower=0, upper=1> y; // Binary outcome (decision)
}

parameters {
  real alpha;               // Intercept
  vector[K] beta;           // Weights for the 3 attributes
}

model {
  // Priors: Weakly informative on the logit scale
  alpha ~ normal(0, 1.5);
  beta ~ normal(0, 1.5);
  
  // Likelihood: Logistic regression mapping linear model to binary outcome
  y ~ bernoulli_logit(alpha + X * beta);
}