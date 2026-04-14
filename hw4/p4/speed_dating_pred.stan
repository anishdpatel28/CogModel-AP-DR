data {
  int<lower=0> N;
  int<lower=1> K;
  matrix[N, K] X;
  array[N] int<lower=0, upper=1> y;

  int<lower=0> M;
  matrix[M, K] X_test;
}

parameters {
  real alpha;
  vector[K] beta;
}

model {
  alpha ~ normal(0, 1.5);
  beta ~ normal(0, 1.5);
  y ~ bernoulli_logit(alpha + X * beta);
}

generated quantities {
  vector[M] p_test;
  for (m in 1:M)
    p_test[m] = inv_logit(alpha + X_test[m] * beta);
}
