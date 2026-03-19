data {
  int<lower=1> N;
  vector[N] x;
  vector[N] y;
}

parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}

model {
  // Priors
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  square(sigma) ~ inv_gamma(1, 1);

  // Likelihood
  y ~ normal(alpha + beta * x, sigma);
}

generated quantities {
  vector[N] y_rep;
  for (n in 1:N) {
    y_rep[n] = normal_rng(alpha + beta * x[n], sigma);
  }
}
