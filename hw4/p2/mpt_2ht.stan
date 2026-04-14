data {
  int<lower=0> n_old;
  int<lower=0> n_new;
  int<lower=0, upper=n_old> hits;
  int<lower=0, upper=n_new> false_alarms;
}

parameters {
  real<lower=0, upper=1> d;
  real<lower=0, upper=1> g;
}

transformed parameters {
  real<lower=0, upper=1> p_hit = d + (1 - d) * g;
  real<lower=0, upper=1> p_fa  = (1 - d) * g;
}

model {
  d ~ beta(1, 1);
  g ~ beta(1, 1);

  hits         ~ binomial(n_old, p_hit);
  false_alarms ~ binomial(n_new, p_fa);
}
