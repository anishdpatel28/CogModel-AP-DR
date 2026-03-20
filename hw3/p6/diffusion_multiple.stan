data {
    int<lower=1> N; // number of rows
    int<lower=1> J; // number of people
    array[N] int<lower=1, upper=J> id;   // person ID for each trial/row
    array[N] real<lower=0> y; // response time
    array[N] int<lower=1, upper=2> condition; // condition indicator
    array[N] int<lower=0, upper=1> choice; // choice indicator
}

transformed data {
  // init a vector to hold the minimum response times
  real eps = 1e-6;
  array[J] real rt_min;

  // initialize to +inf
  for (j in 1:J) {
    rt_min[j] = positive_infinity();
  }

  // compute per-person minimum RT
  for (n in 1:N) {
    if (y[n] < rt_min[id[n]]) {
        rt_min[id[n]] = y[n];
    }
  }
}

parameters {
    // Each person has their own parameters
    // Task difficulty is indexed by drift rate (v), so we need one for each condition:
    array[J] real v1; // Drift rates for Condition 1
    array[J] real v2; // Drift rates for Condition 2
    
    array[J] real<lower=0> a;
    array[J] real<lower=0, upper=1> beta;
    array[J] real<lower=0, upper=1> tau_raw;
}

transformed parameters {
  // A good way to bound non-decision time (tau)
  array[J] real<lower=0> tau;
  for (j in 1:J) {
    tau[j] = tau_raw[j] * (rt_min[j] - eps);
  }
}

model {
    // Priors
    for (j in 1:J) {
        v1[j] ~ normal(0, 3);
        v2[j] ~ normal(0, 3);
        a[j] ~ normal(1.5, 1);
        // beta and tau_raw are constrained between 0 and 1, 
        beta[j] ~ beta(2, 2); 
    }

    // Likelihood
    for (n in 1:N) {
        // Condition 1
        if (condition[n] == 1) {
            if (choice[n] == 1) {
                // Hit upper boundary
                y[n] ~ wiener(a[id[n]], tau[id[n]], beta[id[n]], v1[id[n]]); 
            } else {
                // Hit lower boundary
                y[n] ~ wiener(a[id[n]], tau[id[n]], 1 - beta[id[n]], -v1[id[n]]);
            }
        }

        // Condition 2
        if (condition[n] == 2) {
            if (choice[n] == 1) {
                // Hit upper boundary
                y[n] ~ wiener(a[id[n]], tau[id[n]], beta[id[n]], v2[id[n]]);
            } else {
                // Hit lower boundary
                y[n] ~ wiener(a[id[n]], tau[id[n]], 1 - beta[id[n]], -v2[id[n]]);
            }
        }
    }
}