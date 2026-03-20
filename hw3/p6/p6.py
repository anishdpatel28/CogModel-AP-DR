import pandas as pd
import arviz as az
from cmdstanpy import CmdStanModel

# 1. Load Data
df = pd.read_csv('response_times.csv', sep=';')

# 2. Prepare data dictionary for Stan
stan_data = {
    'N': len(df),
    'J': df['id'].nunique(),
    'id': df['id'].values,
    'y': df['rt'].values,
    'condition': df['condition'].values,
    'choice': df['choice'].values
}

# 3. Compile and fit
model = CmdStanModel(stan_file='diffusion_multiple.stan')

print("Sampling... (This will take a few minutes)")
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    show_progress=True
)

# 4. View Results
idata = az.from_cmdstanpy(posterior=fit)
summary_df = az.summary(idata)
print(summary_df)

