import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the data
df = pd.read_csv('Speed Dating Data.csv', encoding='ISO-8859-1')

# 2. Select variables of interest: 3 predictors and 1 binary outcome
cols_to_keep = ['attr', 'sinc', 'intel', 'dec']
data = df[cols_to_keep].copy()

# 3. Drop rows with missing values in these specific columns
data = data.dropna()

# 4. Separate features (X) and outcome (y)
X = data[['attr', 'sinc', 'intel']]
y = data['dec'].astype(int)

# 5. Standardize the predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")




import cmdstanpy
from cmdstanpy import CmdStanModel

# 1. Compile the Stan model
model = CmdStanModel(stan_file='speed_dating.stan')

# 2. Prepare the data dictionary for Stan
data_dict = {
    'N': X_train.shape[0],
    'K': X_train.shape[1],
    'X': X_train,
    'y': y_train.values
}

# 3. Sample from the posterior
print("Sampling from the posterior...")
fit = model.sample(
    data=data_dict,
    iter_warmup=1000,
    iter_sampling=1000,
    chains=4,
    show_progress=True
)

# 4. Extract summary statistics (includes R-hat, ESS, Mean, and CIs)
summary_df = fit.summary()

# Extract just the parameters of interest
params_of_interest = summary_df.loc[['alpha', 'beta[1]', 'beta[2]', 'beta[3]']]

print("\n--- Posterior Summary ---")
print(params_of_interest[['Mean', '5%', '95%', 'ESS_bulk', 'R_hat']])

# 5. Diagnostic Check
max_rhat = params_of_interest['R_hat'].max()
min_ess = params_of_interest['ESS_bulk'].min()

if max_rhat < 1.05:
    print(f"\nConvergence successful! Maximum R-hat is {max_rhat:.3f}")
else:
    print(f"\nWarning: Convergence issues detected. Maximum R-hat is {max_rhat:.3f}")
