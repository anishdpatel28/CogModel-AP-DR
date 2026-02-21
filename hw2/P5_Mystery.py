import numpy as np
import pandas as pd

# Number of simulations
num_sims = 100_000

suspects = np.random.choice(
    ['Acrobat', 'Brute', 'Inside Man', 'Phantom'], 
    size=num_sims, 
    p=[0.40, 0.20, 0.25, 0.15]
)

rands = np.random.rand(num_sims)

conditions = [
    # Acrobat's probabilities (80% Skylight, 10% Front Door, 10% Wall Hole)
    (suspects == 'Acrobat') & (rands < 0.80),
    (suspects == 'Acrobat') & (rands >= 0.80) & (rands < 0.90),
    (suspects == 'Acrobat') & (rands >= 0.90),
    
    # Brute's probabilities (5% Skylight, 15% Front Door, 80% Wall Hole)
    (suspects == 'Brute') & (rands < 0.05),
    (suspects == 'Brute') & (rands >= 0.05) & (rands < 0.20),
    (suspects == 'Brute') & (rands >= 0.20),
    
    # Inside Man's probabilities (5% Skylight, 90% Front Door, 5% Wall Hole)
    (suspects == 'Inside Man') & (rands < 0.05),
    (suspects == 'Inside Man') & (rands >= 0.05) & (rands < 0.95),
    (suspects == 'Inside Man') & (rands >= 0.95),
    
    # Phantom's probabilities (10% Skylight, 80% Front Door, 10% Wall Hole)
    (suspects == 'Phantom') & (rands < 0.10),
    (suspects == 'Phantom') & (rands >= 0.10) & (rands < 0.90),
    (suspects == 'Phantom') & (rands >= 0.90)
]

# The corresponding outcomes for the conditions above
choices = [
    'Skylight', 'Front Door', 'Wall Hole',
    'Skylight', 'Front Door', 'Wall Hole',
    'Skylight', 'Front Door', 'Wall Hole',
    'Skylight', 'Front Door', 'Wall Hole'
]

# Apply conditions 
entry_methods = np.select(conditions, choices, default='Unknown')

# Create table of probabilities
df = pd.DataFrame({'Suspect': suspects, 'Entry_Method': entry_methods})
approx_probs = df.value_counts(normalize=True).reset_index(name='Approx_Probability')

print(approx_probs.sort_values(by=['Suspect', 'Entry_Method']).to_string(index=False))