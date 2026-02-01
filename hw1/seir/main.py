import matplotlib.pyplot as plt

from simulation import simulate_seir
from viz import plot_results


if __name__ == "__main__":


    parameters = (3.0, 0.5, 0.5)

    # S0, E0, I0, R0
    inits = (9999.0, 1.0, 0.0, 0.0)

    sim = simulate_seir(parameters, inits, days=100)
    
    ### Expected result
    fig = plot_results(sim[2])
    plt.show()
