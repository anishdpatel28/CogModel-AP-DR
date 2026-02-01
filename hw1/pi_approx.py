import random

def estimate_pi(num_points):
    points_inside = 0

    for i in range(num_points):
        if i % 1000 == 0:
            print(f"\rProgress: {i}/{num_points}", end="", flush=True)
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        if (x**2 + y**2) <= 1:
            points_inside += 1
    print(f"\rProgress: {num_points}/{num_points}", end="", flush=True)

    pi_estimate = 4 * (points_inside / num_points)
    return pi_estimate

if __name__ == "__main__":
    print("How many iterations would you like to do?")
    points = input("> ")
    try:
        points = int(points)
    except ValueError:
        print("Invalid input, defaulting to 100000 iterations")
        points = 100000
    
    result = estimate_pi(points)
    print(f"\nApproximated Pi: {result}")