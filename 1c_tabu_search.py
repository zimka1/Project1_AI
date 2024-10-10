import math
import random
import matplotlib.pyplot as plt


# Main parameters for the Tabu Search algorithm
number_of_cities = 0
start_tabu_list_size = 5  # Initial size of the Tabu list
tabu_list_size = start_tabu_list_size
max_tabu_list_size = 50  # Maximum size of the Tabu list
number_of_identical_best_to_stop = 60  # Number of iterations without improvement before stopping
tabu_list = []  # The Tabu list for storing forbidden solutions


# Calculate the Euclidean distance between two points
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Calculate the total distance of the current path
def path(roads_weight, cities):
    total_sum = 0

    for i in range(number_of_cities):
        if i == number_of_cities - 1:
            total_sum += roads_weight[cities[i]][cities[0]]
        else:
            total_sum += roads_weight[cities[i]][cities[i + 1]]

    return round(total_sum, 2)


# Generate neighboring solutions by performing swaps, reversals, and shuffling segments
def make_neighbors(current_path, roads_weight):
    neighbours = []

    # Swap elements to create neighbors
    for i in range(len(current_path) - 1):
        for j in range(i, len(current_path)):
            neighbor = current_path[:]
            neighbor[i], neighbor[j] = current_path[j], current_path[i]
            neighbours.append([path(roads_weight, neighbor), neighbor])

    # Reverse segments to create neighbors
    for i in range(len(current_path) - 2):
        for j in range(i + 2, len(current_path)):
            neighbor = current_path[:]
            neighbor[i:j] = reversed(current_path[i:j])
            neighbours.append([path(roads_weight, neighbor), neighbor])

    # Shuffle segments to create neighbors
    for i in range(len(current_path) - 2):
        for j in range(i + 2, len(current_path)):
            neighbor = current_path[:]
            segment = neighbor[i:j]
            random.shuffle(segment)
            neighbor[i:j] = segment
            neighbours.append([path(roads_weight, neighbor), neighbor])

    return neighbours

# Function to plot the path (ChatGPT)
def plot_path(cities_coordinates, best_path):
    x_coords = [cities_coordinates[city][0] for city in best_path]
    y_coords = [cities_coordinates[city][1] for city in best_path]

    # Add the first city at the end to complete the loop
    x_coords.append(cities_coordinates[best_path[0]][0])
    y_coords.append(cities_coordinates[best_path[0]][1])

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, y_coords, '-o', color='blue', label='Path')
    plt.scatter(x_coords, y_coords, color='red', zorder=5)
    plt.title('Best Found Path')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    for i, city in enumerate(best_path):
        plt.annotate(str(city), (cities_coordinates[city][0], cities_coordinates[city][1]))

    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    global number_of_cities
    global tabu_list
    global tabu_list_size

    # Input: number of cities and their coordinates
    number_of_cities = int(input())
    cities_coordinates = [tuple(map(int, input().split())) for _ in range(number_of_cities)]

    # Build distance matrix (roads_weight) based on city coordinates
    roads_weight = []
    for i in range(number_of_cities):
        row = []
        for j in range(number_of_cities):
            road_weight = round(
                euclidean_distance(cities_coordinates[i][0], cities_coordinates[i][1], cities_coordinates[j][0], cities_coordinates[j][1]), 2
            )
            row.append(road_weight)
        roads_weight.append(row)

    # Generate a random initial path
    random_path = random.sample(range(0, number_of_cities), number_of_cities)
    current_path = [path(roads_weight, random_path), random_path]

    stop_flag = 0
    number_of_identical_best = 0
    number_of_iterations = 0
    best_result = current_path

    while not stop_flag:
        # Generate neighborhood and filter out the ones in the Tabu list
        neighborhood = make_neighbors(current_path[1], roads_weight)
        neighborhood = [neighbor for neighbor in neighborhood if neighbor[1] not in tabu_list]

        # If no valid neighbors, randomize the current path and reset parameters
        if len(neighborhood) == 0:
            random.shuffle(current_path[1])
            current_path[0] = path(roads_weight, current_path[1])
            number_of_identical_best = 0
            tabu_list = []
            tabu_list_size = start_tabu_list_size
            continue

        # Update the current path to the best neighbor
        current_path = min(neighborhood)

        # If the current path is better than the best found so far, update the best result
        if current_path[0] < best_result[0]:
            best_result = current_path
            number_of_identical_best = 0
        else:
            number_of_identical_best += 1

        # Add the current path to the Tabu list
        tabu_list.append(current_path[1])

        # Dynamically adjust the Tabu list size based on iterations without improvement
        tabu_list_size = min(tabu_list_size + (number_of_identical_best * 100 / number_of_identical_best_to_stop) * max_tabu_list_size * 0.01,
                             max_tabu_list_size)

        # Remove the oldest element from the Tabu list if it exceeds the allowed size
        if len(tabu_list) > tabu_list_size:
            tabu_list.pop(0)

        # Stop if no improvement after the specified number of iterations
        if number_of_identical_best > number_of_identical_best_to_stop:
            stop_flag = 1

        number_of_iterations += 1

    # Output the best solution found and its distance
    print(f"Distance: {best_result[0]}")
    print(f"Path: {best_result[1]}")
    print("Number of iterations:", number_of_iterations)

    plot_path(cities_coordinates, best_result[1])


if __name__ == "__main__":
    main()
