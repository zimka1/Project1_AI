import math
import random
import sys

# Main algorithm parameters
number_of_cities = 0
number_of_singles_to_stop = 60  # Number of generations without improvement before stopping
population_size = 250  # Population size
children_size = int(population_size * 2 / 3)  # Number of children generated in each iteration

# Mutation chances
invert_mutation_chance = 0.25
scramble_mutation_chance = 0.50
shift_mutation_chance = 0.75
swap_mutation_chance = 1.0


# Calculate Euclidean distance between two points
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Calculate the total distance of the path
def path(roads_weight, cities):
    total_sum = 0
    for i in range(number_of_cities):
        if i == number_of_cities - 1:
            total_sum += roads_weight[cities[i]][cities[0]]
        else:
            total_sum += roads_weight[cities[i]][cities[i + 1]]

    return round(total_sum, 2)


# Calculate the fitness of an individual (inverse of distance)
def fitness(x):
    return 1 / x if x > 0 else 0


# Invert mutation: reverse a random section of the path
def invert_mutation(child):
    a, b = sorted(random.sample(range(number_of_cities), 2))
    child[a:b] = reversed(child[a:b])
    return child


# Scramble mutation: shuffle a random section of the path
def scramble_mutation(child):
    a, b = sorted(random.sample(range(number_of_cities), 2))
    random.shuffle(child[a:b])
    return child


# Shift mutation: move a random segment to another position
def shift_mutation(child):
    a, b = sorted(random.sample(range(number_of_cities), 2))
    subsequence = child[a:b]
    child = child[:a] + child[b:]
    insert_position = random.randint(0, len(child))
    child = child[:insert_position] + subsequence + child[insert_position:]
    return child


# Swap mutation: swap two random elements in the path
def swap_mutation(child):
    a, b = sorted(random.sample(range(number_of_cities), 2))
    child[a], child[b] = child[b], child[a]
    return child


# Adjust the mutation chance based on how close we are to stopping
def adaptive_mutation_chance(quantity_before_stopping):
    quantity_before_stopping = 100 * quantity_before_stopping / number_of_singles_to_stop
    base_chance = 0.1
    return base_chance + 0.01 * quantity_before_stopping


# Apply mutation to the child based on the current mutation chance
def mutation(child, quantity_before_stopping):
    mutation_chance = adaptive_mutation_chance(quantity_before_stopping)

    if random.random() < mutation_chance:
        if random.random() < invert_mutation_chance:
            child = invert_mutation(child)
        elif random.random() < scramble_mutation_chance:
            child = scramble_mutation(child)
        elif random.random() < shift_mutation_chance:
            child = shift_mutation(child)
        elif random.random() < swap_mutation_chance:
            child = swap_mutation(child)

    return child


# Create a new child by combining segments of two parents and applying mutation
def make_child(parent1, parent2, quantity_before_stopping):
    start, end = sorted(random.sample(range(number_of_cities), 2))

    child = [-1] * number_of_cities
    child[start:end] = parent1[start:end]  # Copy a segment from the first parent

    parent2_index = 0
    for i in range(number_of_cities):
        if child[i] == -1:
            while parent2[parent2_index] in child:
                parent2_index += 1
            child[i] = parent2[parent2_index]  # Fill in the rest from the second parent

    child = mutation(child, quantity_before_stopping)  # Apply mutation
    return child


# Perform crossover to create a new generation of children
def crossover(top_parents, quantity_before_stopping):
    queue_parents = random.sample(range(children_size), children_size)
    children = []

    for i in range(children_size):
        if i != children_size - 1:
            child = make_child(top_parents[queue_parents[i]][2], top_parents[queue_parents[i + 1]][2], quantity_before_stopping)
        else:
            child = make_child(top_parents[queue_parents[i]][2], top_parents[queue_parents[0]][2], quantity_before_stopping)

        if child not in children:
            children.append(child)

    return children


# Calculate the total fitness of the population
def colculate_total_fitness(population):
    return sum(ind[0] for ind in population)


# Build cumulative probabilities for roulette selection
def build_cumulative_probabilities(population, total_fitness):
    cumulative = []
    accumulation_of_probabilities = 0

    for ind in population:
        accumulation_of_probabilities += ind[0]
        cumulative.append(accumulation_of_probabilities / total_fitness)

    return cumulative


# Select one individual based on cumulative probabilities (roulette selection)
def select_one(population, cumulative):
    chance = random.random()

    for i in range(population_size):
        if chance <= cumulative[i]:
            if population[i] != [fitness(sys.maxsize), sys.maxsize, [0] * number_of_cities]:
                return population[i]

    return population[-1]


# Select all parents using roulette selection
def select_all_parents(population):
    total_fitness = colculate_total_fitness(population)
    cumulative = build_cumulative_probabilities(population, total_fitness)
    selected = []

    for _ in range(children_size):
        selected.append(select_one(population, cumulative))

    return selected


def main():
    global number_of_cities

    # Input: number of cities and their coordinates
    number_of_cities = int(input())
    cities_coordinates = [tuple(map(int, input().split())) for _ in range(number_of_cities)]

    # Build distance matrix (roads_weight)
    roads_weight = []
    for i in range(number_of_cities):
        row = []
        for j in range(number_of_cities):
            road_weight = round(
                euclidean_distance(cities_coordinates[i][0], cities_coordinates[i][1], cities_coordinates[j][0], cities_coordinates[j][1]), 2
            )
            row.append(road_weight)
        roads_weight.append(row)

    # Initialize population with random paths
    population = [[fitness(sys.maxsize), sys.maxsize, [0] * number_of_cities] for _ in range(population_size)]

    for i in range(population_size):
        random_cities = random.sample(range(0, number_of_cities), number_of_cities)
        distance = path(roads_weight, random_cities)

        child = [fitness(distance), distance, random_cities]

        if child not in population:
            population[i] = child

    population.sort(key=lambda x: x[0])

    print("Generation number 1")

    for row in population:
        print(row)

    print("<<<<<<<<<<>>>>>>>>>>")

    stop_flag = 0
    last_best_fitness = 0
    quantity_before_stopping = 0
    number_of_generations = 0

    while not stop_flag:
        # Create the next generation of children from selected parents
        children = crossover(select_all_parents(population), quantity_before_stopping)

        # Evaluate the fitness of each child and update the population
        for i in range(len(children)):
            distance = path(roads_weight, children[i])
            child = [fitness(distance), distance, children[i]]

            if child not in population:
                population[i] = child

        population.sort(key=lambda x: x[0])

        # Check if the best fitness remains the same across generations
        if last_best_fitness == population[population_size - 1][0]:
            quantity_before_stopping += 1
            print(quantity_before_stopping)
        else:
            last_best_fitness = population[population_size - 1][0]
            quantity_before_stopping = 1

            print(f"Generation number {number_of_generations}")
            for row in population:
                print(row)
            print("<<<<<<<<<<>>>>>>>>>>")

        # Stop the algorithm if no improvement for a specified number of generations
        if quantity_before_stopping == number_of_singles_to_stop:
            stop_flag = 1

        number_of_generations += 1

    # Output the best individual
    best_individual = population[population_size - 1]

    print("Best individual:")
    print(f"Fitness: {best_individual[0]}")
    print(f"Distance: {best_individual[1]}")
    print(f"Path: {best_individual[2]}")
    print("Number of generations:", number_of_generations)


if __name__ == "__main__":
    main()
