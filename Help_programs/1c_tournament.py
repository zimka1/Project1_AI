import math
import random
import sys

number_of_cities = 0
number_of_singles_to_stop = 60
population_size = 250
children_size = int(population_size * 2 / 3)

# Mutation chances
invert_mutation_chance = 0.25
scramble_mutation_chance = 0.50
shift_mutation_chance = 0.75
swap_mutation_chance = 1.0


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def path(roads_weight, cities):
    total_sum = 0
    for i in range(number_of_cities):
        if i == number_of_cities - 1:
            total_sum += roads_weight[cities[i]][cities[0]]
        else:
            total_sum += roads_weight[cities[i]][cities[i + 1]]

    return round(total_sum, 2)


def fitness(x):
    return 1 / x if x > 0 else 0


def invert_mutation(child):
    a, b = sorted(random.sample(range(number_of_cities), 2))
    child[a:b] = reversed(child[a:b])
    return child


def scramble_mutation(child):
    a, b = sorted(random.sample(range(number_of_cities), 2))
    random.shuffle(child[a:b])
    return child


def shift_mutation(child):
    a, b = sorted(random.sample(range(number_of_cities), 2))

    subsequence = child[a:b]
    child = child[:a] + child[b:]

    insert_position = random.randint(0, len(child))
    child = child[:insert_position] + subsequence + child[insert_position:]

    return child


def swap_mutation(child):
    a, b = sorted(random.sample(range(number_of_cities), 2))
    child[a], child[b] = child[b], child[a]
    return child


def adaptive_mutation_chance(quantity_before_stopping):
    quantity_before_stopping = 100 * quantity_before_stopping / number_of_singles_to_stop
    base_chance = 0.1
    return base_chance + 0.01 * quantity_before_stopping


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


def make_child(parent1, parent2, quantity_before_stopping):
    start, end = sorted(random.sample(range(number_of_cities), 2))

    child = [-1] * number_of_cities
    child[start:end] = parent1[start:end]

    parent2_index = 0

    for i in range(number_of_cities):
        if child[i] == -1:
            while parent2[parent2_index] in child:
                parent2_index += 1
            child[i] = parent2[parent2_index]

    child = mutation(child, quantity_before_stopping)
    return child


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


def tournament_selection(population, tournament_size):
    selected = []

    for _ in range(children_size):
        tournament = random.sample(population, tournament_size)

        tournament.sort(key=lambda x: x[0], reverse=True)
        selected.append(tournament[0])

    return selected


def main():
    global number_of_cities
    global population_size
    global children_size
    global number_of_singles_to_stop

    # number_of_cities = int(input())
    #
    # cities_coordinates = [tuple(map(int, input().split())) for _ in range(number_of_cities)]

    test_file = sys.argv[1]
    output_file = sys.argv[2]
    population_size = int(sys.argv[3])

    children_size = int(population_size * 2 / 3)

    with open(test_file, 'r') as f:
        number_of_cities = int(f.readline().strip())
        cities_coordinates = [tuple(map(int, f.readline().strip().split())) for _ in range(number_of_cities)]

    roads_weight = []

    for i in range(number_of_cities):
        row = []
        for j in range(number_of_cities):
            road_weight = round(
                euclidean_distance(cities_coordinates[i][0], cities_coordinates[i][1], cities_coordinates[j][0], cities_coordinates[j][1]), 2
            )
            row.append(road_weight)
        roads_weight.append(row)

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

    results = []

    while not stop_flag:
        children = crossover(tournament_selection(population, int(children_size / 2)), quantity_before_stopping)

        for i in range(len(children)):
            distance = path(roads_weight, children[i])
            child = [fitness(distance), distance, children[i]]

            if child not in population:
                population[i] = child

        population.sort(key=lambda x: x[0])

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

        if quantity_before_stopping == number_of_singles_to_stop:
            stop_flag = 1

        number_of_generations += 1

    best_individual = population[population_size - 1]

    # results.append(f"{best_population[0]}")
    results.append(f"{best_individual[1]}")
    # results.append(f"{best_population[2]}")
    # results.append(f"{number_of_generations}")

    with open(output_file, 'a') as f:
        for result in results:
            f.write(result + "\n")


if __name__ == "__main__":
    main()
