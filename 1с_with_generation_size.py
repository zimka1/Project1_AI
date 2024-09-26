import math
import random


def pathhhh(roads_weight, cities):
    total_sum = 0
    for i in range(len(cities)):
        if i == len(cities) - 1:
            total_sum += roads_weight[cities[i]][cities[0]][0]
        else:
            total_sum += roads_weight[cities[i]][cities[i + 1]][0]
        # print(cities[i], total_sum)
    return round(total_sum, 2)

def ost_tree(roads_weight, a, n):
    used = [0] * n
    min_edge = [float('inf')] * n
    min_edge[a] = 0
    mst_weight = 0
    last_node = a
    mst_path = []  # Список для хранения пути

    for _ in range(n):
        v = -1
        for i in range(n):
            if not used[i] and (v == -1 or min_edge[i] < min_edge[v]):
                v = i

        if min_edge[v] == float('inf'):
            return mst_weight, mst_path

        mst_weight += min_edge[v]
        used[v] = 1
        last_node = v  # Запоминаем последнюю посещенную вершину
        mst_path.append(v)  # Добавляем вершину в путь

        for weight, to in roads_weight[v]:
            if weight < min_edge[to] and not used[to]:
                min_edge[to] = weight

    # Добавляем путь от последней вершины к начальной для замыкания цикла


    mst_weight += roads_weight[last_node][a][0]
    mst_path.append(a)  # Замыкаем путь на начальную вершину

    return mst_weight, mst_path


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def main():
    n = int(input())
    cities_coordinates = [tuple(map(int, input().split())) for _ in range(n)]
    roads_weight = []
    for i in range(n):
        row = []
        for j in range(n):
            road_weight = round(
                euclidean_distance(cities_coordinates[i][0], cities_coordinates[i][1], cities_coordinates[j][0],
                                   cities_coordinates[j][1]), 2)
            row.append((road_weight, j))
        roads_weight.append(row)

    all_variants = []
    paths = []  # Список для хранения всех путей
    for i in range(n):
        total_sum, path = ost_tree(roads_weight, i, n)
        all_variants.append(total_sum)
        paths.append(path)

    # Находим минимальный путь и соответствующий маршрут
    min_sum = min(all_variants)
    min_index = all_variants.index(min_sum)
    min_path = paths[min_index]
    print("Минимальная длина пути:", min_sum)
    print("Маршрут:", min_path)
    print(pathhhh(roads_weight, min_path))




if __name__ == "__main__":
    main()
