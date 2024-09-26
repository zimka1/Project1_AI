import random
import os


def generate_random_tests(num_tests=1, min_cities=25, max_cities=50, max_coordinate=30):
    os.makedirs('test_cases', exist_ok=True)

    for test_num in range(num_tests):
        number_of_cities = random.randint(min_cities, max_cities)

        file_name = f'test_cases/test_case_{test_num + 1}.txt'

        with open(file_name, 'w') as f:
            f.write(f"{number_of_cities}\n")

            for _ in range(number_of_cities):
                x = random.randint(0, max_coordinate)
                y = random.randint(0, max_coordinate)
                f.write(f"{x} {y}\n")

        print(f"Создан тест: {file_name}")


if __name__ == "__main__":
    generate_random_tests()
