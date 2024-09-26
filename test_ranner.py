import subprocess
import os


# def generate_tests():
#     print("Генерация тестов...")
#     subprocess.run(['python', 'tests_maker.py'])


def run_genetic_algorithm(test_file):
    results = []
    for population_size in range(300, 301, 100):

        open('test_results_1.txt', 'w').close()

        for i in range(100):
            print(f"Запуск теста {i + 1} при популяции {population_size} на файле {test_file}...")
            output_file = 'test_results_1.txt'  # Файл для результатов
            process = subprocess.Popen(['python', '1c.py', test_file, output_file, str(population_size)],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)

            stdout, stderr = process.communicate()
        k = 0
        with open('test_results_1.txt', 'r') as f:
            for i in range(100):
                # ans = f.readline()
                # print(ans)
                ans = float(f.readline())
                if ans == 895.69:
                    k += 1
                else:
                    print(ans)

        with open("total_result.txt", 'a') as f:
            f.write(f"Правильных ответов для популяции {population_size}: {k}/100" + "\n")

        if stderr:
            print("Ошибки:", stderr.decode())

    return results


if __name__ == "__main__":
    # generate_tests()

    # for test_num in range(1, 2):
    test_file = f'test_cases/test_case_1.txt'
    if os.path.exists(test_file):
        run_genetic_algorithm(test_file)
    else:
        print(f"Файл {test_file} не найден.")

