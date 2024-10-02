import subprocess
import os
import matplotlib.pyplot as plt


def run_genetic_algorithm(test_file):
    population_sizes = []
    correct_answers = []

    # for population_size in range(50, 600, 50):
    #     open('test_results_1.txt', 'w').close()
    #
    #     k = 0  # Variable to count correct answers for each population size
    #     for i in range(100):
    #         print(f"Запуск теста {i + 1} при популяции {population_size} на файле {test_file}...")
    #         output_file = 'test_results_1.txt'  # Файл для результатов
    #         process = subprocess.Popen(['python', '1c_tournament.py', test_file, output_file, str(population_size)],
    #                                    stdout=subprocess.PIPE,
    #                                    stderr=subprocess.PIPE)
    #
    #         stdout, stderr = process.communicate()
    #
    #         if stderr:
    #             print("Ошибки:", stderr.decode())
    #
    #     # Reading the results from the test_results_1.txt
    #     with open('test_results_1.txt', 'r') as f:
    #         for i in range(100):
    #             ans = float(f.readline())
    #             if ans == 895.69:
    #                 k += 1
    #             else:
    #                 print(ans)

        # Storing the results for plotting
    population_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
    correct_answers = [36, 39, 55, 53, 57, 60, 64, 61, 63, 70, 71]

        # Logging the results to total_result.txt
        # with open("total_result.txt", 'a') as f:
        #     f.write(f"Правильных ответов для популяции {population_size}: {k}/100\n")

    # Return the data for plotting
    return population_sizes, correct_answers


def plot_results(population_sizes, correct_answers):
    plt.plot(population_sizes, correct_answers, marker='o')

    # Set tick intervals on x-axis (population size) every 50 units
    plt.xticks(range(0, max(population_sizes) + 50, 50))

    # Set tick intervals on y-axis (correct answers) every 1 unit
    plt.yticks(range(0, max(correct_answers) + 5, 5))

    plt.xlabel('Population Size')
    plt.ylabel('Correct Answers')
    plt.title('Relationship between Population Size and Correct Answers')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    test_file = f'test_cases/test_case_1.txt'
    if os.path.exists(test_file):
        population_sizes, correct_answers = run_genetic_algorithm(test_file)
        plot_results(population_sizes, correct_answers)
    else:
        print(f"Файл {test_file} не найден.")
