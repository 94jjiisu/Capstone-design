from copy import deepcopy
import numpy as np
import snake_imit_test_game_8angle as game_comp


def selection(dead_players, max_min_ratio, num_selection):
    player_fitness = []
    scaled_fitness = []
    selected_parents = []
    sum_of_fitness = 0

    for player in dead_players:
        player_fitness.append(player.fitness)
        print(player.fitness)
    if len(set(player_fitness)) == 1:
        player_fitness[np.random.randint(0, len(player_fitness))] += 1

    max = np.max(player_fitness)
    min = np.min(player_fitness)

    for fitness in player_fitness:
        scaled_value = (fitness-min) + (max-min)/(max_min_ratio-1)
        scaled_fitness.append(scaled_value)
        sum_of_fitness += scaled_value

    while len(selected_parents) < num_selection:
        point = np.random.random(1)[0] * sum_of_fitness
        point_sum = 0
        player_index = 0
        for fitness in scaled_fitness:
            point_sum += fitness
            if point <= point_sum:
                if player_index in selected_parents:
                    break
                selected_parents.append(player_index)
                break

            else:
                player_index += 1

    return selected_parents

def crossover(candidates_index ,candidates, num_children, input, hidden1, hidden2, output):  ### Uniform Binary Crossover
    selected_parents = []
    superior = []

    for index in candidates_index:
        selected_parents.append(candidates[index])

    for index in candidates_index:
        superior.append(candidates[index])

    for i in range(int(num_children / 2)):
        parent1 = np.random.choice(selected_parents)
        parent2 = np.random.choice(selected_parents)
        while (parent1.x == parent2.x) and (parent1.y == parent2.y):
            parent2 = np.random.choice(selected_parents)

        offspring1 = game_comp.Player_Ai(input,hidden1,hidden2,output)
        offspring2 = game_comp.Player_Ai(input,hidden1,hidden2,output)  ### genome의 구조를 그대로 복제한 자식 2명 생성

        offspring1.neural_net.weight1 = deepcopy(parent1.neural_net.weight1)  ### 부모 2명의 가중치 1층을 복제한 자식 2명 생성
        offspring2.neural_net.weight1 = deepcopy(parent2.neural_net.weight1)

        mask = np.random.uniform(0, 1, size=offspring1.neural_net.weight1.shape)  ### 마스크를 이용해 유니폼 크로스오버
        offspring1.neural_net.weight1[mask > 0.5] = parent2.neural_net.weight1[mask > 0.5]
        offspring2.neural_net.weight1[mask > 0.5] = parent1.neural_net.weight1[mask > 0.5]

        offspring1.neural_net.weight2 = deepcopy(parent1.neural_net.weight2)  ### 가중치 2층 복제
        offspring2.neural_net.weight2 = deepcopy(parent2.neural_net.weight2)

        mask = np.random.uniform(0, 1, size=offspring1.neural_net.weight2.shape)  ### 가중치 2층 교배
        offspring1.neural_net.weight2[mask > 0.5] = parent2.neural_net.weight2[mask > 0.5]
        offspring2.neural_net.weight2[mask > 0.5] = parent1.neural_net.weight2[mask > 0.5]

        offspring1.neural_net.weight3 = deepcopy(parent1.neural_net.weight3)  ### 가중치 3층 복제
        offspring2.neural_net.weight3 = deepcopy(parent2.neural_net.weight3)

        mask = np.random.uniform(0, 1, size=offspring1.neural_net.weight3.shape)  ### 가중치 3층 교배
        offspring1.neural_net.weight3[mask > 0.5] = parent2.neural_net.weight3[mask > 0.5]
        offspring2.neural_net.weight3[mask > 0.5] = parent1.neural_net.weight3[mask > 0.5]

        superior.append(offspring1)  ###
        superior.append(offspring2)  ### n개의 자식을 생성해서 채워넣음

    return superior

def mutation(superior, prob_mutation, total_num, m, std,input,hidden1,hidden2,output): # random uniform mutation
    next_generation = []
    for i in range(int(total_num/len(superior)) ):
        for bg in superior:
            new_genome = game_comp.Player_Ai(input,hidden1,hidden2,output)
            new_genome.neural_net.weight1 = deepcopy(bg.neural_net.weight1)
            new_genome.neural_net.weight2 = deepcopy(bg.neural_net.weight2)
            new_genome.neural_net.weight3 = deepcopy(bg.neural_net.weight3)

            mean = m
            stddev = std

            if np.random.uniform(0, 1) < prob_mutation:
                new_genome.neural_net.weight1 += new_genome.neural_net.weight1 * np.random.normal(mean, stddev, size=(
                bg.neural_net.input_layer, bg.neural_net.hidden_layer)) / 100 * np.random.randint(-1, 2, (bg.neural_net.input_layer, bg.neural_net.hidden_layer))
            if np.random.uniform(0, 1) < prob_mutation:
                new_genome.neural_net.weight2 += new_genome.neural_net.weight2 * np.random.normal(mean, stddev, size=(
                bg.neural_net.hidden_layer, bg.neural_net.hidden_layer2)) / 100 * np.random.randint(-1, 2, (bg.neural_net.hidden_layer, bg.neural_net.hidden_layer2))
            if np.random.uniform(0, 1) < prob_mutation:
                new_genome.neural_net.weight3 += new_genome.neural_net.weight3 * np.random.normal(mean, stddev, size=(
                bg.neural_net.hidden_layer2, bg.neural_net.output_layer)) / 100 * np.random.randint(-1, 2, (bg.neural_net.hidden_layer2, bg.neural_net.output_layer))
            next_generation.append(new_genome)


    return next_generation
