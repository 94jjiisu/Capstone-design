import numpy as np

#initial steps : 3000
#item w : 300
def selection(dead_players, max_min_ratio):
    player_fitness = []
    scaled_fitness = []
    sum_of_fitness = 0
    point_sum = 0

    for player in dead_players:
        player_fitness.append(player.steps + 300 * player.num_of_items)

    max = np.max(player_fitness)
    min = np.min(player_fitness)

    for fitness in player_fitness:
        scaled_value = (fitness-min) + (max-min)/(max_min_ratio-1)
        scaled_fitness.append(scaled_value)
        sum_of_fitness += scaled_value

    point = np.random.random(1)[0] * sum_of_fitness
    player_index = 0
    for fitness in scaled_fitness:
        point_sum += fitness
        if point < point_sum:

            return player_index

        else:
            player_index += 1
