import Astar_a2
import GA
import numpy as np
import random
import math
import making_map
import NN
import copy
import pygame
from pygame import *
from making_map import *

PIXELUNIT = 20
ONE_STEP = 1
Itemscore = 1000

# 나와 타겟까지의 방향을 반환하는 함수


def get_grad_dir(point, target):
    if point[0] == target[0]:
        if point[1] <= target[1]:
            return 4

        elif point[1] > target[1]:
            return 0

    if ((point[1] - target[1]) / (point[0] - target[0]) > 2.4241 or
            (point[1] - target[1]) / (point[0] - target[0]) < -2.4241) and\
            point[1] < target[1]:
        return 4

    if ((point[1] - target[1]) / (point[0] - target[0]) > 2.4241 or
            (point[1] - target[1]) / (point[0] - target[0]) < -2.4241) and\
            point[1] > target[1]:
        return 0

    if ((point[1] - target[1]) / (point[0] - target[0]) > 0.4241 and
            (point[1] - target[1]) / (point[0] - target[0]) <= 2.4241) and\
            point[1] < target[1]:
        return 3

    if ((point[1] - target[1]) / (point[0] - target[0]) > 0.4241 and
            (point[1] - target[1]) / (point[0] - target[0]) <= 2.4241) and\
            point[1] > target[1]:
        return 7

    if ((point[1] - target[1]) / (point[0] - target[0]) <= -0.4241 and
            (point[1] - target[1]) / (point[0] - target[0]) >= -2.4241) and\
            point[0] < target[0]:
        return 1

    if ((point[1] - target[1]) / (point[0] - target[0]) <= -0.4241 and
            (point[1] - target[1]) / (point[0] - target[0]) >= -2.4241) and\
            point[0] > target[0]:
        return 5

    if ((point[1] - target[1]) / (point[0] - target[0]) <= 0.4241 or
            (point[1] - target[1]) / (point[0] - target[0]) > -0.4241) and\
            point[0] < target[0]:
        return 2

    if ((point[1] - target[1]) / (point[0] - target[0]) <= 0.4241 or
            (point[1] - target[1]) / (point[0] - target[0]) > -0.4241) and\
            point[0] > target[0]:
        return 6

# 체비쇼프 거리 구하기


def get_chebyshev_dist(loc_1, loc_2):
    result = np.max([np.abs(loc_1[0] - loc_2[0]), np.abs(loc_1[1] - loc_2[1])])
    if loc_1[0] == loc_2[0]:
        if loc_1[1] == loc_2[1]:
            result = 0.5
    return result
# 좀비, 플레이어, 아이템 위치 리스트 반환 함수


def points_checker(z_list=None, p_list=None, i_list=None):
    zombie_points = []
    player_points = []
    item_points = []

    if z_list:
        for zombie in z_list:
            [x, y] = [zombie.x, zombie.y]
            zombie_points.append([x, y])

    if i_list:
        for item in i_list:
            [x, y] = [item.x, item.y]
            item_points.append([x, y])

    if p_list:
        for player in p_list:
            [x, y] = [player.x, player.y]
            player_points.append([x, y])

    return [zombie_points, player_points, item_points]

####### 아이템 클래스 #######
class Item(pygame.sprite.Sprite):
    def __init__(self, maps):
        pygame.sprite.Sprite.__init__(self)  # 유진추가
        self.x = None
        self.y = None
        self.maps = maps
        self.steps = None
        #self.map_size = maps.map_size

        self.image = pygame.image.load('images/itemImg.png')
        self.rect = pygame.Rect(0, 0, 1,1)

    def generate_item(self, map_size=None, zombies=None, players=None, items=None,
                      x=None, y=None):
        self.steps = 0
        if x == None and y == None:
            imp_points = copy.copy(self.maps)
            imp_points = imp_points.tolist()
            for i in points_checker(zombies, players, items):
                imp_points.extend(i)

            while True:
                self.x = random.randint(1, map_size[0]-1)
                self.y = random.randint(1, map_size[1]-1)
                if [self.x, self.y] not in imp_points:
                    break

        else:
            self.x = x
            self.y = y
    # 유진추가


    def hit_update(self, hit_list=None, map_size=None, zombies=None, players=None, items=None, x=None, y=None):
        if hit_list and (self in hit_list):
            self.generate_item(map_size, zombies, players, items)

        else:
            self.rect = pygame.Rect(self.x*PIXELUNIT, self.y*PIXELUNIT, 1, 1)

####### 좀비 클래스 #######
class Zombie(pygame.sprite.Sprite):

    def __init__(self, maps, images, vision, delay):
        pygame.sprite.Sprite.__init__(self)  # 유진추가
        self.x = None
        self.y = None
        self.maps = maps
        #self.map_size = maps.map_size
        self.target_p = None
        self.dir = 4
        self.vision = vision
        self.dist = None
        self.DIRECTIONS = [[0, -1], [1, -1], [1, 0], [1, 1],
        [0, 1], [-1, 1], [-1, 0], [-1, -1]]
        self.delay_steps = delay
        self.steps = 0
        self.zombie_src_image = './images/zombieImg_정면.png'
        self.images = images
        self.index = 0
        self.image = self.images[self.dir][self.index]
        self.rect = pygame.Rect(0, 0, 1, 1)


    # 좀비 생성
    def generate_zombie(self, map_size=None, zombies=None, players=None, items=None, x=None, y=None):
        if x == None and y == None:
            imp_points = copy.deepcopy(self.maps)

            imp_points = imp_points.tolist()
            for i in points_checker(zombies, players, items):
                imp_points.extend(i)

            while True:
                self.x = random.randint(1, map_size[0]-1)
                self.y = random.randint(1, map_size[1]-1)
                if [self.x, self.y] not in imp_points:
                    break

        else:
            self.x = x
            self.y = y

        self.dist = self.vision

    # 플레이어 센싱
    def player_sensor(self, p_list, map_size):
        p_points = points_checker(p_list=p_list)[1]

        start_x = self.x - self.vision
        end_x = self.x + self.vision + 1
        if self.x - self.vision <= 0:
            start_x = 0
        if self.x + self.vision >= map_size[0]:
            end_x = map_size[0]

        start_y = self.y - self.vision
        end_y = self.y + self.vision + 1
        if self.y - self.vision <= 0:
            start_y = 0
        if self.y + self.vision >= map_size[1]:
            end_y = map_size[1]

        vision_area = np.mgrid[start_x:end_x:1,
                               start_y:end_y:1].reshape(2, -1).T

        vision_set = set([tuple(x) for x in vision_area])
        p_set = set([tuple(x) for x in p_points])
        players_loc = np.array([x for x in vision_set & p_set])

        nearest_p_index = None

        for i, player_loc in enumerate(players_loc):
            if get_chebyshev_dist([self.x, self.y], player_loc) < self.dist:
                self.dist = get_chebyshev_dist([self.x, self.y], player_loc)
                nearest_p_index = i

        if self.target_p and (nearest_p_index is not None):
            if get_chebyshev_dist([self.target_p.x, self.target_p.y], [self.x, self.y]) > self.dist:
                self.target_p = p_list[p_points.index(
                    [players_loc[nearest_p_index, 0], players_loc[nearest_p_index, 1]])]

        if not self.target_p and (nearest_p_index is not None):
            self.target_p = p_list[p_points.index(
                [players_loc[nearest_p_index][0], players_loc[nearest_p_index][1]])]

        self.dist = self.vision

    # p_step후에 진행하여 잡아먹는 것으로 구현?
    def z_step(self, map_size, zombies, items):
        self.steps += 1
        trial = 0
        imp_points = copy.deepcopy(self.maps)
        imp_points = imp_points.tolist()

        while not self.target_p:
            if trial == 0:
                imp_points.extend(points_checker(zombies)[0])
            trial += 1
            self.dir = np.random.randint(0, 8)

            if [self.x + self.DIRECTIONS[self.dir][0], self.y + self.DIRECTIONS[
            self.dir][1]] not in imp_points:
                [self.x, self.y] = [self.x + self.DIRECTIONS[self.dir][0],
                                     self.y + self.DIRECTIONS[self.dir][1]]
                break

            # do not move (in random walk case)
            if trial > 10:
                break

        if self.target_p:
            result = Astar_a2.Astar(map_size, imp_points,
                (self.x, self.y), (self.target_p.x, self.target_p.y))
            if result:
                imp_points.extend(points_checker(zombies)[0])
                next = list(result)
                if [next[0], next[1]] not in imp_points:
                    self.dir = self.DIRECTIONS.index([next[0]-self.x, next[1]-self.y])
                    [self.x, self.y] = [self.x + self.DIRECTIONS[self.dir][0],
                                         self.y + self.DIRECTIONS[self.dir][1]]
                else :
                    self.dir = np.random.randint(0, 8)
                    while [self.x + self.DIRECTIONS[self.dir][0], self.y + self.DIRECTIONS[
                    self.dir][1]] in imp_points :
                        trial += 1
                        self.dir = np.random.randint(0, 8)
                        if [self.x + self.DIRECTIONS[self.dir][0], self.y + self.DIRECTIONS[
                        self.dir][1]] not in imp_points :
                            [self.x, self.y] = [self.x + self.DIRECTIONS[self.dir][0],
                                                 self.y + self.DIRECTIONS[self.dir][1]]
                            break
                        if trial > 10:
                            break

    def update(self):
        self.image= self.images[self.dir][self.index]
        self.index = (self.index+1)%4
        self.rect = pygame.Rect(self.x*PIXELUNIT, self.y*PIXELUNIT, 1,1)


####### 플레이어 클래스 #######
class Player_Ai(pygame.sprite.Sprite):
    def __init__(self,input,hidden1,hidden2,output):
        pygame.sprite.Sprite.__init__(self)  # 유진추가
        self.x = None
        self.y = None
        self.maps = None
        self.dist_zombie = None
        self.dist_item = None
        self.dist_wall = None
        self.past_dist_zombie = None
        self.past_dist_item = None
        self.past_dist_wall = None
        self.num_of_items = None
        self.steps = None
        self.target_steps = 0
        self.limit_of_steps = None
        self.neural_net = NN.N_net(input,hidden1,hidden2,output)
        self.vision = None
        self.dir = 5
        self.DIRECTIONS = [[0, -1], [1, -1], [1, 0], [1, 1],
        [0, 1], [-1, 1], [-1, 0], [-1, -1]]

        self.images = None
        self.index = 0
        self.image = None
        self.rect = pygame.Rect(0, 0, 1, 1)
        self.fitness = 0

    # 플레이어 생성 함수
    def generate_player(self, map_size, zombies, players, items, maps, images, vision, limit,
                        x=None, y=None):
        self.maps = maps
        self.vision = vision
        self.images = images
        self.image = self.images[self.dir][self.index]
        self.limit_of_steps = limit

        if x == None and y == None:
            imp_points = copy.deepcopy(self.maps)

            imp_points = imp_points.tolist()

            for i in points_checker(zombies, players, items):
                imp_points.extend(i)


            while True:
                self.x = random.randint(1, map_size[0]-1)
                self.y = random.randint(1, map_size[1]-1)

                if [self.x, self.y] not in imp_points:
                    break

        else:
            self.x = x
            self.y = y

        self.dist_zombie = np.ones(8) * self.vision
        self.dist_item = np.ones(8) * self.vision
        self.dist_wall = np.ones(8) * self.vision
        self.num_of_items = 0
        self.steps = 0
        self.target_steps = 0
        self.fitness = 0


    # 좀비, 아이템, 벽 센싱 함수
    def sensor(self, map_size, zombie_list, item_list):
        self.past_dist_zombie = copy.deepcopy(self.dist_zombie)
        self.past_dist_item = copy.deepcopy(self.dist_item)
        self.past_dist_wall = copy.deepcopy(self.dist_wall)

        # reset
        self.dist_zombie = np.ones(8) * self.vision
        self.dist_item = np.ones(8) * self.vision
        self.dist_wall = np.ones(8) * self.vision

        zombie_points = points_checker(z_list=zombie_list)[0]
        item_points = points_checker(i_list=item_list)[2]
        map_points = self.maps

        start_x = self.x - self.vision
        end_x = self.x + self.vision + 1
        if self.x - self.vision <= 0:
            start_x = 0
        if self.x + self.vision >= map_size[0]:
            end_x = map_size[0]

        start_y = self.y - self.vision
        end_y = self.y + self.vision + 1
        if self.y - self.vision <= 0:
            start_y = 0
        if self.y + self.vision >= map_size[1]:
            end_y = map_size[1]
        vision_area = np.mgrid[start_x:end_x:1,
                               start_y:end_y:1].reshape(2, -1).T

        vision_set = set([tuple(x) for x in vision_area])

        m_set = set([tuple([int(x[0]), int(x[1])]) for x in map_points])
        i_set = set([tuple(x) for x in item_points])
        z_set = set([tuple(x) for x in zombie_points])
        zombies_loc = np.array([x for x in vision_set & z_set])
        maps_loc = np.array([x for x in vision_set & m_set])
        items_loc = np.array([x for x in vision_set & i_set])

        for z in zombies_loc:
            if self.dist_zombie[get_grad_dir([self.x, self.y], z)] > get_chebyshev_dist([self.x, self.y], z):
                self.dist_zombie[get_grad_dir([self.x, self.y], z)] = get_chebyshev_dist([
                    self.x, self.y], z)

        for i in items_loc:
            if self.dist_item[get_grad_dir([self.x, self.y], i)] > get_chebyshev_dist([self.x, self.y], i):
                self.dist_item[get_grad_dir([self.x, self.y], i)] = get_chebyshev_dist([
                    self.x, self.y], i)

        for m in maps_loc:
            if self.dist_wall[get_grad_dir([self.x, self.y], m)] > get_chebyshev_dist([self.x, self.y], m):
                self.dist_wall[get_grad_dir([self.x, self.y], m)] = get_chebyshev_dist([
                    self.x, self.y], m)

    # 플레이어 스텝 함수
    def p_step(self, map_size, w_penalty ,zombies=None, players=None):
        trial = 0
        imp_points = copy.deepcopy(self.maps)
        imp_points = imp_points.tolist()
        imp_points.extend(points_checker(p_list = players)[1])

        while set(np.concatenate([self.dist_wall, self.dist_zombie, self.dist_item], axis=0)) == {self.vision}:
            trial += 1
            self.dir = np.random.randint(0, 8)
            if [self.x + self.DIRECTIONS[self.dir][0], self.y + self.DIRECTIONS[self.dir][1]] not in imp_points:
                [self.x, self.y] = [self.x + self.DIRECTIONS[self.dir][0], self.y + self.DIRECTIONS[self.dir][1]]
                self.steps += ONE_STEP
                break

            # do not move (in random walk case)
            if trial > 10:
                self.steps += ONE_STEP
                break

        if (set(np.concatenate([self.dist_wall, self.dist_zombie, self.dist_item], axis=0)) != {self.vision}):
            input = np.ones(
                24) / np.concatenate([self.dist_wall, self.dist_zombie, self.dist_item], axis=0)
            output = self.neural_net.forward(input)
            self.dir = np.argmax(output)

            if [self.x + self.DIRECTIONS[self.dir][0], self.y + self.DIRECTIONS[self.dir][1]] not in imp_points:
                [self.x, self.y] = [self.x + self.DIRECTIONS[self.dir][0], self.y + self.DIRECTIONS[self.dir][1]]
                self.steps += ONE_STEP
            else:
                while trial < 10:
                    trial += 1
                    self.dir = np.random.randint(0, 8)
                    if [self.x + self.DIRECTIONS[self.dir][0], self.y + self.DIRECTIONS[self.dir][1]] not in imp_points:
                        [self.x, self.y] = [self.x + self.DIRECTIONS[self.dir][0], self.y + self.DIRECTIONS[self.dir][1]]
                        self.steps += ONE_STEP
                        self.fitness += w_penalty
                        break

    def calc_fitness(self):
        """
        wall_diff = self.dist_wall - self.past_dist_wall
        item_diff = self.dist_item - self.past_dist_item
        """
        zomb_diff = self.dist_zombie - self.past_dist_zombie
        """
        if np.sum(wall_diff) > 0:
            self.fitness += 50
        elif np.sum(wall_diff) < 0:
            self.fitness -= 100
        elif np.sum(wall_diff) == 0:
            self.fitness = self.fitness

        if np.sum(item_diff) > 0:
            self.fitness -= 200
        elif np.sum(item_diff) < 0:
            self.fitness += 100
        elif np.sum(item_diff) == 0:
            self.fitness -= 50
        """
        if np.sum(zomb_diff) > 0:
            self.fitness += 1 #/ np.max([np.min(zomb_diff), 1])
        elif np.sum(zomb_diff) < 0:
            self.fitness -= 2 #/ np.max([np.min(zomb_diff), 1])
        elif np.sum(zomb_diff) == 0:
            self.fitness = self.fitness

    def show_fitness(self):
        return self.fitness

    def update(self, hit_list=None):

        self.image = self.images[self.dir][self.index]
        self.index = (self.index+1)%4
        self.rect = pygame.Rect(self.x*PIXELUNIT, self.y*PIXELUNIT, 1, 1)
