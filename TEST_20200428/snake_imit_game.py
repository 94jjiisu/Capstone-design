import random
import sys
import time as TTIME
import math
import pygame
import making_map
import GA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import pandas as pd
from pygame.locals import *
from test_game_8angle import *
from making_map import *

FPS = 150

MAP_S = [64, 35]
PIXELUNIT = 20
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

ZOMBIE_DELAY = 2
ZOMBIE_NUM = 9
ZOMBIE_VISION = 25
ZOMBIE_NUM_MAX = 20

MUTATION_PROB = 0.1
PARENTS_NUM = 2
CHILDREN_NUM = 4
TOTAL_PNUM = (PARENTS_NUM + CHILDREN_NUM)
GENERATION_NUM = 1
MAXMIN_RATIO = 100

PLAYER_VISION = 30
PLAYER_LIMIT_STEPS = 300
WALL_TOUCH_PENALTY = -1

ITEM_NUM = 20
GETITEM = 50  #아이템을 먹으면 늘어나는 한계 스텝의 수
itembonus = 10 # 아이템 먹으면 얻는 점수
stepbonus = 0 # 1스텝마다 얻는 점수
MAX_ITEM_STEP = 400

mean = 25
stddev = 10

input = 24
hidden1 = 48
hidden2 = 48
output = 8

level_1to2 =  1
level_2to3 = 2
level_3to4 = 3
GENERATION_MAX = 5

FIXED_LOCATION = True
ORDERED = True
ROULETTE = False

"""
item_bonus와 calc_fitness는 self.fitness에 직접 조작을 가합니다.
두 변수를 꼭 체크하고 아래 함수를 조작하세요.
fitness를 본 game file내에서 조작하도록 주정했으므로 아래 함수만 고치면 GA에도 자동적용됩니다.
WALL_TOUCH_PENALTY도 유의하세요.
"""
def fitness_func(player):
    return player.target_steps

"""
MEMO
아래 1,2,3모두 NN에서 *10 뺸 것
1. target_steps / 24 48 48 8 relu / mu 25 std 10 / prob 0.2 /  p and c 20 40 / zombie_max 20 / z_vision 25 / p_vision 25 / 나누기dist / 순서대로 selection
2. target_steps / 24 48 48 8 relu / mu 25 std 10 / prob 0.1 /  p and c 20 40 / zombie_max 20 / z_vision 25 / p_vision 30 / 나누기dist / 순서대로 selection
"""



first_location_list = []

best_fitness_list = []
average_fitness_list = []

best_steps_list = []
average_steps_list = []


'''###### TEST #####'''
TESTMODE = False
ELITES_TEST = False
STUPIDS_TEST = False

class stupids:
    child=[]
    avg_step_list = []
    best_step_list = []

class elites:
    child=[]
    avg_step_list = []
    best_step_list = []

stupids_avg = 0
stupids_best = 0
elites_avg = 0
elites_best = 0

game_end = False

################################################################
# 파이게임 초기 설정
pygame.init()
DISPLAYSURF = pygame.display.set_mode((MAP_S[0]*PIXELUNIT, MAP_S[1]*PIXELUNIT))
FPSCLOCK = pygame.time.Clock()
BASICFONT1 = pygame.font.Font('fonts/DungGeunMo.ttf', 32)
BASICFONT2 = pygame.font.Font('fonts/DungGeunMo.ttf', 120)
BASICFONT3 = pygame.font.Font('fonts/DungGeunMo.ttf', 55)

# 게임화면 초기 설정 변수 & 함수

pygame.display.set_caption("Zombie game")
icon = pygame.image.load("images/zombie.png")
pygame.display.set_icon(icon)

DISPLAYSURF.fill((0,0,0))

def text_objects(text, BASICFONT1):
    textSurface = BASICFONT1.render(text, True, (0,0,0))
    return textSurface, textSurface.get_rect()

def button(msg,x,y,w,h,ic,ac,action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    if x+w > mouse[0] > x and y+h > mouse[1] > y:
        pygame.draw.rect(DISPLAYSURF, ac,(x,y,w,h))

        if click[0] == 1 and action != None:
            action()
    else:
        pygame.draw.rect(DISPLAYSURF, ic,(x,y,w,h))

    smallText = pygame.font.Font('fonts/DungGeunMo.ttf', 60)
    textSurf, textRect = text_objects(msg, smallText)
    textRect.center = ( int(x+(w/2)), int(y+(h/2)) )
    DISPLAYSURF.blit(textSurf, textRect)

def draw_back():
    background = pygame.image.load("images/maze2.png")
    DISPLAYSURF.blit(pygame.transform.scale(background,(64*20,35*20)), (0,0))

def draw_zombie():
    zom = pygame.image.load("images/zom.png")
    DISPLAYSURF.blit(pygame.transform.scale(zom,(350,350)), (360,160))

def draw_player():
    play = pygame.image.load("images/people.png")
    DISPLAYSURF.blit(pygame.transform.scale(play,(350,350)), (600,160))

def draw_over():

    text1 = BASICFONT2.render('Generation ', True, (BLACK))
    text2 = BASICFONT2.render(str(GENERATION_NUM), True, (BLACK))
    text3 = BASICFONT2.render('->', True, (BLACK))
    text4 = BASICFONT2.render(str(GENERATION_NUM+1), True, (139, 000, 000))
    DISPLAYSURF.blit(text1, (350,80))
    DISPLAYSURF.blit(text2, (350,200))
    DISPLAYSURF.blit(text3, (600, 200))
    DISPLAYSURF.blit(text4, (840, 200))

def draw_change():
    if GENERATION_NUM >= level_1to2:
        lv = 2
    elif GENERATION_NUM >= level_2to3:
        lv = 3
    elif GENERATION_NUM >= level_3to4:
        lv = 4
    text_change_1 = BASICFONT2.render('Change Map ', True, (BLACK))
    text_change_2 = BASICFONT2.render('Lv.' + str(lv-1) + ' ->  ' + str(lv), True, (BLACK))
    DISPLAYSURF.blit(text_change_1, (330,80))
    DISPLAYSURF.blit(text_change_2, (330, 200))


def end_page(who, what):
    if TESTMODE == False:
        text1 = BASICFONT2.render('GAME OVER ', True, (WHITE))
        text2 = BASICFONT3.render('Press the Esc key to quit the game ', True, (WHITE))
        DISPLAYSURF.blit(text1, (80,30))
        DISPLAYSURF.blit(text2, (180,550))
        pygame.time.wait(1000)
    else:
        if who == 'stupids' and what == 'learning':
            text1 = BASICFONT2.render('STUPIDS GAME OVER', True, (WHITE))
            text2 = BASICFONT3.render('Test will start... ', True, (WHITE))
            DISPLAYSURF.blit(text1, (150,30))
            DISPLAYSURF.blit(text2, (300,550))
        elif who == 'elites' and what == 'learning':
            text1 = BASICFONT2.render('ELITES GAME OVER', True, (WHITE))
            text2 = BASICFONT3.render('Test will start... ', True, (WHITE))
            DISPLAYSURF.blit(text1, (200,30))
            DISPLAYSURF.blit(text2, (300,550))
            pygame.time.wait(1000)

        elif who == 'stupids' and what == 'testing':
            text1 = BASICFONT2.render('STUPIDS TEST OVER', True, (WHITE))
            DISPLAYSURF.blit(text1, (150,30))

        elif who == 'elites' and what == 'testing':
            text1 = BASICFONT2.render('ELITES TEST OVER', True, (WHITE))
            DISPLAYSURF.blit(text1, (200,30))
            pygame.time.wait(1000)



def draw_state(players):
    players_step = []
    if len(players) == 0:
        best_step = 0
    else:
        for p in players:
            players_step.append(p.steps)
        best_step = max(players_step)
    FONT= pygame.font.Font('fonts/DungGeunMo.ttf', 20)
    text1 = FONT.render('Best Step : ' + str(best_step), True, (WHITE))
    text2 = FONT.render('Generation : '+ str(GENERATION_NUM), True, (WHITE))
    DISPLAYSURF.blit(text1, (40,30))
    DISPLAYSURF.blit(text2, (40,50))

def draw_leftnum(players, zombies):
    FONT= pygame.font.Font('fonts/DungGeunMo.ttf', 23)
    text_zombies = FONT.render('Zombies: ' + str(len(zombies)), True, (WHITE))
    text_players = FONT.render('Players : ' + str(len(players)), True, (WHITE))
    DISPLAYSURF.blit(text_zombies, (1100,30))
    DISPLAYSURF.blit(text_players, (1100,50))

def draw_time(start_time):
        now_time_min=int((((pygame.time.get_ticks()-start_time)/(1000*60))%60))
        now_time_sec=int(((pygame.time.get_ticks()-start_time)/1000)%60)
        FONT= pygame.font.Font('fonts/DungGeunMo.ttf', 23)
        text_time = FONT.render('Time : ' + str(now_time_min)+':'+str(now_time_sec), True, (WHITE))
        DISPLAYSURF.blit(text_time, (570,30))

def draw_plot(Gen, BS, AS, BF, AF):
    plt.subplot(221)
    plt.plot(Gen, BS)
    plt.xlabel('Generation')
    plt.ylabel('best steps')
    plt.grid()

    plt.subplot(222)
    plt.plot(Gen, AS)
    plt.xlabel('Generation')
    plt.ylabel('average steps')
    plt.grid()

    plt.subplot(223)
    plt.plot(Gen, BF)
    plt.xlabel('Generation')
    plt.ylabel('best fitness')
    plt.grid()

    plt.subplot(224)
    plt.plot(Gen, AF)
    plt.xlabel('Generation')
    plt.ylabel('average fitness')
    plt.grid()

    Rtime = round(int(TTIME.time()),1)
    plt.savefig('%f - Gen-Steps.png'%Rtime)
    plt.show()

def draw_result():
    global stupids_avg, stupids_best, elites_avg, elites_best
    stupids_a = str(stupids_avg)
    stupids_b = str(stupids_best)
    elites_a = str(elites_avg)
    elites_b = str(elites_best)
    t1 = 'stupids_avg: ' + stupids_a + '   elites_avg: ' + elites_a
    t2 = 'stupids_best: ' + stupids_b + '    elites_best: ' + elites_b
    text0 = BASICFONT2.render('See the Result', True, (WHITE))
    text1 = BASICFONT3.render(t1, True, (WHITE))
    text2 = BASICFONT3.render(t2, True, (WHITE))
    DISPLAYSURF.fill(BLACK)
    DISPLAYSURF.blit(text0, (250, 30))
    DISPLAYSURF.blit(text1, (180, 350))
    DISPLAYSURF.blit(text2, (180, 450))


def learning(who):
    global FPSCLOCK, DISPLAYSURF, BASICFONT1, BASICFONT2, BASICFONT3, maps, GENERATION_NUM, TESTMODE, ELITES_TEST, STUPIDS_TEST, GENERATION_MAX

    print(who + ' learning start...')
    print()

    GENERATION_NUM = 1
    # 벽 테두리는 여기서 그립니다!
    maps = []
    maps = Map(MAP_S)
    maps.draw_edge()

    # game.py와 GA.py가 주고받을 플레이어AI 객체 리스트
    child = []
    parent = []

    gen = list(range(1, GENERATION_NUM))
    a_steps = average_steps_list
    b_steps = best_steps_list
    a_fit = average_fitness_list
    b_fit = best_fitness_list

    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                draw_plot(gen, a_steps, b_steps, a_fit, b_fit)
                terminate()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    draw_plot(gen, a_steps, b_steps, a_fit, b_fit)
                    terminate()

    while GENERATION_NUM < GENERATION_MAX:

        candidates = runGame(child, GENERATION_NUM, who, 'learning')

        if ORDERED:
            candidates_index = GA.ordered_selection(candidates, PARENTS_NUM)
        elif ROULETTE:
            candidates_index = GA.roulette_selection(candidates, MAXMIN_RATIO, PARENTS_NUM)
        child = GA.crossover(candidates_index, candidates, CHILDREN_NUM, input, hidden1, hidden2, output)
        child = GA.mutation(child, MUTATION_PROB, mean, stddev)

        print('GENETATION = %d' %GENERATION_NUM)


        GENERATION_NUM += 1


    if who == 'stupids':
        stupids.child = child[:]
        print('stupids learning end')
    if who == 'elites':
        elites.child = child[:]
        print('elites learning end')
    gen = list(range(1, len(a_steps)+1))


    draw_plot(gen, a_steps, b_steps, a_fit, b_fit)
    if TESTMODE == False:
        terminate()


def testing(who):
    print(who, 'testing start...')
    GENERATION_NUM = 1
    generation_n = level_3to4

    # 벽 테두리는 여기서 그립니다!
    maps = []
    maps = Map(MAP_S)
    maps.draw_edge()

    # game.py와 GA.py가 주고받을 플레이어AI 객체 리스트
    child = []
    if who == 'stupids':
        child = stupids.child[:]
    elif who == 'elites':
        child = elites.child[:]

    gen = list(range(1, GENERATION_NUM))
    a_steps = average_steps_list
    b_steps = best_steps_list
    a_fit = average_fitness_list
    b_fit = best_fitness_list

    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                draw_plot(gen, a_steps, b_steps, a_fit, b_fit)
                terminate()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    draw_plot(gen, a_steps, b_steps, a_fit, b_fit)
                    terminate()

    while generation_n < GENERATION_MAX:

        child = runGame(child, GENERATION_NUM, who, 'testing')

        children=[]
        for c in child:
            children.append(c.steps)
        children = np.array(children)
        if who == 'stupids':
            stupids.avg_step_list.append(np.mean(children))
            stupids.best_step_list.append(np.max(children))
        elif who == 'elites':
            elites.avg_step_list.append(np.mean(children))
            elites.best_step_list.append(np.max(children))

        print('GENETATION = %d' %GENERATION_NUM)
        print()

        GENERATION_NUM += 1
        generation_n+=1



    pygame.display.update()
    end_page('elites', 'testing')
    gen = list(range(1, len(a_steps)+1))
    draw_plot(gen, a_steps, b_steps, a_fit, b_fit)

def get_result():
    global stupids, elites, stupids_avg, elites_avg, stupids_best, elites_best

    stupids_avg = 0
    elites_avg = 0
    stupids_best = 0
    elites_best = 0

    for i in range(0, len(stupids.avg_step_list)):
        stupids_avg += stupids.avg_step_list[i]
        elites_avg += elites.avg_step_list[i]
        stupids_best += stupids.best_step_list[i]
        elites_best += elites.best_step_list[i]

    stupids_avg = stupids_avg / len(stupids.avg_step_list)
    elites_avg = elites_avg / len(stupids.avg_step_list)
    stupids_best = stupids_best / len(stupids.avg_step_list)
    elites_best = elites_best / len(stupids.avg_step_list)

    print(stupids_avg)
    print(elites_avg)
    print(stupids_best)
    print(elites_best)


##############################################################


def main():
    global FPSCLOCK, DISPLAYSURF, BASICFONT1, BASICFONT2, BASICFONT3, maps, GENERATION_NUM, TESTMODE, ELITES_TEST, STUPIDS_TEST, GENERATION_MAX, game_end

    if TESTMODE == True:
        if ELITES_TEST == True:
            learning('elites')
            testing('elites')
        if STUPIDS_TEST == True:
            learning('stupids')
            testing('stupids')
        if ELITES_TEST == True and STUPIDS_TEST == True:
            get_result()
            draw_result()
    else:
        learning('elites')

    game_end = True



def runGame(child, GENERATION_NUM, who, what):

    global best_fitness_list, average_fitness_list, best_steps_list, average_steps_list

    # 0. 레벨에 따라 벽좌표를 생성한다
    maps.draw_wall(GENERATION_NUM, level_1to2, level_2to3, level_3to4, TESTMODE, who, what)

    map = np.vstack([maps.wall, maps.edge]) # 테두리 벽과 난이도에 따른 벽 좌표를 가지는 리스트

    # 1. 좀비와 플레이어 이미지를 불러와
    #    zombie_images와 player_images 리스트 안에 넣는다
    zombie_images = []
    player_images = []
    for i in range(0, 8):
        line = []
        for j in range(0, 4):
            line.append(pygame.image.load(
                'images/zombieImg_'+str(i)+'_'+str(j)+'.png'))
        zombie_images.append(line)

    for i in range(0, 8):
        line = []
        for j in range(0, 4):
            line.append(pygame.image.load(
                'images/playerImg_'+str(i)+'_'+str(j)+'.png'))
        player_images.append(line)

    # 3, 4. 객체 생성 및 추가
    # 아이템, 좀비, 플레이어, 죽은 플레이어 객체를 담는 리스트
    items = []
    zombies = []
    players = []
    dead_players = []

    # 플레이어 추가하기
    # 유전알고리즘이 적용된 플레이어AI를 사용하는 경우
    if child:
        for i in child:
            players.append(i)
        if FIXED_LOCATION:
            for i,player in enumerate(players):
                player.generate_player(MAP_S, zombies, players, items, map, player_images, PLAYER_VISION, PLAYER_LIMIT_STEPS, first_location_list[i][0],first_location_list[i][1])
        else :
            for player in players:
                player.generate_player(MAP_S, zombies, players, items, map, player_images, PLAYER_VISION, PLAYER_LIMIT_STEPS)

    #제 1세대
    else:
        for i in range(TOTAL_PNUM):
            p = Player_Ai(input,hidden1,hidden2,output)
            players.append(p)
        for i in range(TOTAL_PNUM):
            players[i].generate_player(MAP_S, zombies, players, items, map, player_images, PLAYER_VISION, PLAYER_LIMIT_STEPS)
            first_location_list.append([players[i].x , players[i].y])
    # 1. 아이템, 좀비, 플레이어 객체를 만들고
    #    객체를 담는 리스트에 넣어준다

    # 아이템 추가하기
    for i in range(ITEM_NUM):
        items.append(Item(map))
    for i in range(ITEM_NUM):
        items[i].generate_item(MAP_S, zombies, players, items)

    # 좀비 추가하기
    for i in range(ZOMBIE_NUM):
        z = Zombie(map, zombie_images, ZOMBIE_VISION, ZOMBIE_DELAY)
        zombies.append(z)

    """
    for i in range(ZOMBIE_NUM):
        zombies[i].generate_zombie(map_size=MAP_S, zombies=zombies, players=players, items=items)
    """


    zombies[0].generate_zombie(map_size=MAP_S, zombies=zombies, players=players, items=items,x=1,y=1)
    zombies[1].generate_zombie(map_size=MAP_S, zombies=zombies, players=players, items=items,x=1,y=17)
    zombies[2].generate_zombie(map_size=MAP_S, zombies=zombies, players=players, items=items,x=1,y=33)
    zombies[3].generate_zombie(map_size=MAP_S, zombies=zombies, players=players, items=items,x=31,y=1)
    zombies[4].generate_zombie(map_size=MAP_S, zombies=zombies, players=players, items=items,x=31,y=33)
    zombies[5].generate_zombie(map_size=MAP_S, zombies=zombies, players=players, items=items,x=62,y=33)
    zombies[6].generate_zombie(map_size=MAP_S, zombies=zombies, players=players, items=items,x=62,y=17)
    zombies[7].generate_zombie(map_size=MAP_S, zombies=zombies, players=players, items=items,x=62,y=1)
    zombies[8].generate_zombie(map_size=MAP_S, zombies=zombies, players=players, items=items,x=31,y=17)


    start_time=pygame.time.get_ticks()

    # 5. 메인 게임 루프를 돈다
    while True:


        # 6. 사용자가 esc를 누르거나 창을 끄면 게임을 종료한다

        gen = list(range(1, GENERATION_NUM))
        a_steps = average_steps_list
        b_steps = best_steps_list
        a_fit = average_fitness_list
        b_fit = best_fitness_list

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                draw_plot(gen,b_steps, a_steps, b_fit, a_fit)
                terminate()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    draw_plot(gen,b_steps, a_steps, b_fit, a_fit)
                    terminate()

        min_item_step = 999999
        for item in items:
            item.steps += 1
            if item.steps < min_item_step:
                min_item_step = item.steps
        if min_item_step > MAX_ITEM_STEP:
            dead_players.extend(players)
            return dead_players

        # 7. 플레이어 객체들의 스텝이 3000을 넘기면
        #    살아있는 플레이어를 죽은 플레이어 리스트로 옮긴다
        natural_dead_index = []
        index = 0
        for player in players:
            if player.steps >= player.limit_of_steps:
                dead_players.append(player)
                natural_dead_index.append(index)
            index += 1

        for i, dead_index in enumerate(natural_dead_index):
            for zombie in zombies:
                if zombie.target_p == players[dead_index - i]:
                    zombie.target_p = None
            del players[dead_index-i]

        wall_dead_index = []
        index = 0
        for p in players:
            for m in map:
                if p.x == m[0] and p.y == m[1]:
                    wall_dead_index.append(index)
                    break
            index += 1

        for i, wall_dead in enumerate(wall_dead_index):
            for zombie in zombies:
                if zombie.target_p == players[wall_dead - i]:
                    zombie.target_p = None
            dead_players.append(players[wall_dead - i])
            del players[wall_dead-i]


        # 8-1. 플레이어 수가 0이 되면 runGame을 종료하고 플레이어AI 객체를 넘긴다
        if not players:
            # 한 세대가 끝나면 next generation이라는 창을 보여주고 다음 세대로 화면전환
            if GENERATION_NUM == GENERATION_MAX-1:
                pygame.display.update()
                DISPLAYSURF.fill(BLACK)
                draw_zombie()
                draw_player()
                end_page(who, what)
                pygame.time.wait(300)
                pygame.display.update()
                return dead_players
            else:
                DISPLAYSURF.fill((176,224,230))
                draw_over()
                pygame.display.update()
                pygame.time.wait(300)

            if GENERATION_NUM == GENERATION_MAX-1:
                pygame.display.update()
                DISPLAYSURF.fill(BLACK)
                draw_zombie()
                draw_player()
                end_page(who, what)
                pygame.display.update()
            else:
                if what == 'learning':
                    if GENERATION_NUM==level_1to2 or GENERATION_NUM==level_2to3 or  GENERATION_NUM==level_3to4:
                        DISPLAYSURF.fill((000,250,154))
                        draw_change()
                        pygame.display.update()
                        pygame.time.wait(30)

            fitlist = []
            steplist = []

            for i in dead_players:
                i.fitness = fitness_func(i)
                fitlist.append(fitness_func(i))

            for j in dead_players:
                steplist.append(j.steps)

            print('average steps: %d' % np.average(steplist))
            print('average fitness: %f' % np.average(fitlist))


            average_fitness_list.append(sum(fitlist) / len(fitlist))
            average_steps_list.append(sum(steplist) / len(steplist))
            best_fitness_list.append(max(fitlist))
            best_steps_list.append(max(steplist))

            for v in range(0, len(average_steps_list)):
                average_steps_list[v] = round(average_steps_list[v], 1)

            for w in range(0, len(average_fitness_list)):
                average_fitness_list[w] = round(average_fitness_list[w], 1)

            print('best_step_list = ', best_steps_list)
            print('best_fitness_list = ', best_fitness_list)
            print('average_step_list =', average_steps_list)
            print('average_fitness_list = ', average_fitness_list)

            return dead_players


        # 8-2. 플레이어가 남아있으면 게임을 진행한다
        else:
            DISPLAYSURF.fill(BLACK)
            draw_state(players)
            draw_leftnum(players, zombies)
            draw_time(start_time)

            # 9. 플레이어와 좀비를 움직인다

            for player in players:
                player.sensor(MAP_S, zombies, items)
                player.p_step(MAP_S, WALL_TOUCH_PENALTY, zombies, players)
                for zombie in zombies:
                    if zombie.target_p == player:
                        player.target_steps += 1
                        break

                """
                player.calc_fitness()
                player.fitness += stepbonus
                """
                #print("player's step is",player.steps)
                #player.dist_comparison(5)
                #print("player's x, y:", player.x, player.y)
            #print()
            for zombie in zombies:
                #print("do move!")
                #print("zombie's x, y:", zombie.x, zombie.y)
                if zombie.steps % zombie.delay_steps == 0:
                    zombie.steps += 1
                    continue
                zombie.player_sensor(players, MAP_S)
                zombie.z_step(MAP_S, zombies, items)
            #print()



            # 10. 플레이어가 아이템을 먹은 경우,
            #     플레이어의 스텝을 증가시키고
            #     아이템의 위치를 바꾼다
            for player in players:
                for item in items:
                    if (player.x == item.x) and (player.y == item.y):
                        item.generate_item(MAP_S, zombies, players, items)
                        #print("get item!")
                        player.num_of_items += 1
                        player.fitness += itembonus
                        player.limit_of_steps += GETITEM


            # 11. 플레이어와 좀비가 부딫힌 경우,
            #     플레이어를 죽이고
            #     좀비를 그 자리에 생성한다


            dead_index = []
            zomb_gen_loc = []
            for zombie in zombies:
                p = 0
                for player in players:
                    if (player.x == zombie.x) and (player.y == zombie.y):
                        #print('collision is happened')
                        #print("p, z:",p ,z)
                        dead_index.append(p)
                        new_loc = [player.x, player.y, p]
                        if new_loc not in zomb_gen_loc:
                            zomb_gen_loc.append(new_loc)
                    p += 1


            for zomb_loc in zomb_gen_loc:
                if len(zombies) < ZOMBIE_NUM_MAX:
                    zom = Zombie(map, zombie_images, ZOMBIE_VISION, ZOMBIE_DELAY)
                    zombies.append(zom)
                    zombies[len(zombies)-1].generate_zombie(MAP_S, zombies, players, items, zomb_loc[0], zomb_loc[1])


            dead_index = set(dead_index)
            dead_index = np.sort(list(dead_index))
            if len(dead_index):
                #print(dead_index)
                for i, index_p in enumerate(dead_index):
                    for zombie in zombies:
                        if zombie.target_p == players[index_p - i]:
                            zombie.target_p = None
                            #print("target reset!")
                    dead_players.append(players[index_p - i])
                    #print("caught!")
                    del players[index_p - i]

        # 12. 플레이어와 좀비, 아이템의 위치와 이미지를 업데이트한다

        for i in range(len(players)):
            players[i].image = players[i].images[players[i].dir][players[i].index]
            players[i].index = (players[i].index+1)%4
            players[i].rect = pygame.Rect(players[i].x*PIXELUNIT, players[i].y*PIXELUNIT, 1, 1)

        for i in range(len(zombies)):
            zombies[i].image = zombies[i].images[zombies[i].dir][zombies[i].index]
            zombies[i].index = (zombies[i].index+1)%4
            zombies[i].rect = pygame.Rect(zombies[i].x*PIXELUNIT, zombies[i].y*PIXELUNIT, 1, 1)

        for i in range(len(items)):
            items[i].rect = pygame.Rect(items[i].x*PIXELUNIT, items[i].y*PIXELUNIT, 1, 1)


        # 13. 벽, 아이템, 플레이어, 좀비를 화면에 그린다
        for i in range(len(maps.edge)):
            DISPLAYSURF.blit(maps.image, (maps.edge[i][0]*PIXELUNIT, maps.edge[i][1]*PIXELUNIT))
        for i in range(len(maps.wall)):
            DISPLAYSURF.blit(maps.image, (maps.wall[i][0]*PIXELUNIT, maps.wall[i][1]*PIXELUNIT))

        for i in range(len(items)):
            DISPLAYSURF.blit(items[i].image, (items[i].x*PIXELUNIT, items[i].y*PIXELUNIT))

        for i in range(len(zombies)):
            DISPLAYSURF.blit(zombies[i].image, (zombies[i].x*PIXELUNIT, zombies[i].y*PIXELUNIT))

        for i in range(len(players)):
            DISPLAYSURF.blit(players[i].image, (players[i].x*PIXELUNIT, players[i].y*PIXELUNIT))


        pygame.display.flip()
        FPSCLOCK.tick(FPS)

def terminate():
    pygame.quit()
    sys.exit()


# 게임 전체 루프

while True:

    if GENERATION_NUM == 1:
        draw_back()
        draw_zombie()
        draw_player()
        button("Game Start!",410,480,500,120,(254,76,64),(157,224,147), main)
        pygame.display.update()
    elif game_end == True:
        break
    else:
        main()
    for event in pygame.event.get():
        pos = pygame.mouse.get_pos()




'''if __name__ == '__main__':
    main()'''
