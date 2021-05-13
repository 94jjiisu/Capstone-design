import random
import numpy as np
import pygame
from pygame.locals import *


class Map(pygame.sprite.Sprite):
    def __init__(self, map_size=None):
        pygame.sprite.Sprite.__init__(self)
        self.edge = np.array([[0, 0]])
        self.wall = np.array([[0, 0]])
        self.map_size = map_size
        self.image = pygame.image.load('images/wallImg.png')
        self.rect = pygame.Rect(0, 0, 1, 1)

    """
    맵 테두리에 벽을 만드는 함수
    width : 맵의 가로
    height : 맵의 높이
    """

    def edge_generator(self, width, height):
        self.edge = np.array([])

        width_num = np.arange(width)
        height_num = np.arange(height)
        height_num_minus1 = np.arange(height-1)

        U = np.zeros(width)
        U = np.dstack([width_num, U]).reshape(width, 2)

        D = np.full(width, height-1)
        D = np.dstack([width_num, D]).reshape(width, 2)

        L = np.zeros(height-1)
        L = np.dstack([L, height_num_minus1]).reshape(height-1, 2)

        R = np.full(height-1, width-1)
        R = np.dstack([R, height_num_minus1]).reshape(height-1, 2)

        self.edge = np.vstack([U, D, L, R])


    def straight_wall(self, dir, start, end):

        if dir == 'U':
            a = np.full(start[1]-end[1], start[0])
            b = np.arange(start[1]-end[1]) + end[1]
            ab = np.dstack([a, b]).reshape(start[1]-end[1], 2)
            self.wall = np.vstack([self.wall, ab])

        elif dir == 'D':
            a = np.full(end[1]-start[1], end[0])
            b = np.arange(end[1]-start[1])+start[1]+1
            ab = np.dstack([a, b]).reshape(end[1]-start[1], 2)
            self.wall = np.vstack([self.wall, ab])


        elif dir == 'R':

            a = np.arange(end[0]-start[0])+start[0]
            b = np.full(end[0]-start[0], start[1])
            ab = np.dstack([a, b]).reshape(end[0]-start[0], 2)
            self.wall = np.vstack([self.wall, ab])

        elif dir == 'L':

            a = np.arange(end[0]-start[0]+1)+start[0]
            b = np.full(end[0]-start[0]+1, end[1])
            ab = np.dstack([a, b]).reshape(end[0]-start[0]+1, 2)
            self.wall = np.vstack([self.wall, ab])


    """
    L자형 벽을 만드는 함수
    지정한 방향순서대로 뻗어나가면서 벽을 만든다
    """

    def L_shape_wall(self, dir, start, vertex, end):
        wall = []

        if dir == "UL":
            self.straight_wall("U", start, vertex)
            self.straight_wall("L", vertex, end)

        elif dir == "UR":
            self.straight_wall("U", start, vertex)
            self.straight_wall("R", vertex, end)

        elif dir == "DL":
            self.straight_wall("D", start, vertex)
            self.straight_wall("L", vertex, end)

        elif dir == "DR":
            self.straight_wall("D", start, vertex)
            self.straight_wall("R", vertex, end)

        elif dir == "LU":
            self.straight_wall("L", start, vertex)
            self.straight_wall("U", vertex, end)

        elif dir == "RU":
            self.straight_wall("R", vertex, start)
            self.straight_wall("U", vertex, end)

        elif dir == "LD":
            self.straight_wall("L", start, vertex)
            self.straight_wall("D", vertex, end)

        elif dir == "RD":
            self.straight_wall("R", vertex, start)
            self.straight_wall("D", vertex, end)



    """
    U자형 벽을 만드는 함수
    지정한 방향순서대로 뻗어나가면서 벽을 만든다
    """

    def U_shape_wall(self, dir, start, vertex_1, vertex_2, end):

        if dir == "ULD":
            self.straight_wall("U", start, vertex_1)
            self.straight_wall("L", [vertex_1[0]-1, vertex_1[1]], vertex_2)
            self.straight_wall("D", [vertex_2[0], vertex_2[1]-1], end)

        elif dir == "URD":
            self.straight_wall("U", [start[0], start[1]+1], vertex_1)
            self.straight_wall("R", vertex_1, vertex_2)
            self.straight_wall("D", [vertex_2[0], vertex_2[1]-1], end)

        elif dir == "DLU":
            self.straight_wall("D", vertex_1, start)
            self.straight_wall("L", vertex_2, vertex_1)
            self.straight_wall("U", end, [vertex_2[0], vertex_2[1]+1])

        elif dir == "DRU":
            self.straight_wall("D", [start[0], start[1]-1], vertex_1)
            self.straight_wall("R", vertex_1, vertex_2)
            self.straight_wall("U", [vertex_2[0], vertex_2[1]+1], end)

        elif dir == "LUR":
            self.straight_wall("L", start, vertex_1)
            self.straight_wall("U", [vertex_1[0], vertex_1[1]+1], vertex_2)
            self.straight_wall("R", end, [vertex_2[0]+1, vertex_2[1]])

        elif dir == "RUL":
            self.straight_wall("R", start, vertex_1)
            self.straight_wall("U", [vertex_1[0], vertex_1[1]+1], vertex_2)
            self.straight_wall("L", [vertex_2[0]-1, vertex_2[1]], end)

        elif dir == "LDR":
            self.straight_wall("L", vertex_1, start)
            self.straight_wall("D", vertex_1, vertex_2)
            self.straight_wall("R", vertex_2, [end[0]+1, end[1]])

        elif dir == "RDL":
            self.straight_wall("R", start, vertex_1)
            self.straight_wall("D", [vertex_1[0], vertex_1[1]-1], vertex_2)
            self.straight_wall("L", end, [vertex_2[0]-1, vertex_2[1]])


    def draw_edge(self):
        self.edge_generator(self.map_size[0], self.map_size[1])

    def draw_wall(self, generationNum, to2, to3, to4, TESTMODE, who, what):
        if what == 'testing':
            self.wall = np.array([[0, 0]])
            self.L_shape_wall("LU", [5, 13], [10, 13], [10, 7])
            self.straight_wall("L", [8, 18], [14, 18])
            self.U_shape_wall("DRU", [6, 26], [6, 30], [13, 30], [13, 26])

            self.L_shape_wall("DR", [17, 5], [17, 10], [23, 10])
            self.U_shape_wall("LDR", [25, 15], [20, 15], [20, 20], [25, 20])
            self.L_shape_wall("LD", [22, 25], [28, 25], [28, 30])
            self.straight_wall("U", [27, 10], [27, 5])

            self.L_shape_wall("LU", [35, 13], [40, 13], [40, 7])
            self.straight_wall("L", [38, 18], [44, 18])
            self.U_shape_wall("DRU", [36, 26], [36, 30], [43, 30], [43, 26])

            self.L_shape_wall("DR", [47, 5], [47, 10], [53, 10])
            self.U_shape_wall("LDR", [55, 15], [50, 15], [50, 20], [55, 20])
            self.L_shape_wall("LD", [52, 25], [58, 25], [58, 30])
            self.straight_wall("U", [57, 10], [57, 5])


        elif who == 'stupids' and what == 'learning':
            self.wall = np.array([[0, 0]])
            self.L_shape_wall("DR", [5, 4], [5, 9], [10, 9])
            self.U_shape_wall("RDL", [16, 4], [20, 4], [20, 9], [16, 9])

            self.straight_wall("L", [10, 14], [17, 14])
            self.L_shape_wall("UR", [24, 13], [24, 7], [28, 7])

            self.U_shape_wall("DRU", [5, 18], [5, 23], [10, 23], [10, 18])
            self.straight_wall("L", [20, 20], [27, 20])
            self.straight_wall("U", [27, 20], [27, 15])

            self.straight_wall("U", [8, 32], [8, 27])
            self.straight_wall("L", [8, 27], [14, 27])
            self.U_shape_wall("URD", [19, 30], [19, 24], [24, 24], [24, 30])


            self.L_shape_wall("DR", [35, 4], [35, 9], [40, 9])
            self.U_shape_wall("RDL", [46, 4], [50, 4], [50, 9], [46, 9])

            self.straight_wall("L", [40, 14], [47, 14])
            self.L_shape_wall("UR", [56, 13], [56, 7], [60, 7])

            self.U_shape_wall("DRU", [35, 18], [35, 23], [40, 23], [40, 18])
            self.straight_wall("L", [50, 20], [57, 20])
            self.straight_wall("U", [57, 20], [57, 15])

            self.straight_wall("U", [38, 32], [38, 27])
            self.straight_wall("L", [38, 27], [44, 27])
            self.U_shape_wall("URD", [49, 30], [49, 24], [54, 24], [54, 30])



        elif generationNum > to2 and generationNum <= to3:
            self.wall = np.array([[0, 0]])
            self.straight_wall("L", [6, 6], [12,6])
            self.straight_wall("L", [14, 12], [20, 12])
            self.straight_wall("L", [21, 18], [26, 18])
            self.straight_wall("L", [14, 23], [20, 23])
            self.straight_wall("L", [6, 29], [12, 29])

            self.straight_wall("U", [5, 21], [5, 12])
            self.straight_wall("U", [27, 10], [27, 5])
            self.straight_wall("U", [27, 29], [27, 25])

            self.straight_wall("L", [36, 6], [42,6])
            self.straight_wall("L", [44, 12], [50, 12])
            self.straight_wall("L", [51, 18], [56, 18])
            self.straight_wall("L", [44, 23], [50, 23])
            self.straight_wall("L", [36, 29], [42, 29])

            self.straight_wall("U", [35, 21], [35, 12])
            self.straight_wall("U", [57, 10], [57, 5])
            self.straight_wall("U", [57, 29], [57, 25])


            #self.straight_wall("R", [35, 9], [59, 9])
            #self.straight_wall("U", [54, 20], [54, 12])
            #self.straight_wall("U", [36, 30], [36, 18])

        elif generationNum > to3 and generationNum <= to4:
            self.wall = np.array([[0, 0]])

            self.L_shape_wall("LD", [9, 5], [14, 5], [14, 9])
            self.L_shape_wall("LU", [5, 16], [10, 16], [10, 12])
            self.L_shape_wall("LD", [11, 20], [16, 20], [16, 25])
            self.L_shape_wall("DL", [6, 25], [6, 29], [10, 30])

            self.L_shape_wall("DR", [20, 7], [20, 12], [26, 12])
            self.L_shape_wall("UL", [23, 21], [23, 17], [27, 17])
            self.L_shape_wall("LU", [22, 29], [27, 29], [27, 25])

            self.L_shape_wall("LD", [39, 5], [44, 5], [44, 9])
            self.L_shape_wall("LU", [35, 16], [40, 16], [40, 12])
            self.L_shape_wall("LD", [41, 20], [46, 20], [46, 25])
            self.L_shape_wall("DL", [36, 25], [36, 29], [40, 30])

            self.L_shape_wall("DR", [50, 7], [50, 12], [56, 12])
            self.L_shape_wall("UL", [53, 21], [53, 17], [57, 17])
            self.L_shape_wall("LU", [52, 29], [57, 29], [57, 25])

        elif generationNum > to4:
            self.wall = np.array([[0, 0]])

            self.U_shape_wall("RDL", [7, 5], [12, 5], [12, 13], [7, 13])
            self.U_shape_wall("URD", [6, 21], [6, 17], [13, 17], [13, 21])
            self.U_shape_wall("DRU", [6, 26], [6, 30], [13, 30], [13, 26])

            self.U_shape_wall("DRU", [20, 5], [20, 11], [27, 11], [27, 5])

            self.U_shape_wall("LDR", [26, 16], [20, 16], [20, 22], [26, 22])
            self.U_shape_wall("LDR", [26, 25], [20, 25], [20, 31], [26, 31])


            self.U_shape_wall("RDL", [37, 5], [42, 5], [42, 13], [37, 13])
            self.U_shape_wall("URD", [36, 21], [36, 17], [43, 17], [43, 21])
            self.U_shape_wall("DRU", [36, 26], [36, 30], [43, 30], [43, 26])

            self.U_shape_wall("DRU", [50, 5], [50, 11], [57, 11], [57, 5])

            self.U_shape_wall("LDR", [56, 16], [50, 16], [50, 22], [56, 22])
            self.U_shape_wall("LDR", [56, 25], [50, 25], [50, 31], [56, 31])


            #self.U_shape_wall("LUR", [6, 14], [30, 14], [30, 9], [15, 9])
            #self.U_shape_wall("DLU", [25, 29], [25, 19], [14, 19], [14, 24])
            #self.U_shape_wall("URD", [40, 10], [40, 18], [50, 18], [50, 10])
            #self.U_shape_wall("LDR", [59, 24], [30, 24], [30, 29], [49, 29])
