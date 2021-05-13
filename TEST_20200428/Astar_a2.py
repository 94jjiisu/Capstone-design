import numpy as np
import math
from heapq import *

# 휴리스틱 함수로 체비셰프거리 이용
def Heuristic(x, y):
    D1 = 1
    D2 = 1
    dx = abs(x[0] - y[0])
    dy = abs(x[1] - y[1])
    return D1 * (dx + dy) + (D2 - 2 * D1) * min(dx, dy)

# start = zombie_p -> 좀비의 현재 위치 / goal = player_p -> 플레이어의 위치
def Astar(map_size, imp_points ,zombie_p, player_p):

    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

    close_ = set()      # 탐색이 끝난 좌표
    open_ = {}          # 탐색 대상인 좌표
    g = {zombie_p:0}    # 초기 좀비좌표와 이동한 좀비좌표의 거리 비용
    f = {zombie_p:Heuristic(zombie_p, player_p)}  # f = g와 heuristic의 합
    heap = []

    heappush(heap, (f[zombie_p], zombie_p))

    while heap:

        # heap에서 f값이 최소인 좌표값 가져오기
        current = heappop(heap)[1]

        # 목표점에 도달(player위치와 같아짐)
        if current == player_p:
            path = []
            while current in open_:
                path.append(current)
                current = open_[current]
            path = path[::-1]

            #출력부분
            start_pos = 0
            end_pos = len(path)
            div = 10
            #print("<좀비의 이동경로>")
            for idx in range(start_pos, end_pos+div, div):
                out = path[start_pos:start_pos+div]
                #if out != []:
                #    print("ZOMBIE PATH", out)
                start_pos = start_pos+div
            if len(path) == 0:
                return []
            zombie_next = path[0]
            return zombie_next
            #print("\n<좀비의 다음경로> : " , zombie_next)
            
        close_.add(current)

        # 인접한 8개의 좌표 탐색
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            neighbor_list = list(neighbor)
            
            # 대각선 방향인 경우와 직선 방향인 경우의 비용을 다르게 설정
            if (abs(i) + abs(j)) == 2:
                tent_g = g[current] + 1.4
            else :
                tent_g = g[current] + 1           

            # map 크기 벗어나지 않는 조건 & 벽에 닿지 않을 조건
            if 0 <= neighbor[0] < map_size[0]:
                if 0 <= neighbor[1] < map_size[1]:
                    if neighbor_list in imp_points:
                        continue
                else:
                    continue
            else:
                continue

            # 이미 탐색을 한 좌표 & tent_g값이 기존의 g값보다 더 큰 경우 -> 탐색하지 않음
            if neighbor in close_ and tent_g >= g.get(neighbor, 0):
                continue

            # 비용이 제일 작은 값으로 갱신
            if  tent_g < g.get(neighbor, 0) or neighbor not in [i[1]for i in heap]:
                open_[neighbor] = current
                g[neighbor] = tent_g
                f[neighbor] = tent_g + Heuristic(neighbor, player_p)
                heappush(heap, (f[neighbor], neighbor))

    return False
