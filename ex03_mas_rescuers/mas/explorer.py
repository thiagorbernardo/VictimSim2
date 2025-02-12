# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

import sys
import os
import random
import math
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
import heapq


class No:
    def __init__(self):
        self.parent_x = 0  # coordenada pai x
        self.parent_y = 0
        self.fn = float("inf")  # custo total g+h
        self.gn = float("inf")  # Custo apartir do nó inicial
        self.hn = 0  # Custo do nó até a vitima


class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0


class Explorer(AbstAgent):
    def __init__(self, env, config_file, resc, id):
        """Construtor do agente random on-line
        @param env: a reference to the environment
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.id = id
        self.walk_stack = Stack()  # a stack to store the movements
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc  # reference to the rescuer agent
        self.x = 0  # current x position relative to the origin 0
        self.y = 0  # current y position relative to the origin 0
        self.map = Map()  # create a map for representing the environment
        self.victims = {}  # a dictionary of found victims: (seq): ((x,y), [<vs>])
        # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals
        self.plan_x = 0
        self.plan_y = 0
        self.plan = []
        self.plan_walk_time = 0
        self.plan_rtime = 0
        self.plan_walk = []

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())

    def get_next_position(self):
        """Randomically, gets the next position that can be explored (no wall and inside the grid)
        There must be at least one CLEAR position in the neighborhood, otherwise it loops forever.
        """
        # Check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()

        # 0: (0, -1),             #  u: Up
        # 1: (1, -1),             # ur: Upper right diagonal
        # 2: (1, 0),              #  r: Right
        # 3: (1, 1),              # dr: Down right diagonal
        # 4: (0, 1),              #  d: Down
        # 5: (-1, 1),             # dl: Down left left diagonal
        # 6: (-1, 0),             #  l: Left
        # 7: (-1, -1)             # ul: Up left diagonal

        # Different exploration patterns based on explorer ID
        if self.id == 1:  # First explorer - prioritize up - left
            # left, upleft, downleft, up, down, right, downright, upright
            position = [6, 7, 5, 0, 4, 2, 3, 1]
        elif self.id == 2:  # Second explorer - prioritize up - right
            # right, upright, up, down, left, downleft, upleft, downright
            position = [2, 1, 0, 4, 6, 5, 7, 3]
        elif self.id == 3:  # Third explorer - prioritize down - left
            # left, downleft, down, up, right, upright, upleft, downright
            position = [6, 5, 4, 0, 2, 1, 7, 3]
        else:  # Fourth explorer - prioritize down - right
            # right, downright, down, up, left, upleft, downleft, upright
            position = [2, 3, 4, 0, 6, 7, 5, 1]

        # Loop until a CLEAR position is found
        i = 0
        while True:
            # Pega coordenadas atual e soma com o caminho a ser andado
            dx, dy = Explorer.AC_INCR[position[i]]
            coord = (self.x + dx, self.y + dy)

            # Se o mapa não foi explorado , e a coordenada não é muro e nem fim
            if (obstacles[position[i]] == VS.CLEAR) and self.map.get(coord) == None:
                coord = (dx, dy)
                return coord

            # se nenhuma coordenada é valida começa a voltar os passos
            if i == 7:
                i = 0
                if self.walk_stack.items == None:
                    continue
                else:
                    self.come_back()
            i += 1

    def explore(self):
        # get an random increment for x and y
        dx, dy = self.get_next_position()
        # print(dx,dy)
        # input()

        # Moves the body to another position
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()

        # Test the result of the walk action
        # Should never bump, but for safe functionning let's test
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add(
                (self.x + dx, self.y + dy),
                VS.OBST_WALL,
                VS.NO_VICTIM,
                self.check_walls_and_lim(),
            )
            # print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

        if result == VS.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            self.walk_stack.push((dx, dy))

            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            # print("coordenadas: ",self.x,self.y)

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                # print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                # print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")

            # Calculates the difficulty of the visited cell
            difficulty = rtime_bef - rtime_aft
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            # print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

        return

    def come_back(self):
        dx, dy = self.walk_stack.pop()
        dx = dx * -1
        dy = dy * -1

        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            print(
                f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}"
            )
            return

        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            # print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")

    def deliberate(self) -> bool:
        """The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        consumed_time = self.TLIM - self.get_rtime()
        if consumed_time < self.get_rtime() * 4:
            self.explore()
            return True

        # time to come back to the base
        if (
            self.walk_stack.is_empty() or (self.x == 0 and self.y == 0)
        ) and not self.plan_walk:
            # time to wake up the rescuer
            # pass the walls and the victims (here, they're empty)
            # print(f"{self.NAME}: rtime {self.get_rtime()}, invoking the rescuer")
            # input(f"{self.NAME}: type [ENTER] to proceed")
            # self.resc.go_save_victims(self.map, self.victims)
            self.resc.sync_explorers(self.map, self.victims)
            return False
        if not self.plan_walk:
            self.plan_rtime = self.get_rtime()
            self.plan_walk = consumed_time
            self.a_star_search((self.x, self.y), (0, 0), (0, 0))
        else:
            self.come_back2()

        return True

    def track(self, no_details, dest):
        row, col = dest
        self.plan_x = row
        self.plan_y = col

        plano = []
        self.plan_walk = []

        # Caminha do destino até a origem
        while not (
            no_details[row][col].parent_i == row
            and no_details[row][col].parent_j == col
        ):
            temp_row = no_details[row][col].parent_i
            temp_col = no_details[row][col].parent_j

            dx = row - temp_row  # Correção para funcionar com negativos e positivos
            dy = col - temp_col  # Correção para funcionar com negativos e positivos

            row, col = temp_row, temp_col
            plano.append((dx, dy, False))  # Adiciona passos normais

        # Ajustamos o último movimento para ser o primeiro real
        if plano:
            last_dx, last_dy, _ = plano[0]
            plano[0] = (last_dx, last_dy, True)  # Marca o último passo como True

        plano.reverse()  # Reverte para ficar na ordem correta
        self.plan_walk.extend(plano)

    def a_star_search(self, src, dest, start):
        # Check if the source and destination are valid

        # Check if we are already at the destination
        if self.is_destination(src[0], src[1], dest):
            print("We are already at the destination")
            return

        # TODO: Não tenho tamannho do mapa para inicializar as listas
        min_x, max_x, min_y, max_y = self.map.get_min_max_map()
        max_y = max_y - min_y + 1
        max_x = max_x - min_x + 1

        # Inicializa lista fechada de nós
        closed_list = [[False for _ in range(max_y)] for _ in range(max_x)]
        # Inicializa os nós do mapa inteiro
        no_details = [[No() for _ in range(max_y)] for _ in range(max_x)]

        # Initialize the start no details
        i = src[0]
        j = src[1]
        no_details[i][j].fn = 0
        no_details[i][j].gn = 0
        no_details[i][j].hn = 0
        no_details[i][j].parent_i = i
        no_details[i][j].parent_j = j

        # Inicializa lista aberta (Nos para ser visitado) com o começo em src
        open_list = []
        heapq.heappush(open_list, (0.0, i, j))

        # Initialize the flag for whether destination is found
        found_dest = False

        # Main loop of A* search algorithm
        while len(open_list) > 0:
            # Remove o elemento com o menor f
            p = heapq.heappop(open_list)

            # Marca na lista fechada que a coordenada (i,j) foi visitada
            i = p[1]
            j = p[2]
            closed_list[i][j] = True

            # Pega direção disponivel
            actions_res = self.map.get((i, j))[2]

            for k, ar in enumerate(actions_res):
                # Se caminho não for livre, apenas pulamos.
                # No mapa não foi implementado VS.UNKS
                if ar != VS.CLEAR:
                    # print(f"{self.NAME} {k} not clear")
                    continue

                dir = Explorer.AC_INCR[k]

                new_i = i + dir[0]
                new_j = j + dir[1]

                # Verifica se o sucessor é valido.
                difficulty = self.map.get((new_i, new_j))
                if difficulty == None or ar != VS.CLEAR:
                    # print("posição mapa NUla")
                    continue

                if self.is_destination(new_i, new_j, dest):
                    # Set the parent of the destination no
                    if dir[0] == 0 or dir[1] == 0:
                        g_new = no_details[i][j].gn + self.COST_LINE * difficulty[0]
                    else:
                        g_new = no_details[i][j].gn + self.COST_DIAG * difficulty[0]

                    no_details[new_i][new_j].parent_i = i
                    no_details[new_i][new_j].parent_j = j
                    # print("The destination no is found")
                    self.plan_rtime -= g_new + self.COST_FIRST_AID
                    self.plan_walk_time += g_new + self.COST_FIRST_AID
                    if self.plan_walk_time > self.plan_rtime * 4 and dest != start:
                        self.plan_rtime += g_new + self.COST_FIRST_AID
                        self.plan_walk_time -= g_new + self.COST_FIRST_AID
                        return
                    self.track(no_details, (new_i, new_j))
                    found_dest = True

                    # print("AQUI\n CUSTOS: ", self.plan_rtime, self.plan_visited)
                    return
                else:

                    # Calculate the new f, g, and h values
                    if dir[0] == 0 or dir[1] == 0:
                        g_new = no_details[i][j].gn + self.COST_LINE * difficulty[0]
                    else:
                        g_new = no_details[i][j].gn + self.COST_DIAG * difficulty[0]

                    h_new = self.calculate_h_value(new_i, new_j, dest)
                    f_new = g_new + h_new

                    # If the no is not in the open list or the new f value is smaller
                    if (
                        no_details[new_i][new_j].fn == float("inf")
                        or no_details[new_i][new_j].fn > f_new
                    ):
                        # Add the no to the open list
                        heapq.heappush(open_list, (f_new, new_i, new_j))
                        # Update the no details
                        no_details[new_i][new_j].fn = f_new
                        no_details[new_i][new_j].gn = g_new
                        no_details[new_i][new_j].hn = h_new
                        no_details[new_i][new_j].parent_i = i
                        no_details[new_i][new_j].parent_j = j

        # If the destination is not found after visiting all nos

        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)

    def is_unblocked(grid, row, col):
        return grid[row][col] == 1

    # Check if a no is the destination
    def is_destination(self, row, col, dest):
        return row == dest[0] and col == dest[1]

    # Calcula o valor de h
    def calculate_h_value(self, row, col, dest):
        return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5

    def come_back2(self):
        dx, dy, _ = self.plan_walk.pop(0)

        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            print(
                f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}"
            )
            return

        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            consumed_time = self.TLIM - self.get_rtime()

            self.x += dx
            self.y += dy
            # print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
