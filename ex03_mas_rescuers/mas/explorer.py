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
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.id = id
        self.walk_stack = Stack()  # a stack to store the movements
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals
        

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())

    def get_next_position(self):
        """ Randomically, gets the next position that can be explored (no wall and inside the grid)
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
        if self.id == 1: # First explorer - prioritize up - left
            # left, upleft, downleft, up, down, right, downright, upright
            position = [6, 7, 5, 0, 4, 2, 3, 1]
        elif self.id == 2: # Second explorer - prioritize up - right
            # right, upright, up, down, left, downleft, upleft, downright
            position = [2, 1, 0, 4, 6, 5, 7, 3]
        elif self.id == 3: # Third explorer - prioritize down - left
            # left, downleft, down, up, right, upright, upleft, downright
            position = [6, 5, 4, 0, 2, 1, 7, 3]
        else: # Fourth explorer - prioritize down - right
            # right, downright, down, up, left, upleft, downleft, upright
            position = [2, 3, 4, 0, 6, 7, 5, 1]

        # Loop until a CLEAR position is found
        i = 0
        while True:
            # Pega coordenadas atual e soma com o caminho a ser andado
            dx, dy = Explorer.AC_INCR[position[i]]
            coord = (self.x+dx, self.y+dy)
            
            # Se o mapa não foi explorado , e a coordenada não é muro e nem fim
            if (obstacles[position[i]] == VS.CLEAR) and self.map.get(coord) == None:
                coord = (dx, dy)
                return coord
            
            #se nenhuma coordenada é valida começa a voltar os passos
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
        #input()

        # Moves the body to another position
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()

        # Test the result of the walk action
        # Should never bump, but for safe functionning let's test
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            # print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

        if result == VS.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            self.walk_stack.push((dx, dy))
            

            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy     
            #print("coordenadas: ",self.x,self.y)     

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                # print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                #print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")
            
            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            #print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

        return

    def come_back(self):
        dx, dy = self.walk_stack.pop()
        dx = dx * -1
        dy = dy * -1

        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
        
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            #print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
        
    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        consumed_time = self.TLIM - self.get_rtime()
        if consumed_time < self.get_rtime():
            self.explore()
            return True

        # time to come back to the base
        if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
            # time to wake up the rescuer
            # pass the walls and the victims (here, they're empty)
            # print(f"{self.NAME}: rtime {self.get_rtime()}, invoking the rescuer")
            #input(f"{self.NAME}: type [ENTER] to proceed")
            # self.resc.go_save_victims(self.map, self.victims)
            self.resc.sync_explorers(self.map, self.victims)
            return False

        self.come_back()
        return True


