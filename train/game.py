import sys
import numpy as np
from AIs import manh, rl_reload
import random

def generate_pieces_of_cheese(nb_pieces, width, height, symmetry, player1_location, player2_location, start_random):
    if start_random:
        remaining = nb_pieces + 2
    else:
        remaining = nb_pieces
        player1_location = (0, 0)
        player2_location = (width - 1, height - 1)
    pieces = []
    candidates = []
    considered = []
    if symmetry:
        if nb_pieces % 2 == 1 and (width % 2 == 0 or height % 2 == 0):
            sys.exit("The maze has even width or even height and thus cannot contain an odd number of pieces of cheese if symmetric.")
        if nb_pieces % 2 == 1:
            pieces.append((width // 2, height // 2))
            considered.append((width // 2, height // 2))
            remaining = remaining - 1
    for i in range(width):
        for j in range(height):
            if (not(symmetry) or not((i,j) in considered)) and (i,j) != player1_location and (i,j) != player2_location:
                candidates.append((i,j))
                if symmetry:
                    considered.append((i,j))
                    considered.append((width - 1 - i, height - 1 - j))
    while remaining > 0:
        if len(candidates) == 0:
            sys.exit("Too many pieces of cheese for that dimension of maze")
        chosen = candidates[random.randrange(len(candidates))]
        pieces.append(chosen)
        if symmetry:
            a, b = chosen
            pieces.append((width - a - 1, height - 1 - b))
            symmetric = (width - a - 1, height - 1 - b)
            candidates = [i for i in candidates if i != symmetric]
            remaining = remaining - 1
        candidates = [i for i in candidates if i != chosen]
        remaining = remaining - 1
    if not(start_random):
        pieces.append(player1_location)
        pieces.append(player2_location)
    return pieces[:-2], pieces[-2], pieces[-1]

class PyRat(object):

    def __init__(self, width=21, height=15, round_limit=200, cheeses=40, symmetric=False, start_random=True, opponent=manh):
        self.preprocess = False
        self.symmetric = symmetric
        self.start_random = start_random
        self.height = height
        self.width = width
        self.cheeses = cheeses
        self.piecesOfCheese = list()
        self.round_limit = round_limit
        self.round = 0
        self.score = 0
        self.opponent = opponent
        
        self.map = {}
        for x in range(self.width) :
            for y in range(self.height) :
                self.map[(x, y)] = {}
                if x - 1 >= 0 :
                    self.map[(x, y)][(x - 1, y)] = 1
                if x + 1 < self.width :
                    self.map[(x, y)][(x + 1, y)] = 1
                if y - 1 >= 0 :
                    self.map[(x, y)][(x, y - 1)] = 1
                if y + 1 < self.height :
                    self.map[(x, y)][(x, y + 1)] = 1
        
        self.reset()

    def _update_state(self, action, enemy_action):
        """
        Input: actions and states
        Ouput: new states and reward
        """

        MOVE_DOWN = 'D'
        MOVE_LEFT = 'L'
        MOVE_RIGHT = 'R'
        MOVE_UP = 'U'

        (xx,yy) = self.enemy
        enemy_action_x = 0
        enemy_action_y = 0

        if enemy_action == MOVE_DOWN:
            if yy > 0:
                enemy_action_x = 0
                enemy_action_y = -1
        elif enemy_action == MOVE_UP:
            if yy < self.height - 1:
                enemy_action_x = 0
                enemy_action_y = +1
        elif enemy_action == MOVE_LEFT:
            if xx > 0:
                enemy_action_x = -1
                enemy_action_y = 0
        elif enemy_action == MOVE_RIGHT:
            if xx < self.width - 1:
                enemy_action_x = +1
                enemy_action_y = 0
        else:
            print("FUUUU")
            enemy_action_x = 0
            enemy_action_y = 0
#            raise Exception("INVALID MOVEMENT ENEMY")

        self.enemy = (xx+enemy_action_x,yy+enemy_action_y)

        self.round += 1
        action_x = 0
        action_y = 0
        if action == 0:  # left
            action_x = -1
        elif action == 1:  # right
            action_x = 1
        elif action == 2:  # up
            action_y = 1
        else:
            action_y = -1  # down
        (x,y) = self.player
        new_x = x + action_x 
        new_y = y + action_y 
        self.illegal_move = False
        if new_x < 0 or new_x > self.width-1 or new_y < 0 or new_y > self.height-1:
            new_x = x
            new_y = y
            self.illegal_move = True
        self.player = (new_x, new_y)       
        self._draw_state()       
        
    def _draw_state(self):      
        im_size = (2*self.height-1,2*self.width-1,2)
        self.canvas = np.zeros(im_size)
        (x,y) = self.player
        center_x, center_y = self.width-1, self.height-1
        for (x_cheese,y_cheese) in self.piecesOfCheese:
            self.canvas[y_cheese+center_y-y,x_cheese+center_x-x,0] = 1
        (x_enemy, y_enemy) = self.enemy
        self.canvas[y_enemy+center_y-y,x_enemy+center_x-x,1] = 1
#        self.canvas[center_y,center_x,2] = 1
        return self.canvas
        
    def _get_reward(self):
        (x,y) = self.player
        (xx,yy) = self.enemy
        
        if self.round > self.round_limit:
            return -1
        elif (x,y) in self.piecesOfCheese:
            if (xx,yy) == (x,y):
                self.score += 0.5
                self.enemy_score += 0.5
            else:
                if (xx,yy) in self.piecesOfCheese:
                    self.piecesOfCheese.remove((xx,yy))
                    self.enemy_score += 1.0                    
                self.score += 1
            self.piecesOfCheese.remove((x,y))
            if self.enemy_score == self.score and self.score >= self.cheeses/2:
                return 0
            elif self.enemy_score > self.cheeses/2:
                return -1                
            elif self.score > self.cheeses/2:
                return 1                
            else:
                return 1
        elif (xx,yy) in self.piecesOfCheese:
            self.piecesOfCheese.remove((xx,yy))
            self.enemy_score += 1.0                    
            if self.enemy_score > self.cheeses/2:
                return -1
            else:
                return 0.
        else:
            return 0.
    
    def _is_over(self):
        if self.score > self.cheeses/2 or self.enemy_score > self.cheeses/2 or (
            (self.score == self.enemy_score) and self.score >= self.cheeses/2
            ) or self.round > self.round_limit:
            return True
        else:
            return False

    def observe(self):
        return np.expand_dims(self.canvas, axis=0)


    def act(self, action):
        enemy_action = self.opponent.turn(self.map,self.width,self.height,self.enemy,self.player,self.enemy_score,self.score,self.piecesOfCheese,3000)
        self._update_state(action,enemy_action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self):
        self.piecesOfCheese, self.player, self.enemy = generate_pieces_of_cheese(
            self.cheeses, self.width, self.height, self.symmetric, 
            (-1,-1), (-1,-1), self.start_random)
        self.round = 0
        self.illegal_move = False
        self.score = 0
        self.enemy_score = 0
        self._draw_state()
        if not self.preprocess:
            self.opponent.preprocessing(self.map,self.width,self.height,self.enemy,self.player,self.piecesOfCheese,30000)
            self.preprocess = True
