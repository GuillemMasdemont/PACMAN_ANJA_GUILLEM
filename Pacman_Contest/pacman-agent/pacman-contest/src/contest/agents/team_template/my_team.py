# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import math
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.depth = 4
        self.temperature = 1.0
        self.trajectory = []

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)


    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """

        #Simplification for testing purposes.
        #if self.index == 2: 
        #    return Directions.STOP

        #Deterministic at the beginning. 
        my_pos = game_state.get_agent_position(self.index)
        if my_pos[0] == 1 and Directions.NORTH in game_state.get_legal_actions(self.index):
            return Directions.NORTH
        
        #We implement twisted mini-max + alpha-beta prunning. 
        counter = 0
        def max_agent(state, index, depth, alpha, beta): 
            nonlocal counter

            #print(f'Entered max agent, Index = {index}')

            if depth > self.depth: 
                counter += 1 
                return self.evaluate(state, Directions.STOP), Directions.STOP

            value = -float('inf')
            if state.get_agent_position(index) is None: 
                counter += 1 
                raise ValueError('Error, we always know our index!')
            
            else: 
                new_index = (index + 1) % 4
                
                action_rewards = {}
                for action in state.get_legal_actions(index): 
                    counter += 1 
                    # print('Entered max value')
                    new_value = self.reward(state, index, action) + min_agent(state.generate_successor(index, action), new_index, depth+1, alpha, beta)
                    action_rewards[action] = new_value

                    if value < new_value: 
                        value = new_value 
                        a = action 

                    if value > beta and depth != 0: 
                        return value, a, action_rewards

                    alpha = max(alpha, value)
                
                self.trajectory.append((state, action_rewards))

                return value, a, action_rewards
            

        def min_agent(state, index, depth, alpha, beta): 
            nonlocal counter

            #print(f'Entered min agent, Index = {index}')
            if depth > self.depth: 
                counter += 1 
                return self.evaluate(state, Directions.STOP)

            value = float('inf')
            new_index = (index + 1) % 4
            if state.get_agent_position(index) is None: 
                counter += 1 
                return max_agent(state, new_index, depth+1, alpha, beta)[0]
                
            else: 
                for action in state.get_legal_actions(index): 
                    counter += 1 
                    #print('Entered min agent')
                    new_value = self.reward(state, index, action) + max_agent(state.generate_successor(index, action), new_index, depth+1, alpha, beta)[0]
                    if new_value < value: 
                        value = new_value 

                    if value < alpha: 
                        return value 
                    
                    beta = min(beta, value)

                return value
            
        def softmax_action_choice(action_rewards, temperature=1.0):
            
            if temperature <= 0: temperature = 1e-6
            actions = list(action_rewards.keys())

            if not actions:
                raise ValueError("Error: No actions available")

            vals = [float(action_rewards[a]) for a in actions]
            m = max(vals)

            exps = []
            for v in vals:
                x = (v - m) / temperature
                #x = max(min(x, 700), -700) 
                exps.append(math.exp(x))

            sum_exps = sum(exps)
            if sum_exps == 0:
                probs = [1.0 / len(actions)] * len(actions) #Equiprobable
            else:
                probs = [e / sum_exps for e in exps]

            print('-----')
            print('Actions expanded:', counter)
            print(actions)
            print(probs)
            print('-----')

            chosen_action = random.choices(actions, weights=probs, k=1)[0]
            return chosen_action
        

        _, a, action_rewards = max_agent(game_state, self.index, 0, -float('inf'), float('inf'))
        return softmax_action_choice(action_rewards, self.temperature)
        
        # To do: Can we implement a local minimax (only taking nearby teammates and opponents)
        # Reward function + Value function.  


    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def reward(self, game_state, index, action): 
        return 3

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    def reward(self, game_state, index, action):

        # def is_dead_end(pos):
        #     walls = game_state.get_walls()
        #     x, y = my_pos
        #     open_directions = 0
        #     for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        #         if not walls[int(x + dx)][int(y + dy)]:
        #             open_directions += 1
        #     return open_directions == 1  # Dead end if only one open direction

        successor = game_state.generate_successor(index, action)
        reward = 0

        # Reward for eating or not eating pacman: 
        my_team = self.get_team(game_state)
        opponents = self.get_opponents(game_state)
        food = self.get_food(game_state).as_list()
        my_pos = successor.get_agent_state(self.index).get_position()

        # Case we eat him: 
        if index in my_team:
            for opponent_idx in opponents:
                if game_state.get_agent_state(opponent_idx).is_pacman and not successor.get_agent_state(opponent_idx).is_pacman:
                    reward += 1000 # we have eaten pacman. 
    
        # Case they eat us: 
        elif index in opponents:
            if game_state.get_agent_state(index).is_pacman and not successor.get_agent_state(index).is_pacman:
                reward -= 1000 #we have been eaten. 

        prev_pos = game_state.get_agent_state(self.index).get_position()
        if my_pos == prev_pos:
            reward -= 2  # discourage idling
        else:
            reward += 0.5  # small reward for exploring new tiles

        # # if I have been lingering in same spot, penalize more
        # if len(self.trajectory) >= 4:
        #     last_positions = [state.get_agent_state(self.index).get_position() for state, _ in self.trajectory[-4:]]
        #     if all(self.get_maze_distance(pos, my_pos) <2 for pos in last_positions):
        #         reward -= 500  # penalize lingering

        # reward for eating cookie when ghost is close 
        ate_capsule = False
        capsules = self.get_capsules(game_state)
        for cap_pos in capsules:
            if game_state.get_agent_state(self.index).get_position() == cap_pos:
                ate_capsule = True
                break   
        if ate_capsule:
            reward += 50 

        distances = [self.get_maze_distance(successor.get_agent_state(self.index).get_position(), game_state.get_agent_state(opponent_idx).get_position()) for opponent_idx in self.get_opponents(game_state) if game_state.get_agent_state(opponent_idx).get_position() is not None and game_state.get_agent_state(opponent_idx).is_pacman == False]
        if (game_state.get_agent_state(self.index).is_pacman) and len(distances) > 0:
            min_distance = min(distances)
            if min_distance <= 3:
                reward -= 10 * (4 - min_distance)  # closer ghost penalizes more

        ate_food = False
        for food_pos in food:
            if game_state.get_agent_state(self.index).get_position() == food_pos and not successor.get_agent_state(self.index).get_position() == food_pos:
                ate_food = True
                break   
        if ate_food:
            reward += 50

        # reward for getting closer to food
        if len(food) > 0:
            min_distance_before = min([self.get_maze_distance(prev_pos, food_pos) for food_pos in food])
            min_distance_after = min([self.get_maze_distance(my_pos, food_pos) for food_pos in food])
            if min_distance_after < min_distance_before:
                reward += 10

        # # avoid dead ends when ghosts are nearby
        # ghost_positions = [game_state.get_agent_state(opponent_idx).get_position() for opponent_idx in self.get_opponents(game_state) if game_state.get_agent_state(opponent_idx).get_position() is not None and not game_state.get_agent_state(opponent_idx).is_pacman]
        # for ghost_pos in ghost_positions:
        #     dist_to_ghost = self.get_maze_distance(my_pos, ghost_pos)
        #     if dist_to_ghost <= 3 and is_dead_end(my_pos):
        #         reward -= 15  # penalize dead ends when ghosts are close
    

        distance_to_own_side = 1e-5
        mid_width = game_state.data.layout.width // 2
        distance_to_own_side = abs(my_pos[0] - mid_width)

        #reward for depositing food, especially when closer to own side.
        if game_state.get_agent_state(self.index).is_pacman and (self.red and successor.get_score() > game_state.get_score()) or (not self.red and successor.get_score() > game_state.get_score()):
            multiplier = 200 * 1/(distance_to_own_side + 1)**2
            print('Deposit multiplier:', multiplier)
            score_diff = (successor.get_score() - game_state.get_score())
            if score_diff > 0 and self.red:
                print(multiplier * (score_diff)**2)
                reward += multiplier * (score_diff)**2
            elif score_diff < 0 and not self.red:
                reward += multiplier * (score_diff)**2

        #penalise stopping 
        if action == Directions.STOP:
            reward -= 10

        return reward
    

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):

        weights = {'successor_score': 100, 'distance_to_food': -1}
        my_state = game_state.get_agent_state(self.index)
        if my_state.is_pacman and my_state.num_carrying > 2:
            # prioritize safety
            weights['distance_to_food'] = -0.2
            weights['successor_score'] = 50
        return weights


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def reward(self, game_state, index, action):

        successor = game_state.generate_successor(index, action)
        reward = 0

        # Reward for eating or not eating pacman: 
        my_team = self.get_team(game_state)
        opponents = self.get_opponents(game_state)

        # Case we eat him: 
        if index in my_team:
            for opponent_idx in opponents:
                if game_state.get_agent_state(opponent_idx).is_pacman and not successor.get_agent_state(opponent_idx).is_pacman:
                    reward += 1000 # we have eaten pacman. 
    
        # Case they eat us: 
        elif index in opponents:
            if game_state.get_agent_state(index).is_pacman and not successor.get_agent_state(index).is_pacman:
                reward -= 1000 #we have been eaten. 

        
        # Reward for getting closer to an enemy invader
        for opponent_idx in self.get_opponents(game_state):
            opponent_state = game_state.get_agent_state(opponent_idx)
            
            if opponent_state.is_pacman and opponent_state.get_position() is not None:
                my_pos = game_state.get_agent_state(self.index).get_position()
                successor_pos = successor.get_agent_state(self.index).get_position()
                opponent_pos = opponent_state.get_position()
                dist_before = self.get_maze_distance(my_pos, opponent_pos)
                dist_after = self.get_maze_distance(successor_pos, opponent_pos)
                if dist_after < dist_before:
                    reward += 100  

        # if you cant see an invader, follow disappearing food
        invaders = [game_state.get_agent_state(opponent_idx) for opponent_idx in self.get_opponents(game_state) if game_state.get_agent_state(opponent_idx).is_pacman]
        if len(invaders) == 0:
            food = self.get_food_you_are_defending(game_state).as_list()
            disappearing_food = []
            for food_pos in food:
                if game_state.get_agent_state(self.index).get_position() == food_pos and not successor.get_agent_state(self.index).get_position() == food_pos:
                    disappearing_food.append(food_pos)
            if len(disappearing_food) > 0:
                my_pos = successor.get_agent_state(self.index).get_position()
                dists = [self.get_maze_distance(my_pos, food_pos) for food_pos in disappearing_food]
                if len(dists) > 0:
                    min_dist = min(dists)
                    reward += 10 / (min_dist + 1)
    
        return reward

    def get_features(self, game_state, action):
        features = util.Counter()
        
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()

        center_x = game_state.get_walls().width // 2
        center_y = game_state.get_walls().height // 2
        
        center_pos = (center_x, center_y)

        if game_state.has_wall(center_pos[0], center_pos[1]):
            pass 

        features['distance_to_center'] = self.get_maze_distance(my_pos, center_pos)

        return features

    def get_weights(self, game_state, action):
        return {'distance_to_center': -5}
