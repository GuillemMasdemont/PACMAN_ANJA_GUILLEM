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
                first='DefensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
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

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """

        my_pos = game_state.get_agent_position(self.index)
        #Simplification 
        if self.index == 2: 
            return Directions.STOP

        #Deterministic at the beginning. 
        my_pos = game_state.get_agent_position(self.index)
        print(my_pos[0])
        if my_pos[0] == 1 and Directions.NORTH in game_state.get_legal_actions(self.index):
            return Directions.NORTH


        #We implement twisted mini-max
        counter = 0
        def max_agent(state, index, depth): 
            nonlocal counter
            if depth > self.depth: 
                counter += 1 
                return self.evaluate(state, Directions.STOP), Directions.STOP

            value = -float('inf')
            if state.get_agent_position(index) is None: 
                print(f'If enters this loop is an ERROR!, Index {index}')
                counter += 1 
                return 'ERROR'
            
            else: 
                new_index = (index + 1) % 4
                for action in state.get_legal_actions(index): 
                    counter += 1 
                    # print('Entered max value')
                    new_value = self.reward(state, index, action) + min_agent(state.generate_successor(index, action), new_index, depth+1)
                    if value < new_value: #pacman has to scape!  
                        value = new_value 
                        a = action 
                return value, a
            

        def min_agent(state, index, depth): 
            nonlocal counter
            # print(depth) # This can be noisy, commenting out for now
            if depth > self.depth: 
                counter += 1 
                return self.evaluate(state, Directions.STOP)

            value = float('inf')
            new_index = (index + 1) % 4
            if state.get_agent_position(index) is None: 
                counter += 1 
                return max_agent(state, new_index, depth+1)[0]
                
            else: 
                for action in state.get_legal_actions(index): 
                    counter += 1 
                    #print('Entered min agent')
                    new_value = self.reward(state, index, action) + max_agent(state.generate_successor(index, action), new_index, depth+1)[0]
                    if new_value < value: 
                        value = new_value 

                return value
        
        #print(self.index)
        a = max_agent(game_state, self.index, 0)[1]
        print('Nodes expanded', counter)


        # To do: 
        # Localized minimax + alpha-beta prunning 
        # Softmax for being explorative! 

        return a



        actions = game_state.get_legal_actions(self.index)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)

        if not actions:
            return Directions.STOP
        
        # --- BFS with depth 2 to get rewards for each initial action ---
        action_rewards = []
        for action1 in actions:
            successor1 = self.get_successor(game_state, action1)
            reward1 = self.reward(game_state, action1)
            

            # This will hold the best reward from the second level of actions
            max_reward2 = 0
            actions2 = successor1.get_legal_actions(self.index)
            if Directions.STOP in actions2:
                actions2.remove(Directions.STOP)

            if actions2: # If there are legal moves for the second step
                rewards2 = [self.reward(successor1, action2) for action2 in actions2]
                if rewards2:
                    max_reward2 = max(rewards2)
            
            current_total_reward = reward1 + max_reward2
            action_rewards.append(current_total_reward)

        temperature = 1 
        max_reward = max(action_rewards) #Gergely trick :) 
        exp_rewards = [math.exp(r - max_reward)/temperature for r in action_rewards]
        sum_exp_rewards = sum(exp_rewards)

        if sum_exp_rewards == 0:
            return random.choice(actions)

        probabilities = [r / sum_exp_rewards for r in exp_rewards]
        chosen_action = random.choices(actions, weights=probabilities, k=1)[0]
        
        if self.index == 0: 
            print('move')
            print(actions)
            print(action_rewards)

        return chosen_action

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
        return {'successor_score': 100, 'distance_to_food': -1}


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
