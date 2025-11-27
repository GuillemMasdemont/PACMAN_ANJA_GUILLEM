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
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='SmartReflexAgent', second='SmartReflexAgent', num_training=0):
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
        self.depth = 3

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        """
        # --- Get Opponent Information ---
        opponent_indices = self.get_opponents(game_state)
        for opponent_index in opponent_indices:
            opponent_pos = game_state.get_agent_position(opponent_index)
            if opponent_pos:
                print(f"Agent {self.index}: I can see opponent {opponent_index} at position {opponent_pos} from postions {game_state.get_agent_position(0), game_state.get_agent_position(2)}!")
            opponent_state = game_state.get_agent_state(opponent_index)
            if opponent_state.is_pacman:
                print(f"Agent {self.index}: Opponent {opponent_index} is an invader (Pacman)!")


        # --- Get Noisy Distances to All Agents ---
        noisy_distances = game_state.get_agent_distances()
        print(f"Agent {self.index}: Noisy distances to all agents are: {noisy_distances}")
        """

        # We have information about the agent approximate locations. 
        # We have information about all the food pelets. 
        # We have complete vision in our field. 
        # We can't see the opponents team in their field in more than 5 manhattan distance. 

        # 3 components -> first initial study 
        # adversial search solver (dynamic would be better)
        # reward functions (dynamically (from experience and first initial study.))

        # 

        # We need to implement a selective minimax algorithm. 
            #- When coordinate opponents, when not to. 
        
        # good reward function 



        #Check is we are a pacman. 
        # my_agent_state = game_state.get_agent_state(self.index)
        # We can check whether an agent is pacman or not using agent_state.is_pacman == True. 

        #We know the current score, and the steps left in our pacman game. 
        #s +=1 for each food pellet. 
        #s += 100 for eating a ghost.
        #when we die, the food is spread over all positions. 

        #print(game_state.generate_successor(1, Directions.SOUTH))
        """
        for a1 in actions:   
            successor = self.get_successor(game_state, a1)
            for a2 in successor.get_legal_actions(self.index): 
                print(successor.get_legal_actions(0))
                print(self.evaluate(successor, a2))
        """

        #Observation: include epsilon greedy - to make behaviour slightly different.
        #minimax: divide and conquer? 
        
        counter = 0
        def max_agent(state, index, depth): 
            nonlocal counter
            if depth > self.depth: 
                counter += 1 
                return self.evaluate(state, Directions.STOP), Directions.STOP
            
            value = -10e+10
            if state.get_agent_position(index) is None: 
                print(f'If enters this loop is an ERROR!, Index {index}')
                counter += 1 
                return self.evaluate(state, Directions.STOP), Directions.STOP
            
            else: 
                new_index = (index + 1) % 4
                for action in state.get_legal_actions(index): 
                    counter += 1 
                    new_value = min_agent(state.generate_successor(index, action), new_index, depth+1)
                    if value < new_value and action != Directions.STOP: #pacman has to scape!  
                        value = new_value 
                        a = action 
                return value, a

        def min_agent(state, index, depth): 
            nonlocal counter
            # print(depth) # This can be noisy, commenting out for now
            if depth > self.depth: 
                counter += 1 
                return self.evaluate(state, Directions.STOP)

            value = 10e+10
            new_index = (index + 1) % 4
            if state.get_agent_position(new_index) is None: 
                counter += 1 
                return self.evaluate(state, Directions.STOP)
            
            else: 
                for action in state.get_legal_actions(new_index): 
                    counter += 1 
                    new_value = max_agent(state.generate_successor(new_index, action), new_index, depth+1)[0]
                    if new_value < value: 
                        value = new_value 

                return value
        
        #print(self.index)
        a = max_agent(game_state, self.index, 0)[1]
        print(counter)
        return a

        """

        #Implement as function that prices agents that are separated, or than can cover more terrain.
        
        actions = game_state.get_legal_actions(self.index)
        values = []

        for action in actions:
            successor_state = self.get_successor(game_state, action)
            
            min_value = float('inf')
            
            num_agents = game_state.get_num_agents()
            next_agent_index = (self.index + 1) % num_agents
            
            # We check if the opponent is visible (otherwise we just skip this step). 
            if next_agent_index in self.get_opponents(successor_state) and successor_state.get_agent_position(next_agent_index) is not None:
                opponent_actions = successor_state.get_legal_actions(next_agent_index)
                if not opponent_actions:
                    # Opponent is visible but has no actions, evaluate our move
                    min_value = self.evaluate(game_state, action)
                else:
                    # Find the worst outcome from the opponent's moves
                    for opponent_action in opponent_actions:
                        # Trick for computing the min_value 
                        grand_successor_state = successor_state.generate_successor(next_agent_index, opponent_action)
                        min_value = min(min_value, self.evaluate(grand_successor_state, Directions.STOP))
            else:
                # If next agent is a teammate or an invisible opponent, just evaluate our move
                min_value = self.evaluate(game_state, action)
            values.append(min_value)



        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        """
        
        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

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

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights #util.counter() allows for that .

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


class SmartReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        """
        Calculates features for the agent based on the successor state.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # 1. Number of pellets left on the opponent's side
        food_list = self.get_food(successor).as_list()
        features['pellets_left'] = len(food_list)

        # 2. Distance to the closest food pellet
        if len(food_list) > 0:
            min_dist_pellet = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_closest_pellet'] = min_dist_pellet

        # 3. Pellets the agent is currently carrying
        features['pellets_carrying'] = my_state.num_carrying

        # 4. Distance to the closest visible enemy
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        visible_enemies = [e for e in enemies if e.get_position() is not None]
        if len(visible_enemies) > 0:
            dists = [self.get_maze_distance(my_pos, e.get_position()) for e in visible_enemies]
            features['closest_enemy_dist'] = min(dists)

        # 5. Is the agent a Pacman (on the opponent's side)?
        features['is_pacman'] = 1 if my_state.is_pacman else 0

        # 6. Distance to home territory
        if my_state.is_pacman:
            home_x = game_state.get_walls().width // 2
            if self.red: # We are the red team, home is on the left
                home_x -= 1

            possible_home_positions = [(home_x, y) for y in range(game_state.get_walls().height) if not game_state.has_wall(home_x, y)]
            if possible_home_positions:
                min_dist_home = min([self.get_maze_distance(my_pos, pos) for pos in possible_home_positions])
                features['distance_to_home'] = min_dist_home

        # 7. Did we eat an invader?
        # Compare the number of invaders in the current state vs the successor state.
        current_invaders = [e for e in [game_state.get_agent_state(i) for i in self.get_opponents(game_state)] if e.is_pacman and e.get_position() is not None]
        successor_invaders = [e for e in [successor.get_agent_state(i) for i in self.get_opponents(successor)] if e.is_pacman and e.get_position() is not None]
        if len(successor_invaders) < len(current_invaders):
            features['ate_invader'] = 1

        return features

    def get_weights(self, game_state, action):

        my_state = game_state.get_agent_state(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        
        # Check for visible invaders
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        
        # Check for visible, non-scared ghosts
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None and e.scared_timer < 5]

        # Defensive mode: 
        if len(invaders) > 0:
            print('entered defensive mode')
            return {
                #'is_pacman': -1000,  # discourage_being_pacman
                'ate_invader': 10000, # This is the most important thing to do
                'closest_enemy_dist': -10, # Get closer to the invader
            }

        # --- Return Home Mode ---
        if my_state.num_carrying > 3:
            if ghosts:
                closest_ghost_dist = min([self.get_maze_distance(my_state.get_position(), g.get_position()) for g in ghosts])
                if closest_ghost_dist < 4:
                    return {'distance_to_home': -20, 'closest_enemy_dist': 100}

            return {
                'pellets_carrying': 200, # Value the pellets we have
                'distance_to_home': -20, # Strongly encourage returning home
            }

        # --- Offensive Mode ---
        return {
            'pellets_left': -100,
            'distance_to_closest_pellet': -5,
            'closest_enemy_dist': -10, # Stay away from ghosts while attacking
        }



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
        
        my_pos = successor.get_agent_state(self.index).get_position()
        
        # Compute distance to the nearest food
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance


        # Opponent distance consideration 
        opponents_distances = [10] 
        opponent_indices = self.get_opponents(game_state)
        for opponent_index in opponent_indices:
            opponent_pos = game_state.get_agent_position(opponent_index)
            opponent_state = game_state.get_agent_state(opponent_index)
            if opponent_pos and not opponent_state.is_pacman:
                    opponents_distances.append(self.get_maze_distance(my_pos, opponent_pos))
        
        closest_distance = min(opponents_distances)
        if closest_distance == 0: 
            print('dying?')

        if closest_distance <= 5:
            features['distance_to_opponent'] = closest_distance         

        return features

    def get_weights(self, game_state, action):
        # If we are being chased, prioritize escaping over eating food.
        # A higher weight for distance_to_opponent will make the agent run away.
        return {'successor_score': 100, 'distance_to_food': -1, 'distance_to_opponent': 1000}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
