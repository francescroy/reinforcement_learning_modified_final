

# This is a toy problem modeled as a Markov decision process.
# Author: Francesc Roy Campderrós

import numpy as np
from random import *
import math
import sys

X_SIZE =7 # 5,7,9
Y_SIZE =7 # 5,7,9
NUM_STATES = X_SIZE * Y_SIZE
GAMMA = 0.90
OPTIMAL_X, OPTIMAL_Y = 3,3 # 2,3,4
OPTIMAL_FINAL_STATE = False # Can be false if using TD-learning or DP methods but must be true if some MonteCarlo method...
COST_STEP = 0.10
NUM_ACTIONS = 5


class ChanceNode:
    def __init__(self, x,y,action):
        self.action = action
        self.x = x
        self.y = y
        self.trans_probabilities = np.zeros((NUM_STATES,), dtype=float)

        if action=='N':

            self.set_trans_probability(x, y + 1, 0.7)
            self.set_trans_probability(x + 1, y, 0.15)
            self.set_trans_probability(x - 1, y, 0.15)
        elif action=='S':

            self.set_trans_probability(x, y - 1, 0.7)
            self.set_trans_probability(x + 1, y, 0.15)
            self.set_trans_probability(x - 1, y, 0.15)
        elif action=='W':

            self.set_trans_probability(x - 1, y, 0.7)
            self.set_trans_probability(x, y + 1, 0.15)
            self.set_trans_probability(x, y - 1, 0.15)
        elif action=='E':

            self.set_trans_probability(x + 1, y, 0.7)
            self.set_trans_probability(x, y + 1, 0.15)
            self.set_trans_probability(x, y - 1, 0.15)
        elif action=='·':
            self.set_trans_probability(x ,y, 0.8)
            self.set_trans_probability(x, y + 1, 0.05)
            self.set_trans_probability(x, y - 1, 0.05)
            self.set_trans_probability(x + 1, y, 0.05)
            self.set_trans_probability(x - 1, y, 0.05)

    def set_trans_probability(self,x,y,prob):

        if(0 <= x and x <= X_SIZE-1 and 0 <= y and y <= Y_SIZE-1):
            self.trans_probabilities[x + y*Y_SIZE] = prob
        else:
            self.trans_probabilities[self.x + self.y*Y_SIZE] += prob

    def possible_next_sates(self,states):

        possible_sates = []

        for x in range(X_SIZE):
            for y in range(Y_SIZE):
                prob_to_that_state = self.trans_probabilities[x + y * Y_SIZE]

                if prob_to_that_state != 0:
                    possible_sates.append([find_state(x, y, states), prob_to_that_state])
                    # print (str(x) + " - " +str(y) + " with prob: "+ str(prob_to_that_state))

        return possible_sates

    def next_state(self, states):

        possible_sates = self.possible_next_sates(states)

        # Com a minim hi haura dos possible_state's no?
        random_int = randint(0, 99)
        definitive_next_state = None

        acumulative=0

        for s in possible_sates:

            acumulative = acumulative + s[1]

            if random_int < acumulative * 100:
                definitive_next_state = s[0]
                break


        return definitive_next_state

class State:
    def __init__(self, x, y, end,cost):
        self.x = x
        self.y = y
        self.end = end
        self.chance_nodes = None
        self.cost = cost

        if end==False:
            self.chance_nodes = [ChanceNode(x,y,'N'),ChanceNode(x,y,'S'),ChanceNode(x,y,'W'),ChanceNode(x,y,'E'),ChanceNode(x,y,'·')]

    def next_state(self, action, states):

        if self.end==True:
            return self # o None?
        if action=='N':
            return self.chance_nodes[0].next_state(states)
        if action=='S':
            return self.chance_nodes[1].next_state(states)
        if action=='W':
            return self.chance_nodes[2].next_state(states)
        if action=='E':
            return self.chance_nodes[3].next_state(states)
        if action=='·':
            return self.chance_nodes[4].next_state(states)

    def get_chance_node(self, action):
        if self.end ==True:
            return None
        if action == 'N':
            return self.chance_nodes[0]
        if action == 'S':
            return self.chance_nodes[1]
        if action == 'W':
            return self.chance_nodes[2]
        if action == 'E':
            return self.chance_nodes[3]
        if action == '·':
            return self.chance_nodes[4]


def find_state(x,y,states):

    result = None

    for s in states:
        if s.x==x and s.y==y:
            result = s

    return result

def compute_cost(x,y):
    return (pow(x - OPTIMAL_X, 2) + pow(y - OPTIMAL_Y, 2))

def compute_reward(x,y,action):

    desired_x =x
    desired_y =y

    if action=='N':
        desired_y = desired_y + 1
    elif action=='S':
        desired_y = desired_y - 1
    elif action == 'W':
        desired_x = desired_x - 1
    elif action == 'E':
        desired_x = desired_x + 1
    elif action == '·':
        pass

    reward = 0.0

    if (0 <= desired_x and desired_x <= X_SIZE - 1 and 0 <= desired_y and desired_y <= Y_SIZE - 1):
        if action!='·':
            reward =  (compute_cost(x,y) - compute_cost(desired_x,desired_y)) - COST_STEP
    else:
        reward = -COST_STEP


    return reward

def get_nice_policy(states):
    policy = [None]*NUM_STATES

    for x in range(0,int((X_SIZE/2))):
        for y in range(Y_SIZE):
            policy[x + y*Y_SIZE] = 'S'

    for x in range(X_SIZE):
        for y in range(0,int((Y_SIZE/2))):
            policy[x + y * Y_SIZE] = 'E'

    for x in range(int((X_SIZE/2))+1,X_SIZE):
        for y in range(Y_SIZE):
            policy[x + y*Y_SIZE] = 'N'

    for x in range(int((X_SIZE/2)),X_SIZE):
        for y in range(int((Y_SIZE/2)+1),Y_SIZE):
            policy[x + y * Y_SIZE] = 'W'

    for x in range(X_SIZE):
        for y in range(Y_SIZE):
            if policy[x + y * Y_SIZE] == None:
                if states[x + y * Y_SIZE].end==False:
                    policy[x + y * Y_SIZE] = '·'



    return policy

def get_random_policy(states):

    policy= []

    for x in range(X_SIZE):
        for y in range(Y_SIZE):

            if(find_state(x,y,states).end==False):


                rand_dir = randint(0, 4)
                if rand_dir==0:
                    policy.append('N')
                if rand_dir==1:
                    policy.append('S')
                if rand_dir==2:
                    policy.append('W')
                if rand_dir==3:
                    policy.append('E')
                if rand_dir==4:
                    policy.append('·')
                



            else:
                policy.append(None)

    return policy

def get_fixed_random_policy(states):
    return ['S', 'N', 'S', '·', '·', 'N', 'S', '·', 'W', 'E', '·', 'S', '·', 'W', 'W', 'S', 'S', 'S', 'E', '·', 'E', 'E', 'W', '·', 'S', 'S', 'W', 'N', 'E', '·', 'N', '·', 'E', 'S', 'W', 'S', '·', 'N', 'W', '·', 'N', 'S', 'S', 'W', '·', 'S', '·', 'S', 'E']

def print_V(V):
    for i in range(X_SIZE):
        for j in range(Y_SIZE):
            print(str(i) + " " + str(j) + ": " + str(V[i + j * Y_SIZE]))

def print_policy(policy):
    print()
    for y in range(Y_SIZE - 1, -1, -1):
        for x in range(X_SIZE):

            if policy[x + y * Y_SIZE] != None:
                print(policy[x + y * Y_SIZE], end=" ")
            else:
                print(" ", end=" ")
        print()
    print()

def get_random_state(states):
    random_state = find_state(randint(0, X_SIZE - 1), randint(0, Y_SIZE - 1), states)
    while random_state.end == True:
        random_state = find_state(randint(0, X_SIZE - 1), randint(0, Y_SIZE - 1), states)
    return random_state

def print_wait_info(loop,number_of_iterations):
    if (loop % (number_of_iterations / 10) == 0 and loop != 0):
        print("|", end=' ')
        sys.stdout.flush()

def get_num_action(action):
    if(action=='N'):
        return 0
    if(action=='S'):
        return 1
    if(action=='W'):
        return 2
    if(action=='E'):
        return 3
    if(action=='·'):
        return 4

def get_action(action_num):
    if(action_num==0):
        return 'N'
    if(action_num==1):
        return 'S'
    if(action_num==2):
        return 'W'
    if(action_num==3):
        return 'E'
    if(action_num==4):
        return '·'

def argmax_a(Q,state):

    best_action = get_action(0)
    best_q = Q[0][state.x + state.y * Y_SIZE]

    if Q[1][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(1)
        best_q = Q[1][state.x + state.y * Y_SIZE]

    if Q[2][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(2)
        best_q = Q[2][state.x + state.y * Y_SIZE]

    if Q[3][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(3)
        best_q = Q[3][state.x + state.y * Y_SIZE]

    if Q[4][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(4)
        best_q = Q[4][state.x + state.y * Y_SIZE]

    return best_action

def max_a(Q,state):

    best_action = get_action(0)
    best_q = Q[0][state.x + state.y * Y_SIZE]

    if Q[1][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(1)
        best_q = Q[1][state.x + state.y * Y_SIZE]

    if Q[2][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(2)
        best_q = Q[2][state.x + state.y * Y_SIZE]

    if Q[3][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(3)
        best_q = Q[3][state.x + state.y * Y_SIZE]

    if Q[4][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(4)
        best_q = Q[4][state.x + state.y * Y_SIZE]

    return best_q




def main():

    states = []

    for x in range(X_SIZE):
        for y in range(Y_SIZE):

            if (x==OPTIMAL_X and y == OPTIMAL_Y):
                states.append(State(x,y,OPTIMAL_FINAL_STATE,compute_cost(x,y)))
            else:
                states.append(State(x, y, False,compute_cost(x,y)))


    #policy_example = get_nice_policy(states) # is simply a list of strings...
    policy_example = get_fixed_random_policy(states)  # is simply a list of strings...

    print_policy(policy_example)

    print("What do you want to do?:")

    option_selected = input()




    # MODEL-FREE MDP ALGO: Q-LEARNING, FIND [ESTIMATED] OPTIMAL POLICY
    if option_selected == "5":

        policy_actual = policy_example

        ALPHA = 0.001  # Which is the right value? After 50% of iteration decay... after 80% decay... LEARNING RATE...
        current_state = get_random_state(states)
        Q = []
        for i in range(NUM_ACTIONS):
            Q.append([0.0] * NUM_STATES)

        EPSILON = 1.00
        number_of_iterations = 5000000
        DECAYING_EPSILON = 1.0/number_of_iterations

        for t in range(number_of_iterations):

            print_wait_info(t, number_of_iterations)

            action = policy_actual[current_state.x + current_state.y * Y_SIZE]
            next_state = current_state.next_state(action, states)
            reward = current_state.cost - next_state.cost

            Q_t_minus_1 = Q[get_num_action(action)][current_state.x + current_state.y * Y_SIZE]

            if t%(number_of_iterations/10)== 0 and t!=0:
                ALPHA= ALPHA/2

            Q[get_num_action(action)][current_state.x + current_state.y *Y_SIZE] = Q_t_minus_1 + ALPHA * (reward + GAMMA * max_a(Q, next_state) - Q_t_minus_1)

            policy_actual[current_state.x + current_state.y * Y_SIZE] = argmax_a(Q, current_state)
            random_int = randint(0, 99)
            if random_int < int(EPSILON * 100):
                policy_actual[current_state.x + current_state.y * Y_SIZE] = get_action(randint(0, 4))

            EPSILON= EPSILON - DECAYING_EPSILON

            #current_state = next_state
            current_state= get_random_state(states)

        print_policy(policy_actual)









    # Suposant que no ens donen les probabilitats, com trobes V de una policy pi?: TD learning... Osigui model free es com RL ja no...? CLAU...

    # After TD learning, Q learning...

    # potser podria usar threads per usar diferents CPU's...
    # mes endavant -> imagina que les transition probabilities cambiessin through time... que en el fons es el que passa al autoscaling problem...

    # l'ultim montecarlo esta bé perque aconsegueix estimator of V que es unbiased!! tot i que es veu que la variance es bastant gran...

    ## THEORY ##
    # It would be nice to undestand why DP policy evaluation algo. works... is because you are using bootstrapping...
    # It's much easier to understand MC policy evaluation algo. works... it relies on sampling, not on bootstrapping...

if __name__ == '__main__':
    main()