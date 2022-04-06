# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # an increase in state score can also be considered a + point for evaluation score 
        statescorediff = successorGameState.getScore() - currentGameState.getScore()

        # this stops pacman from doing random movements 
        direction = currentGameState.getPacmanState().getDirection()

        # if we get closer to food this is a + point for evaluating score 
        pos = currentGameState.getPacmanPosition()
        curstateclosestfood = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        succsessorfoods = [manhattanDistance(newPos, food) for food in newFood.asList()]
        succsessorclosestfood = 0 if not succsessorfoods else min(succsessorfoods)
        foodgotclose = curstateclosestfood - succsessorclosestfood

        # calculates nearest ghost from this state , this will be required for evaluating 
        # since we don't want to get closer to a ghost 
        nearestghostdist = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        # now we decide evaluating score result according to all factors 

        if nearestghostdist <= 1 or action == Directions.STOP:	#highest priority is closest ghost, because survial is important
            return 0
        if statescorediff > 0:	#moving to a higher state score , if no ghost nearby
            return 8
        elif foodgotclose > 0:	#close food has high priority
            return 4
        elif action == direction:	#to stop pacman doing wierd movements
            return 2
        else:
            return 1


        
        
        #return successorGameState.getScore() + score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        totalghosts = gameState.getNumAgents() - 1

        #maximiser is used by pacman 
        def maximiser(gameState,depth):
            currDepth = depth + 1 # beacause all agents are done we increase depth now 
            if gameState.isWin() or gameState.isLose() or currDepth==self.depth:   # no need to search further 
                return self.evaluationFunction(gameState)
            maxvalue = -999999	#required so that first min gets stored 
            actions = gameState.getLegalActions(0) #all possible legal actions
            for action in actions: # iterate over all actions , to generate states
                successor= gameState.generateSuccessor(0,action) #successor of agent from this index
                maxvalue = max (maxvalue,minimiser(successor,currDepth,1)) #checks for maximum of all minimum utility produced by other agents 
            return maxvalue
        
        #ghosts use minimiser 
        def minimiser(gameState,depth, agentIndex):
            minvalue = 999999	#so that first max gets stored atleast 
            if gameState.isWin() or gameState.isLose():   # no need to search further
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex) #all possible legal actions
            for action in actions:
                successor= gameState.generateSuccessor(agentIndex,action) #successor of agent from this index
                if agentIndex == (gameState.getNumAgents() - 1):	# last agent so we go to pacman after this 
                    minvalue = min (minvalue,maximiser(successor,depth))
                else:
                    minvalue = min(minvalue,minimiser(successor,depth,agentIndex+1))
            return minvalue
        
        #starting with pacman 
        actions = gameState.getLegalActions(0) #all possible legal actions
        currentScore = -999999
        bestaction = ''
        for action in actions:
            nextState = gameState.generateSuccessor(0,action) #successor of agent from this index
            # Next level is a min level. Hence calling min for successors of the root.
            score = minimiser(nextState,0,1)
            # Choosing the action which is Maximum of the successors.
            if score > currentScore:
                bestaction = action
                currentScore = score
        return bestaction
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #maximiser is used by pacman
        def maximiser(gameState,depth,alpha, beta):
            currDepth = depth + 1 # beacause all agents are done we increase depth now 
            if gameState.isWin() or gameState.isLose() or currDepth==self.depth:   #no need to search further
                return self.evaluationFunction(gameState)
            maxvalue = -999999 #required so that first min gets stored
            actions = gameState.getLegalActions(0) #all possible legal actions
            alpha1 = alpha
            for action in actions:
                successor= gameState.generateSuccessor(0,action) #successor of agent from this index
                maxvalue = max (maxvalue,minimiser(successor,currDepth,1,alpha1,beta))
                if maxvalue > beta: # if maxvalue is greater than beta , then no need to search further because the above min layer will return less than equal to beta only 
                    return maxvalue
                alpha1 = max(alpha1,maxvalue)
            return maxvalue
        
        #ghosts use minimiser
        def minimiser(gameState,depth,agentIndex,alpha,beta):
            minvalue = 999999 #so that first max gets stored atleast 
            if gameState.isWin() or gameState.isLose():   #no need to search further
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex) #all possible legal actions
            beta1 = beta
            for action in actions:
                successor= gameState.generateSuccessor(agentIndex,action) #successor of agent from this index
                if agentIndex == (gameState.getNumAgents()-1): # last agent so we go to pacman after this
                    minvalue = min (minvalue,maximiser(successor,depth,alpha,beta1))
                    if minvalue < alpha:
                        return minvalue
                    beta1 = min(beta1,minvalue)
                else:
                    minvalue = min(minvalue,minimiser(successor,depth,agentIndex+1,alpha,beta1))
                    if minvalue < alpha:  # if minvalue less than alpha we don't need to search further because above layer max layer will return greater equal to alpha only
                        return minvalue
                    beta1 = min(beta1,minvalue)
            return minvalue

        # Alpha-Beta Pruning
        actions = gameState.getLegalActions(0) #all possible legal actions
        currentScore = -999999
        bestaction = ''
        alpha = -999999 #relevant initialisation so that logical errors don't happen because of this 
        beta = 999999
        for action in actions:
            nextState = gameState.generateSuccessor(0,action)#successor of agent from this index
            # Next level is a min level thats why finding min for successors of the root.
            score = minimiser(nextState,0,1,alpha,beta)
            # Choosing  Maximum utility of the successors.
            if score > currentScore:
                bestaction = action
                currentScore = score
            # Updating alpha value at root.    
            if score > beta:
                return bestaction
            alpha = max(alpha,score)
        return bestaction		
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #maximiser is used by pacman
        def maximiser(gameState,depth):
            currDepth = depth + 1 # beacause all agents are done we increase depth now 
            if gameState.isWin() or gameState.isLose() or currDepth==self.depth:   #no need to search further
                return self.evaluationFunction(gameState)
            maxvalue = -999999 #required so that first min gets stored
            actions = gameState.getLegalActions(0) #all possible legal actions
            totalmaxvalue = 0
            numberofactions = len(actions)
            for action in actions:
                successor= gameState.generateSuccessor(0,action) #successor of agent from this index
                maxvalue = max (maxvalue,expectLevel(successor,currDepth,1)) #max value from all expected values 
            return maxvalue
        
        #ghosts use minimiser
        def expectLevel(gameState,depth, agentIndex):
            if gameState.isWin() or gameState.isLose():   #no need to search further
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex) #all possible legal actions
            totalexpectedvalue = 0
            numberofactions = len(actions)
            for action in actions:
                successor= gameState.generateSuccessor(agentIndex,action) #successor of agent from this index
                if agentIndex == (gameState.getNumAgents() - 1): # last agent so we go to pacman after this
                    expectedvalue = maximiser(successor,depth)
                else:
                    expectedvalue = expectLevel(successor,depth,agentIndex+1)
                totalexpectedvalue = totalexpectedvalue + expectedvalue # sum all expected values in this branch 
            if numberofactions == 0: #edge case 
                return  0
            return float(totalexpectedvalue)/float(numberofactions) # formula for calculating expected value 
        
        #starting with pacman 
        actions = gameState.getLegalActions(0) #all possible legal actions
        currentScore = -999999
        bestaction = ''
        for action in actions:
            nextState = gameState.generateSuccessor(0,action) #successor of agent from this index
            # Next level is a expect level. Hence calling expectLevel for successors of the root.
            score = expectLevel(nextState,0,1)
            # Choosing the action which is Maximum of the successors.
            if score > currentScore: # stores max 
                bestaction = action
                currentScore = score
        return bestaction

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    location = currentGameState.getPacmanPosition() # location of pacman for manhattan distance 
    foods = currentGameState.getFood().asList()  # food locations 
    nearestfood = min(manhattanDistance(location, food) for food in foods) if foods else 0.5  # returns closest foods distance if foods are there
    score = currentGameState.getScore()  # using this can differentiate some actions 

    '''
      pacman may not move sometimes because we are giving every action same value sometimes , 
      but this will change when ghost gets closer . this doesn't makes game worse because we 
      never die because of this 
    '''
    evaluation = 1.0 / nearestfood + score # closest food distance is in inverse relation, because the closer the better 
    return evaluation
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
