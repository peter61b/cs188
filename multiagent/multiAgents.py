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
        
        foodList = newFood.asList()
        closest = None
        currPos = currentGameState.getPacmanPosition()
        eval = successorGameState.getScore()

        #get nearest food
        
        if foodList:
            for loc in foodList:
                if closest == None or manhattanDistance(newPos,loc) < manhattanDistance(newPos, closest):
                    closest = loc
            eval += 1/ manhattanDistance(newPos, closest) 
                
        

        #Now see if newPos is straying further from closest
        
        #Higher progress = closer to nearest
        
        nextScore = successorGameState.getScore()
        
        
        
        for pos,time in zip(newGhostStates,newScaredTimes):
            dist = manhattanDistance(newPos, pos.getPosition())
            if time == 0:
                if dist < 2:
                    eval -= 10
            elif time > 0:
                eval += 10/dist

         
        
        if action == Directions.STOP:
            eval -=15
        
        "*** YOUR CODE HERE ***"
        
        #distance to ghost:
        
        
        
        return eval

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

        def minimax(gameState: GameState,action,agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0:
                return maximizer(agentIndex, action, gameState, depth)
            else:
                return minimizer(agentIndex, action, gameState, depth)

        def minimizer(agentIndex,action,gameState,depth):
            score = float('inf')
            
            for action in gameState.getLegalActions(agentIndex):
                v = gameState.generateSuccessor(agentIndex, action)
                nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
                a = minimax(v, action,nextAgentIndex, depth + 1 if nextAgentIndex == 0 else depth) 
                score = min(score,a)
            return score
        
        def maximizer(agentIndex, action,gameState,depth):
            score = float('-inf')
            
            for action in gameState.getLegalActions(agentIndex):
                v = gameState.generateSuccessor(agentIndex, action)
                a = minimax(v, action,(agentIndex+1) % gameState.getNumAgents(), depth)
                score = max(score,a)
            return score
        #Returns whether or not the game state is a losing state
        
        
        score, bestMove = float('-inf'), None
        for action in gameState.getLegalActions(0):
            v = gameState.generateSuccessor(0, action)
            x = minimax(v, action, 1, 0)
            if x > score:
                score = x
                bestMove = action
        return bestMove

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def minimizer(gameState, alpha, beta, agentIndex, depth):
            v = float('inf')
            nextIndex = (agentIndex + 1) % gameState.getNumAgents()
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                v = min(v, value(successor, alpha,beta, nextIndex, depth + 1 if nextIndex==0 else depth ))
                if v < alpha:
                    return v
                else:
                    beta = min(v, beta)
            return v
                    
        def maximizer(gameState, alpha, beta, depth):
            v = float('-inf')
            
            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                v = max(v, value(successor, alpha, beta, 1, depth))
                
                if v > beta: 
                    return v
                
                else: 
                    alpha = max(v, alpha)
            return v
        
        def value(gameState, alpha, beta, agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:  
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return maximizer(gameState, alpha,beta, depth)
            else: 
                return minimizer(gameState, alpha, beta, agentIndex, depth)
            
        
        alpha, bestMove = float('-inf'), None
        
        for action in gameState.getLegalActions(0):
            v = gameState.generateSuccessor(0, action)
            x = value(v, alpha, float('inf'), 1, 0)
            if x > alpha:
                alpha = x
                bestMove = action
        return bestMove

        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState: GameState):        
        
        def expectimax(gameState: GameState,action,agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0:
                return maximizer(agentIndex, action, gameState, depth)
            else:
                return expected(agentIndex, action, gameState, depth)

        def expected(agentIndex,action,gameState,depth):
            EV = 0
            actions = gameState.getLegalActions(agentIndex)
            
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
                EV += expectimax(successor, action,nextAgentIndex, depth + 1 if nextAgentIndex == 0 else depth) 
                
            return EV / len(actions)
        
        def maximizer(agentIndex, action,gameState,depth):
            score = float('-inf')
            
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                a = expectimax(successor, action,(agentIndex+1) % gameState.getNumAgents(), depth)
                score = max(score,a)
            return score
        #Returns whether or not the game state is a losing state
        
        
        score, bestMove = float('-inf'), None
        for action in gameState.getLegalActions(0):
            v = gameState.generateSuccessor(0, action)
            x = expectimax(v, action, 1, 0)
            if x > score:
                score = x
                bestMove = action
        return bestMove
    


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Pacman is munching on all his opps and making them bite curb and crash tf out, getting all the pellets, and scoremaxxing.
    """
    evalScore = 0
    
    position = currentGameState.getPacmanPosition()
    
    food = currentGameState.getFood()
    
    ghost_states = currentGameState.getGhostStates()
    
    scared_times = [ghostState.scaredTimer for ghostState in ghost_states]
    
    
   
   #account for division by 0
    gScore = 11 / max(1, sum([manhattanDistance(position, ghost.getPosition()) for ghost in ghost_states]))
    # If the ghosts are scared score is increased by g
    if scared_times == [0]:
        gScore *= -1
    # nearest food reciprocal
    fScore = 11 / (min([manhattanDistance(position, food) for food in food.asList()] + [200]) +1.1)
    # The evaluation score

    eval = (currentGameState.getScore() + gScore + fScore)
    
    return eval




# Abbreviation
better = betterEvaluationFunction

