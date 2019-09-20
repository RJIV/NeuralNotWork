# baselineTeam.py
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    z = zip(actions, values)
    # print(z)
    # print('maxValue', maxValue)
    # print(values)
    # print(actions)
    # print(bestActions)

    foodLeft = len(self.getFood(gameState).asList())

    # print(gameState.getCapsules())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    # print(min(bestActions))
    # print(maxValue)
    
    # return bestActions[bestActions.index(min(bestActions))]
    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    # if features['onOffense'] == 1:
    #   print("Action", action, "Features: ", features)
    #   print("Action", action, "Weights: ", weights)
    #   print("Action", action, "F*W: ", features*weights)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()  
    
    
    # Noteworthy data!
    myPos = successor.getAgentState(self.index).getPosition()
    isRed = gameState.isOnRedTeam(self.index)
    iAmPacman = successor.getAgentState(self.index).isPacman
    # print("Pacman:", iAmPacman)
    # Getting defenders location..
    teamIndicies = gameState.getRedTeamIndices() if isRed else gameState.getBlueTeamIndices()
    defenderIndex = teamIndicies[teamIndicies.index(self.index) - 1]
    defenderDistance = self.getMazeDistance(myPos, successor.getAgentState(defenderIndex).getPosition())
    # Map size
    # attrs = vars(gameState.data.layout)
    # print("Attributes", attrs)
    mapDimensions = (gameState.data.layout.width, gameState.data.layout.height)
    safePoint = (mapDimensions[0] / 2, mapDimensions[1] / 2) 
    distanceFromSafePoint = self.getMazeDistance(myPos, safePoint)

    features['successorScore'] = -len(foodList)
    features['onOffense'] = 1
    features['numCarrying'] = gameState.getAgentState(self.index).numCarrying

    # Incentive for a ghost to become pacman
    if not iAmPacman:
        features['risk'] = 0
    else:
        # print("Am I in risk?")
        features['risk'] = distanceFromSafePoint  * features['numCarrying']
        # features['risk'] = defenderDistance     
        # features['risk'] = max(0, features['numCarrying'] * (defenderDistance - 3))

    # Distance to capsule... Red goes for blue. Blue goes for red
    capsuleLocation = gameState.getBlueCapsules() if isRed else gameState.getRedCapsules()
    if len(capsuleLocation) > 0: 
        cdist = self.getMazeDistance(myPos, capsuleLocation[0])
        # print("CDIST", cdist)
        if cdist > 0:
          features['distanceToCapsule'] = cdist
        else: 
          features['distanceToCapsule'] = 1
          # print("distanceToCapsule", features['distanceToCapsule'])
          # print("Capsule Location: ", capsuleLocation[0])
        
    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    # Distance from chasing defenders
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(ghosts) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
      features['ghostDistance'] = min(dists)

    # Prevent excessive stopping
    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -5, 'distanceToCapsule': -100, 'onOffense':0, 'stop': -110, 'numCarrying': 10, 'ghostDistance': 1,
    'risk': -4, 'reverse': -20}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
  
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    food = self.getFoodYouAreDefending(successor).asList()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
