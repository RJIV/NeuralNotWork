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
from __future__ import print_function
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
               first = 'gameAgent', second = 'gameAgent'):
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
    # if self.index == 1:
      # print(values, file=sys.stderr)
      # print(self.getPreviousObservation(), file=sys.stderr)

    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    # if self.index == 1:
    #   print(bestActions, file=sys.stderr)

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2 or gameState.getAgentState(self.index).numCarrying > 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

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

    # if self.index == 1:
      # print(str(features) + str(weights), file=sys.stderr)
      # print(gameState.getAgentState(self.index)) # Print out a text representation of the world.

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

class gameAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getDesiredScore(self, gameState):
     if gameState.isOnRedTeam(self.index):
       return gameState.getScore()
     else:
       return gameState.getScore() * -1


  def getFeatures(self, gameState, action):   
    features = util.Counter() 
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    isRed = gameState.isOnRedTeam(self.index)

    teamIndicies = gameState.getRedTeamIndices() if isRed else gameState.getBlueTeamIndices()
    allyIndex = teamIndicies[teamIndicies.index(self.index) - 1]
    allyDistance = self.getMazeDistance(myPos, successor.getAgentState(allyIndex).getPosition())
    # features['allyDistance'] = allyDistance
    if self.getDesiredScore(gameState) <= 0:
     attack = 1
    else:
      attack = 0

    if attack == 1:
      foodList = self.getFood(successor).asList()    
      features['successorScore'] = -len(foodList)#self.getScore(successor)

      # Compute distance to the nearest food
      if len(foodList) > 0: # This should always be True,  but better safe than sorry
        myPos = successor.getAgentState(self.index).getPosition()
        minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
        features['distanceToFood'] = minDistance

        # Determine if the enemy is closer to you than they were last time
        # and you are in their territory.
        # Note: This behavior isn't perfect, and can force Pacman to cower 
        # in a corner.  I leave it up to you to improve this behavior.
        close_dist = 9999.0
        if self.index == 1 and gameState.getAgentState(self.index).isPacman:
          opp_fut_state = [successor.getAgentState(i) for i in self.getOpponents(successor)]
          chasers = [p for p in opp_fut_state if p.getPosition() != None and not p.isPacman]
          if len(chasers) > 0:
            close_dist = min([float(self.getMazeDistance(myPos, c.getPosition())) for c in chasers])

          # View the action and close distance information for each 
          # possible move choice.
        features['fleeEnemy'] = 1.0/close_dist

        return features
    
    else: # Defense  
      mapDimensions = (gameState.data.layout.width, gameState.data.layout.height)
      safePoint = (mapDimensions[0] / 2, mapDimensions[1] / 2)
      distanceFromSafePoint = self.getMazeDistance(myPos, safePoint)

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
    else:
      features['patrolCenter'] = 1 * distanceFromSafePoint
      # print('Patrol', safePoint, distanceFromSafePoint)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features
    
  def getWeights(self, gameState, action):
    #print(self.getScore(self))
    if self.getDesiredScore(gameState) <= 0:
      attack = 1
    else:
      attack = 0
    if attack == 1:
      return {'successorScore': 100, 'distanceToFood': -1, 'fleeEnemy': -100.0}
    else:
      return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'patrolCenter': -1}
