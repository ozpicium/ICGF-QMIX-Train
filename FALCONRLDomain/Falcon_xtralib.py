import random

GENERIC_ALPHA = [0.1,0.1,0.1]
FAST_BETA = [1.0,1.0,1.0]

SELECT_GAMMA = [0.33,0.33,0.33]
SELECT_RHO = [0.2,0.0,0.5]

ESTIMATE_RHO = [0.2,0.0,0.0]

LEARNING_RHO = [0.2,1.0,0.75]

NEUTRALQ = 0.5

RLGAMMA = 0.1
RLALPHA = 0.5

MAXBOUNDQ = 1.0

initxepsilon = 1.0

#xepsilon = initxepsilon
#dxepsilon = 0.01
#minxepsilon = 0.0

def estimateQ(reward=None, maxQ=None, thisQ=None, rlgamma=RLGAMMA, rlalpha=RLALPHA):
    Q = -1.0
    TDError = reward + rlgamma*maxQ - thisQ
    Q = thisQ + rlalpha * TDError
    if Q < 0:
	    Q = 0.0
    elif Q > MAXBOUNDQ:
	    Q = 1.0
    return Q

def isAll1NormVector(vect=None):
    return all([x >= 1.0 for x in vect])

def oneHotRandomized(vect=None):
    vc = [0.0] * len(vect)
    vc[random.choice(range(len(vc)))] = 1.0
    return vc


class EpsilonGreedy:
    def __init__(self, epsilon=1.0, depsilon=0.1, minepsilon=0.0):
        self.setParam(epsilon=epsilon, depsilon=depsilon, minepsilon=minepsilon)

    def setParam(self, epsilon=1.0, depsilon=0.1, minepsilon=0.0):
        self.mainepsilon = epsilon
        self.epsilon = self.mainepsilon
        self.depsilon = depsilon
        self.minepsilon = minepsilon

    def toExplore(self):
        return self.epsilon > random.random()

    def epsilonDecay(self):
        self.epsilon -= self.depsilon 
        if self.epsilon < self.minepsilon:
            self.epsilon = self.minepsilon

    def resetEpsilon(self):
        self.epsilon = self.mainepsilon

    def setParamEpsilon(self, epsilon=None, depsilon=None, minepsilon=None):
        if epsilon != None:
            self.epsilon = epsilon

        if depsilon != None:
            self.depsilon = depsilon

        if minepsilon != None:
            self.minepsilon = minepsilon

 

