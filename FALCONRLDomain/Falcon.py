
from fusionART import *
from Falcon_xtralib import *

STATE_FIELD = 0
ACTION_FIELD = 1
REWARD_FIELD = 2

class Falcon:
    def __init__(self, f_art=None, flengths=None, fSchema=None, fParams=None, defParams=True):
        self.falpha = GENERIC_ALPHA
        self.fbeta = FAST_BETA
        self.sel_gamma = SELECT_GAMMA
        self.sel_rho = SELECT_RHO

        self.est_gamma = SELECT_GAMMA
        self.est_rho = ESTIMATE_RHO

        self.learn_gamma = SELECT_GAMMA
        self.learn_rho = LEARNING_RHO

        self.exploreStrategy = EpsilonGreedy()

        self.rlgamma = RLGAMMA
        self.rlalpha = RLALPHA

        if f_art:
            self.fusionart = f_art            
        elif fSchema:
            self.fusionart = FusionART(schema=fSchema, beta=self.fbeta,alpha=self.falpha,gamma=self.sel_gamma,rho=self.sel_rho)
        else:
            if flengths:
                self.fusionart = FusionART(numspace=len(flengths),lengths=flengths,beta=self.fbeta,alpha=self.falpha,gamma=self.sel_gamma,rho=self.sel_rho)
        if defParams:
            self.fusionart.setParam('alpha', self.falpha)
            self.fusionart.setParam('beta', self.fbeta)
            self.fusionart.setParam('gamma', self.sel_gamma)
            self.fusionart.setParam('rho', self.sel_rho)
        elif fParams:
            for p in fParams:
                self.fusionart.setParam(p,fParams[p])

    def set_alphabeta_param(self,alpha=GENERIC_ALPHA, beta=FAST_BETA):
        self.falpha = alpha
        self.fbeta = beta

    def set_select_gammarho_param(self, gamma=SELECT_GAMMA, rho=SELECT_RHO):
        self.sel_gamma = gamma
        self.sel_rho = rho

    def set_estimate_gammarho_param(self, gamma=SELECT_GAMMA, rho=ESTIMATE_RHO):
        self.est_gamma = gamma
        self.est_rho = rho

    def set_learn_gammarho_param(self, gamma=SELECT_GAMMA, rho=LEARNING_RHO):
        self.learn_gamma = gamma
        self.learn_rho = rho

    def doDASelActQ(self, state=None):
        MaxQ = [NEUTRALQ, 1-NEUTRALQ]
        self.fusionart.setParam('gamma', self.sel_gamma)
        self.fusionart.setParam('rho', self.sel_rho)

        if hasattr(self.fusionart, 'F1Fields'):
            sname = self.fusionart.F1Fields[STATE_FIELD]['name']
            aname = self.fusionart.F1Fields[ACTION_FIELD]['name']
            avalues = [1] * len(self.fusionart.F1Fields[ACTION_FIELD]['val'])
            rname = self.fusionart.F1Fields[REWARD_FIELD]['name']
            self.fusionart.updateF1bySchema([{'name':sname, 'val':state}, {'name':aname, 'val':avalues, 
                                                'name':rname, 'val':[1.0]}])
        else:
            avalues = [1] * len(self.fusionart.activityF1[ACTION_FIELD])
            self.fusionart.setActivityF1([state, avalues, [1.0,0.0]])
        
        JselAct = self.fusionart.resSearch()
        if not self.fusionart.uncommitted(JselAct):
            avalues = self.fusionart.doRetrieve(JselAct, k=ACTION_FIELD)
        
        self.fusionart.setParam('gamma', self.est_gamma)
        self.fusionart.setParam('rho', self.est_rho)

        qavalues = [1] * len(self.fusionart.F1Fields[ACTION_FIELD]['val'])
        self.fusionart.setActivityF1(qavalues,kidx=ACTION_FIELD)
        JestQ = self.fusionart.resSearch()
        if not self.fusionart.uncommitted(JestQ):
            MaxQ = self.fusionart.doRetrieve(JestQ, k=REWARD_FIELD)
        
        return avalues, MaxQ

    def doDALearn(self, state=None, actions=None, estQ=None):
        self.fusionart.setParam('gamma', self.learn_gamma)
        self.fusionart.setParam('rho', self.learn_rho)

        if hasattr(self.fusionart, 'F1Fields'):
            sname = self.fusionart.F1Fields[STATE_FIELD]['name']
            aname = self.fusionart.F1Fields[ACTION_FIELD]['name']
            rname = self.fusionart.F1Fields[REWARD_FIELD]['name']
            self.fusionart.updateF1bySchema([{'name':sname, 'val':state}, {'name':aname, 'val':actions}, 
                                                {'name':rname, 'val':[estQ]}])
        else:
            self.fusionart.setActivityF1([state, actions, [estQ,1-estQ]])
        Jlearn = self.fusionart.resSearch()
        ucommit = self.fusionart.uncommitted(Jlearn)
        self.fusionart.autoLearn(Jlearn)

        return not ucommit, Jlearn

    #some note about RLCycle -- explorActs can be specified if random action selection depends on some contraints imposed by the domain problems.
    # If the random action selection can be conducted independently from the domain, then explorActs can be left unspecified, so that the random selection
    # can be conducted internally by RLCycle itself
    def RLCycle(self, prevState=None, prevActions=None, currentState=None, explorActs=None, prevQ=None, reward=None, terminalstate=False):
        estQ = NEUTRALQ
        xacts = copy.deepcopy(self.fusionart.activityF1[ACTION_FIELD])
        if explorActs:
            xacts = explorActs
        else:
            xacts = oneHotRandomized(vect=xacts)

        #action selection
        if self.exploreStrategy.toExplore():
            acts = xacts
            Qval = NEUTRALQ
        else:
            acts, Q = self.doDASelActQ(state=currentState)
            Qval = Q[0]
        if isAll1NormVector(acts):
            acts = xacts

        #Q Learning
        if prevState != None:
            if terminalstate:
                estQ = reward
            else:
                estQ = estimateQ(reward=reward, maxQ=Qval, thisQ=prevQ, rlgamma=self.rlgamma, rlalpha=self.rlalpha)
            uncommit, nodeidx = self.doDALearn(state=prevState, actions=prevActions, estQ=estQ)
    
        return acts, Qval, estQ

        
         



        



            