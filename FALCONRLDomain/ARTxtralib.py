#22 Oct 2021
#----------------------------
#Extra functions library for fusion ART beyond FuzzyART model

import copy
import numpy as np
import math


ROUNDDIGIT = 10 #number of digit precision for round


#--Functions for ART2 (ART2A-C)-----------------------------------------------------------

#choiceFieldFuncART2: calculate the choice function for a node j in field k
#given input xck, a weight vector wjck, alphack, and gammack for contribution parameter and choice parameters 
#return the activation value of node j for particular field k
#The template for ART2 choice function for a particular channel/field can be defined as follows:
def choiceFieldFuncART2(xck,wjck,alphack,gammack):
    tp = np.dot(np.array(xck),np.array(wjck))
    btm = np.linalg.norm(np.array(xck))*np.linalg.norm(np.array(wjck))
    return gammack * (round(float(tp)/float(btm),ROUNDDIGIT))

#matchFuncART2: ART2 match function of weight vector wjck with vector xck
#return the match value. 
##The template function for ART2 template matching for a particular channel/field can defined as follows: 
def matchFuncART2(xck,wjck):
    m = 0.0
    denominator = 0.0
    tp = np.dot(np.array(xck),np.array(wjck))
    btm = np.linalg.norm(np.array(xck))*np.linalg.norm(np.array(wjck))
    if btm <= 0:
        return 1.0
    return round(float(tp)/float(btm),10)

#The template include the checking of the match with vigilance parameter
def resonanceFieldART2(xck, wjck, rhok):
    return matchFuncART2(xck, wjck) < rhok

#updWeightsART2: ART template learning function of weight vector wjck with vector xck
#return the updated weight. 
##The template function for ART2 template learning for a particular channel/field can defined as follows: 
def updWeightsART2(rate, weightk, inputk):
    w = np.array(weightk)
    i = np.array(inputk)
    uw = (((1-rate)*w) + (rate*i)).tolist()
    return uw


def ART2ACModelOverride(fa_model, k=-1):		#Override all the resonance search functions with ART2 (ART2A-C) for F1 field(s). 
	#k is the index of the channel to override 
	#set the choice function to activate category field
    fa_model.setChoiceActFunction(cfunction=choiceFieldFuncART2, k=k)

    #set the weight update function
    fa_model.setUpdWeightFunction(ufunction=updWeightsART2, k=k)

    #set the resonance search function
    fa_model.setResonanceFieldFunction(rfunction=resonanceFieldART2, k=k)

    #set the match function
    fa_model.setMatchValFieldFunction(mfunction=matchFuncART2, k=k)

#--------------------------------------------------------------------------------
