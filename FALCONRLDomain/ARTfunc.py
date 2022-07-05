## updated 29 July 2021 by Budhitama Subagdja -- to add bypassing condition for the sequence readout based on schema to handle reading out generalized values
## updated 21 July 2021 by Budhitama Subagdja -- to get rid of unnecessary print out of maxval, maxidxs etc.
## updated 12 May 2021 by Budhitama Subagdja
## from 01 Apr 2020
##-------------------------------------------


import copy
import numpy as np
import math

#choiceFuncFuzzy: calculate the choice function for a node j in field k
#given input xck, a weight vector wjck, alphack, and gammack for contribution parameter and choice parameters 
#return the activation value of node j for particular field k
def choiceFieldFuncFuzzy(xck,wjck,alphack,gammack):
	xwj = np.amin(np.array([xck,wjck]),axis=0)
	tp = np.sum(xwj)
	btm = np.sum(wjck)+alphack
	return gammack * (float(tp)/float(btm))


#choiceFuncFuzzy: calculate the choice function for a node j
#given input xc, a weight vector wjc, alphac as learning rate, and gammac for contribution parameter as list of parameters
#return the activation value of node j
def choiceFuncFuzzy(xc,wjc,alphac,gammac):
	tj = 0
	tp = 0
	btm = 0
	for k in range(len(xc)):
		if type(wjc[k]).__module__ == 'numpy':
			xwj = np.amin(np.array([xc[k],wjc[k]]),axis=0)
			tp = np.sum(xwj)
			btm = np.sum(wjc[k])+alphac[k]
		else:
			tp = 0
			btm = alphac[k]
			for i in range(len(xc[k])):
				tp += min(xc[k][i],wjc[k][i])
				btm += wjc[k][i]
		tj += gammac[k] * (float(tp)/float(btm))
	return tj


#choiceActFuzzy: calculate the choice function for the entire F2
#given input xc, all weights wc, alphac, and gammac
#return the vector of F2
def choiceActFuzzy(xc,wc,alphac,gammac,F2=[]):
	if(len(F2)<len(wc)):
		F2 = [0.0]*len(wc)
	for j in range(len(wc)):
		F2[j] = choiceFuncFuzzy(xc,wc[j],alphac,gammac)
	return F2

#actReadoutFuzzy: readout the template with Fuzzy AND
#wj is the template (weights) to readout to xold
#returning xnew as the new readout vector
def actReadoutFuzzy(xold,wj):
	xnew = []
	for k in range(len(xold)):
		if type(wj[k]).__module__ == 'numpy':
			xnew.append(np.amin([xold[k],wj[k]],axis=0))
		else:
			xn = []
			for i in range(len(xold[k])):
				xn.append(min(xold[k][i],wj[k][i]))
			xnew.append(xn)
	return xnew
	

#matchFuncFuzzy: Fuzzy match function of weight vector wjck with vector xck
#return the match value
def matchFuncFuzzy(xck,wjck):
	m = 0.0
	denominator = 0.0
	if type(wjck).__module__ == 'numpy':
		xcko = np.amin(np.array([xck,wjck]),axis=0)
		m = np.sum(xcko)
		denominator = np.sum(xck)
	else:
		for i in range(len(xck)):
			m += min(xck[i],wjck[i])
			denominator += xck[i]
	if denominator <= 0:
		return 1.0
	return m/denominator
  
#matchValFuzzy: Fuzzy match of  weights wjc with fields xc
#returning the list of match value for every field
def matchValFuzzy(xc,wjc):
	mF1 = []
	if len(xc) > 0:
		mF1 = [0.0]*len(xc)
		for k in range(len(xc)):
			mF1[k] = matchFuncFuzzy(xc[k],wjc[k])
	return mF1

#resonanceFuzzy: given fields xc, weights wjc, and vigilances rho
#check wether it is in resonance condition or not
#return True (resonance) or False (not in resonance)
def resonanceFuzzy(xc,wjc,rho):
	matched = True
	for k in range(len(xc)):
		if matchFuncFuzzy(xc[k],wjc[k]) < rho[k]:
			matched = False
	return matched
	
def resonanceFieldFuzzy(xck, wjck, rhok):
	print('matchFuncFuzzy ', matchFuncFuzzy(xck, wjck), ' rhok ', rhok)
	return matchFuncFuzzy(xck, wjck) < rhok

def updWeightFuzzy(rate, weight, input):
	return (1- rate)*weight + rate*min(weight,input)
	

def updWeightsFuzzy(rate, weightk, inputk):
	w = np.array(weightk)
	i = np.array(inputk)
	uw = (((1-rate)*w) + (rate*(np.amin([w,i],axis=0)))).tolist()
	return uw

#mresonance: check if given the match values and rhos. it's in resonance
#return True or False
def mresonance(ms,rho):
	if type(ms).__module__ == 'numpy':
		return np.all(ms >= rho)
	else:
		for k in range(len(ms)):
			if ms[k] < rho[k]:
				return False
	return True
		
#pmismatch: check if given matching criteria mcriteria, it's in perfect mismatch.
#return True or False
def pmismatch(match,mcriteria):
	if type(match).__module__ == 'numpy':
		return np.all(match >= mcriteria)
	else:
		for k in range(len(match)):
			if match[k] < mcriteria[k]:
				return False
	return True
	
#genComplementVect: generate complement-coded vector from a vector val
#return a new complement coded vector as a list
def genComplementVect(val):
	compl = [1-i for i in val]
	ccode = val + compl
	return list(ccode)
	
#genActivityVect: generate activity vector given a vector val that can be complemented or not (compl)
#return a new activity vector as a list
def genActivityVect(val,compl=False):
	if(compl):
		#return genComplementVect(val)
		return complementEncode(val)
	return list(val)
	
#readOutVectSym: readout a structure of input/output field (in JSON) given an activity vector which can be complemented (compl)
#return a (JSON) structure of in input/output field {'val': <vector values>, 'vcompl':<complement of val if specified>} 
def readOutVectSym(activity, compl=False):
	if(compl):
		#ilen = len(activity)/2
		#v = activity[0:ilen]
		#c = activity[ilen:ilen*2]
		v, c = deComplement(activity)
		vc = {'val':v, 'vcompl':c}
		return vc
	return {'val':list(activity)}

def readOutVectSchema(activity, fschema):
	c = False
	if 'compl' in fschema:
		c = fschema['compl']
	return 
	if(compl):
		ilen = len(activity)/2
		v = activity[0:ilen]
		c = activity[ilen:ilen*2]
		vc = {'val':v, 'vcompl':c}
		return vc
	return {'val':list(activity)}

#readOutVectAttribs: readout a structure of input/output field (JSON) with attribute, given an activity vector
#return a (JSON) structure of input/output field {'attval': <dictionary of attribute-value of the field>, 'attcompl':<dictionary of the complement of attribute-value of the field>, 'compl':indicate if the values are complemented}
def readOutVectAttribs(activity, compl=False, attr=[]):
	afield = {}
	afield['attval'] = {}
	afield['compl'] = compl
	if compl:
		afield['compl'] = True
		afield['attcompl'] = {}
		ilen = len(activity)/2
		if len(attr)>0:
			for i in range(len(attr)):
				afield['attval'][attr[i]] = activity[i]
				afield['attcompl'][attr[i]] = activity[i+ilen]
		else:
			for i in range(ilen):
				afield['attval'][i] = activity[i]
				afield['attcompl'][i] = activity[i+ilen]
	else:
		afield['compl'] = False
		if len(attr)>0:
			for i in range(len(attr)):
				afield['attval'][attr[i]] = activity[i]
		else:
			for i in range(len(activity)):
				afield['attval'][i] = activity[i]
	return afield

#initField: initialized a field structure given the length of the field vector or the list of attributes
#return a (JSON) structure of a field {'name': <optional name of the field>, 'compl':<indicate complemented>','val':<value vector>, 'vcompl':<complemented value vector>} 
def initField(fname='',fcompl=False,flen=0,fattr=[]):
	fld = {}
	if len(fname)>0:
		fld['name']=fname
	fld['compl']=fcompl
	if len(fattr)>0:
		flen = len(fattr)
		fld['attrib'] = list(fattr)
	#fld['val']=[1.0]*flen
	fld['val']=[0.0]*flen
	if fld['compl']:
	#	fld['vcompl'] = [1.0]*flen
		fld['vcompl'] = [0.0]*flen
	return fld
	

#initField: initialized a field structure given the field schema specification in fschema
#return a (JSON) structure of a field {'name': <optional name of the field>, 'compl':<indicate complemented>','val':<value vector>, 'vcompl':<complemented value vector>} 
def initFieldSchema(fschema):
	uschema = copy.deepcopy(fschema)
	fname = ''
	attrib = []
	cmpl = False
	if 'name' in fschema:
		fname = fschema['name']
	if 'compl' in fschema:
		cmpl = fschema['compl']
	if 'attrib' in fschema:
		attrib = list(fschema['attrib'])
	if 'val' in fschema:
		uschema.update(initField(fname,fcompl=cmpl,flen=len(fschema['val']),fattr=attrib))
	else:
		uschema.update(initField(fname,fcompl=cmpl,fattr=attrib))
	return uschema

#getActivityFromField: generate activity vector (maybe complemented) given a field structure or schema
#return the activity vector based on the field schema
def getActivityFromField(fschema):
	act = []
	if 'compl' in fschema:
		if fschema['compl']:
			if 'vcompl' in fschema:
				act = fschema['val'] + fschema['vcompl']
			else:
				c = [1-i for i in fschema['val']]
				act = fschema['val'] + c
		else:
			act = list(fschema['val'])
	else:
		act = list(fschema['val'])
	return act
	

#setValFieldSchema: set a value/values of a schema based on an index in the list (optional)
#return the schema with the updated value/values
def setValFieldSchema(fschema,val,vcom=[],idx=-1):
	uschema = copy.deepcopy(fschema)
	if idx>=0:
		uschema['val'][idx] = val
		if not type(vcom) == list:
			if 'vcompl' in uschema:
				uschema['vcompl'][idx] = vcom
	else:
		uschema['val'] = list(val)
	if uschema['compl']:
		if len(vcom)>0:
			uschema['vcompl'] = list(vcom)
		else:
			uschema['vcompl'] = [(1-i) for i in uschema['val']]
	return uschema
	
#setValAttrFieldSchema: set a value/values of a schema based on the attribute of the value
#return the schema with the updated value/values
def setValAttrFieldSchema(fschema,val,att):
	uschema = copy.deepcopy(fschema)
	idx = -1
	if 'attrib' in fschema:
		idx = uschema['attrib'].index(att)
	else:
		idx = att
	uschema = setValFieldSchema(uschema,val,idx=idx)
	return uschema
	

def setSchemabyAttVal(fschema,attval):
	uschema = copy.deepcopy(fschema)
	if 'attrib' in fschema:
		for att in attval.keys():
			if att in fschema['attrib']:
				idx = fschema['attrib'].index(att)
				uschema = setValFieldSchema(uschema,attval[att],idx=idx)
	return uschema
	

def isSchemaWithAtt(fschema,att,val):
	if att in fschema:
		if fschema[att] == val:
			return True
	return False

def refreshComplSchema(fschema):
	uschema = copy.deepcopy(fschema)
	if 'compl' in uschema:
		if uschema['compl']:
			for a in range(len(uschema['val'])):
				uschema['vcompl'][a] = 1 - uschema['val'][a]
	return uschema


def listAttVal(dlist, attr):
	rlist = []
	for i in range(len(dlist)):
		if attr in dlist[i]:
			rlist.append(dlist[i][attr])
		else:
			return []
	return rlist
	
def attValList(olist,dlist,attr):
	for i in range(len(olist)):
		dlist[i][attr] = olist[i]
	return dlist
	
#complementEncode: make a complement coded list (appended w/ inversed list
#return a doubled length list with appended complement
def complementEncode(oList):
	if type(oList).__module__ != 'numpy':
		arrvect = np.array(oList)
		carrvect = 1 - arrvect
		return np.append(arrvect,carrvect).tolist()
	else:
		return np.append(oList, 1-oList)

#deComplement: take the uncomplemented part of a complemented list/array		
def deComplement(oList):
	return oList[:int(len(oList)/2)], oList[int(len(oList)/2):] 
	

#normalizeVals: normalize values with specified maximum and minimum value		
#return a list (or numpy array if the input is numpy array) with normalized values (range from 0 to 1)
def normalizeVals(oList, vmax, vmin):
	if type(oList).__module__ != 'numpy':
		arrvect = np.array(oList)
		return ((abs(vmin-arrvect))/(vmax-vmin)).tolist()
	else:
		return (abs(vmin-oList))/(vmax-vmin)
		
#deNormalizeVals: denormalize values with specified maximum and minimum value		
#return a list (or numpy array if the input is numpy array) with values denormalized
def deNormalizeVals(oList, vmax, vmin):
	if type(oList).__module__ != 'numpy':
		arrvect = np.array(oList)
		return (vmax - ((vmax-vmin)*arrvect)).tolist()
	else:
		return vmax - ((vmax-vmin)*oList)
	
	
#decayVals: reduce or decay all values in oList based on factor tau. It cuts the value to 0 if < tresh
#return the list with decayed values	
def decayVals(oList, tau=0.1, tresh=0.0, accdig=10, decaying=True):	#updated to handle rounding accuracy 29320
	decay = 1 - tau
	if decaying:
		#if type(oList).__module__ != 'numpy':
		arrvect = np.array(oList)
		arrvect = np.round(arrvect * decay,accdig)				#***
		arrvect[arrvect < tresh] = 0.0
		return arrvect.tolist()
		'''else:
			oList = oList * decay
			oList[oList < tresh] = 0.0
			return oList'''
	else:
		#if type(oList).__module__ != 'numpy':
		arrvect = np.array(oList)
		arrvect = np.round(arrvect / decay, accdig)				#***
		arrvect[arrvect > 1-tresh] = 0.0
		return arrvect.tolist()
		'''else:
			oList = oList / decay
			oList[oList > 1-tresh] = 0.0
			return oList'''
	

def inverseDecay(val, tau=0.1, stau=0.0, accdig=10, decaying=True):   #added to handle repetition 26320
	decay = 1 - tau
	subinc = 1 + stau
	if decaying:
		return np.round(1 - ((1 - val)*decay*subinc), accdig)
	else:
		return np.round(1 - ((1 - val)/(decay*subinc)), accdig)


def InversdecayVals(oList, tau=0.1, tresh=0.0, stau=0.0, tidx=-1, decaying=True):   #updated for repetition 26320
	decay = 1 - tau
	subinc = 1 + stau #**
	arrvect = np.array(oList)
	tidv = 0
	if tidx >= 0:
		tidv = inverseDecay(arrvect[tidx], tau=tau, stau=stau, decaying=decaying)
	arrvect[arrvect > 0] = inverseDecay(arrvect[arrvect > 0], tau=tau, stau=0.0, decaying=decaying)
	if decaying:
		#arrvect[arrvect > 0]*=decay
		#arrvect[arrvect > 0]+= tau
		#arrvect[arrvect > (1-tresh)] = 0.0
		#arrvect[arrvect > 0] = 1 - ((1 - arrvect[arrvect > 0])*decay)  #**
		#arrvect[arrvect > 0] = inverseDecay(arrvect[arrvect > 0], tau=tau, stau=0.0, decaying)
		arrvect[arrvect > (1-tresh)] = 0.0
	else:
		#arrvect[arrvect > 0]-= tau
		#arrvect[arrvect > 0]/=decay
		#arrvect[arrvect < tresh] = 0.0
		#arrvect[arrvect > 0] = 1 - ((1 - arrvect[arrvect > 0])/decay)
		arrvect[arrvect < tresh] = 0.0
	if tidx >= 0:
		arrvect[tidx] = tidv
	return arrvect.tolist()
		

def complementDecayVals(oList,coList, tidx=-1, tau=0.1, stau=0.0, tresh=0.0, decaying=True):	#updated for repetition 26320
	oL = decayVals(oList,tau=tau,tresh=tresh,decaying=decaying)
	coL = InversdecayVals(coList, tau=tau, stau=stau, tidx = tidx, tresh=tresh, decaying=decaying)
	#if type(oList).__module__ != 'numpy':
	arroL = np.array(oL)
	arrcoL = np.array(coL)
	arroL[arroL < tresh] = 0.0
	arrcoL[arrcoL > (1-tresh)] = 0.0
	return arroL.tolist(), arrcoL.tolist()
	'''else:
		oL[oL < tresh] = 0.0
		coL[coL > (1-tresh)] = 0.0
		return oL, coL'''


def insertDecayVals(oList, tau=0.1, tresh=0.0, idx=-1, maxv=0.9, accdig=10, duprep=True):
	decay = 1-tau
	if idx >= 0:
		oL = decayVals(oList, tau, tresh)
		if idx < len(oList):
			if (oL[idx] > 0.0) and duprep:
				oL.append(maxv)
			else:
				oL[idx] = maxv
		else:
			oL.append(maxv) 
		return np.round(np.array(oL),accdig).tolist()
	return oList
			
def maxReadoutVals(oList, tau=0.1, tresh=0.0, accdig=10, queue=True):
	decay = 1-tau
	if queue:
		maxval = max(oList)
		for val in oList:
			if (val > 0) and (val < maxval):
				maxval = val
		maxidxs = [i for i in range(len(oList)) if oList[i]==maxval]
		oL = list(oList)
		for j in maxidxs:
			oL[j] = 0.0
		oL = decayVals(oL,tau,tresh)
		return maxidxs[0], np.round(np.array(oL),accdig).tolist()
	else:
		maxval = max()
		maxidxs = [i for i in range(len(oList)) if oList[i]==maxval]
		oL = list(oList)
		for j in maxidxs:
			oL[j] = 0.0
		oL = decayVals(oL, tau, tresh, decaying=False)
		return maxidxs[0], np.round(np.array(oL),accdig).tolist()
		
'''   #this is the old version-----
def maxComplReadoutVals(oList, coList, tau=0.1, tresh=0.0, stau=0.01, accdig=10, queue=True):
	decay = 1-tau
	subdecay = 1-stau
	idx=-1
	lmax = 0
	oL = list(oList)
	coL = list(coList)
	if queue:
		maxval = max(coL)
		maxidxs = [i for i in range(len(coL)) if coL[i]==maxval]
		if len(maxidxs) > 1:
			maxco = [oL[maxidxs[i]] + coL[maxidxs[i]] for i in range(len(maxidxs))]
			idx = maxidxs[maxco.index(min(maxco))]
			#idx = maxidxs[maxco.index(max(maxco))]
			print('maxco: ', maxco, ' ', idx)
		else:
			idx = maxidxs[0]
		if (idx >= 0): 
			lmax = coL[idx]
			if lmax > 0:
				if oL[idx]+coL[idx] <= 1.0:
					oLo, coLo = complementDecayVals(oL, coL, tau=tau, tresh=tresh)
					for j in maxidxs:
						coLo[j] = lmax
					oLo[idx] = 0.0
					coLo[idx] = 0.0
				else:
					oLo, coLo = complementDecayVals(oL, coL, tau=tau, tresh=tresh)
					print("oLo, coLo: ", oLo, ' ', coLo)
					oLo[idx] = oLo[idx]/subdecay
					for j in maxidxs:
						coLo[j] = lmax
					print("new oLo, coLo: ", oLo, ' ', coLo)
				return idx, np.round(np.array(oLo),accdig).tolist(), np.round(np.array(coLo),accdig).tolist()
			idx = -1
	else:
		maxval = max(oL)
		maxidxs = [i for i in range(len(oL)) if oL[i]==maxval]	
		if len(maxidxs) > 1:
			maxco = [oL[maxidxs[i]] + coL[maxidxs[i]] for i in range(len(maxidxs))]
			idx = maxidxs[maxco.index(max(maxco))]
		else:
			idx = maxidxs[0]
		
		if (idx >= 0): 
			lmax = oL[idx]
			if lmax > 0:
				if oL[idx]+coL[idx] <= 1.0:
					oLo, coLo = complementDecayVals(oL, coL, tau=tau, tresh=tresh, decaying=False)
					oLo[idx] = 0.0
					coLo[idx] = 0.0
				else:
					oLo, coLo = complementDecayVals(oL, coL, tau=tau, tresh=tresh, decaying=False)
					oLo[idx] = oLo[idx]/subdecay
					if oLo[idx] > 1-tresh:
							oLo[idx] = 1-tresh
				return idx, np.round(np.array(oLo),accdig).tolist(), np.round(np.array(coLo),accdig).tolist()
			idx = -1
		
	return idx, np.round(np.array(oL),accdig).tolist(), np.round(np.array(coL),accdig).tolist()
'''
				
#updated 29 July 2021 by Budhitama Subagdja -- adding bpasscomp flag to bypass the reading out of generalized values (if True)
def maxComplReadoutVals(oList, coList, tau=0.1, tresh=0.0, stau=0.01, accdig=10, queue=True, bpasscomp=False): 
	decay = 1 - tau
	subinc = 1 + stau #**

	idx=-1
	lmax = 0
	oL = list(oList)
	coL = list(coList)
	oLo = list(oL)
	coLo = list(coL)
	if queue:
		maxval = max(coL)
		maxidxs = [i for i in range(len(coL)) if coL[i]==maxval]
		if (len(maxidxs) >= 1) and (maxval > 0):
			maxco = [ round(oL[maxidxs[i]] + coL[maxidxs[i]],accdig) for i in range(len(maxidxs))]
			#maxco = [ oL[maxidxs[i]] + coL[maxidxs[i]] for i in range(len(maxidxs))]
			if (1.0 in maxco) or bpasscomp:
				if bpasscomp:
					idx = maxidxs[0]
				else:
					idx = maxidxs[maxco.index(1.0)]
					#print("condition 1, maxval: ", maxval, " val: ", oL[idx], " ",  idx)
				oLo, coLo = complementDecayVals(oL, coL, tau=tau, tresh=tresh)
				oLo[idx] = 0.0
				coLo[idx] = 0.0
				#return idx, np.round(np.array(oLo),accdig).tolist(), np.round(np.array(coLo),accdig).tolist()
				return idx, np.array(oLo).tolist(), np.array(coLo).tolist()
				
				#idx = maxidxs[maxco.index(min(maxco))]
				#idx = maxidxs[maxco.index(max(maxco))]
				#print('maxco: ', maxco, ' ', idx)
			else:
				#print("condition 1 only")
				idx = maxidxs[0]
		if (idx >= 0): 
			print("condition 2, maxval: ", maxval, " val: ", oL[idx], " ",  idx)
			lmax = coL[idx]
			if lmax > 0:
				mdl = modLog(1-lmax,base=decay,accdig=accdig)
				if (mdl == 1) or (mdl == 0):
					oLo, coLo = complementDecayVals(oL, coL, tau=tau, tresh=tresh)
					coLo[idx] = 1 - oLo[idx]
				else:
					tmpcv = inverseDecay(lmax, tau=tau, stau=stau, accdig=accdig, decaying=False)
					oLo, coLo = complementDecayVals(oL, coL, tau=tau, tresh=tresh)
					coLo[idx] = tmpcv
				#return idx, np.round(np.array(oLo),accdig).tolist(), np.round(np.array(coLo),accdig).tolist()
				return idx, np.array(oLo).tolist(), np.array(coLo).tolist()
	else:
		maxval = max(oL)
		maxidxs = [i for i in range(len(oL)) if oL[i]==maxval]	
		if (len(maxidxs) >= 1) and (maxval > 0) :
			maxco = [round(oL[maxidxs[i]] + coL[maxidxs[i]],accdig) for i in range(len(maxidxs))]
			if 1.0 in maxco:
				idx = maxco.index(1.0)
				oLo, coLo = complementDecayVals(oL, coL, tau=tau, tresh=tresh, decaying=False)
				oLo[idx] = 0.0
				coLo[idx] = 0.0
				#return idx, np.round(np.array(oLo),accdig).tolist(), np.round(np.array(coLo),accdig).tolist()
				#idx = maxidxs[maxco.index(min(maxco))]
				#idx = maxidxs[maxco.index(max(maxco))]
				#print('maxco: ', maxco, ' ', idx)
			else:
				maxco = [modLog(1-coL[maxidxs[i]],base=decay,accdig=accdig) for i in range(len(maxidxs))]
				if 0.0 in maxco:
					idx = maxidxs[maxco.index(0.0)]
					oLo, coLo = complementDecayVals(oL, coL, tau=tau, tidx=idx, stau=stau, tresh=tresh, decaying=False)
					oLo[idx] = 1 - coLo[idx]
				else:
					idx = maxidxs[0]
					tmpv = round(oL[idx]*decay, accdig)
					oLo, coLo = complementDecayVals(oL, coL, tau=tau, tidx=idx, stau=stau, tresh=tresh, decaying=False)
					oLo[idx] = tmpv
			return idx, np.round(np.array(oLo),accdig).tolist(), np.round(np.array(coLo),accdig).tolist()		
	return idx, np.round(np.array(oL),accdig).tolist(), np.round(np.array(coL),accdig).tolist()

def maxComplReadoutValsX(oList, coList, tau=0.1, tresh=0.0, stau=0.01, accdig=10, queue=True):
	decay = 1 - tau
	subinc = 1 + stau #**
	subdecay = 1 - stau #***

	idx=-1
	lmax = 0
	oL = list(oList)
	coL = list(coList)
	oLo = list(oL)
	coLo = list(coL)
	if queue:
		maxval = max(coL)
		print('maxval=', maxval)
		maxidxs = [i for i in range(len(coL)) if coL[i]==maxval]
		print('maxidxs: ', maxidxs)
		if (len(maxidxs) >= 1) and (maxval > 0):
			maxco = [ round(oL[maxidxs[i]] + coL[maxidxs[i]],accdig) for i in range(len(maxidxs))]
			#maxco = [ oL[maxidxs[i]] + coL[maxidxs[i]] for i in range(len(maxidxs))]
			print('maxco: ', maxco)
			#if 1.0 in maxco:
			dtone = [maxco[i] for i in range(len(maxco)) if maxco[i] <= 1.0]
			if len(dtone) > 0:
				#idx = maxidxs[maxco.index(1.0)]
				idx = maxidxs[maxco.index(dtone[0])]
				print('idx: ', idx)
				oLo, coLo = complementDecayVals(oL, coL, tau=tau, tresh=tresh)
				oLo[idx] = 0.0
				coLo[idx] = 0.0
				#return idx, np.round(np.array(oLo),accdig).tolist(), np.round(np.array(coLo),accdig).tolist()
				return idx, np.array(oLo).tolist(), np.array(coLo).tolist()
				
				#idx = maxidxs[maxco.index(min(maxco))]
				#idx = maxidxs[maxco.index(max(maxco))]
				#print('maxco: ', maxco, ' ', idx)
			else:
				idx = maxidxs[0]
		if (idx >= 0): 
			lmax = coL[idx]
			print('lmax: ', lmax)
			if lmax > 0:
				print('checking modlog?')
				maxv = oL[idx]
				#mdl = modLog(1-lmax,base=decay,accdig=accdig)
				mdl = modLog(maxv,base=decay,accdig=accdig)
				if (mdl == 1) or (mdl == 0):
					oLo, coLo = complementDecayVals(oL, coL, tau=tau, tresh=tresh)
					coLo[idx] = 1 - oLo[idx]
				else:
					#tmpcv = inverseDecay(lmax, tau=tau, stau=stau, accdig=accdig, decaying=False)
					tmpcv = inverseDecay(lmax, tau=tau, accdig=accdig, decaying=False)
					print('tmpcv', tmpcv)
					oLo, coLo = complementDecayVals(oL, coL, tau=tau, tresh=tresh)
					coLo[idx] = tmpcv
					oLo[idx] /= subdecay
				print('idx: ', idx)
				return idx, np.round(np.array(oLo),accdig).tolist(), np.round(np.array(coLo),accdig).tolist()
				#return idx, np.array(oLo).tolist(), np.array(coLo).tolist()
	else:
		maxval = max(oL)
		maxidxs = [i for i in range(len(oL)) if oL[i]==maxval]	
		if (len(maxidxs) >= 1) and (maxval > 0) :
			maxco = [round(oL[maxidxs[i]] + coL[maxidxs[i]],accdig) for i in range(len(maxidxs))]
			if 1.0 in maxco:
				idx = maxco.index(1.0)
				oLo, coLo = complementDecayVals(oL, coL, tau=tau, tresh=tresh, decaying=False)
				oLo[idx] = 0.0
				coLo[idx] = 0.0
				#return idx, np.round(np.array(oLo),accdig).tolist(), np.round(np.array(coLo),accdig).tolist()
				#idx = maxidxs[maxco.index(min(maxco))]
				#idx = maxidxs[maxco.index(max(maxco))]
				#print('maxco: ', maxco, ' ', idx)
			else:
				maxco = [modLog(1-coL[maxidxs[i]],base=decay,accdig=accdig) for i in range(len(maxidxs))]
				if 0.0 in maxco:
					idx = maxidxs[maxco.index(0.0)]
					oLo, coLo = complementDecayVals(oL, coL, tau=tau, tidx=idx, stau=stau, tresh=tresh, decaying=False)
					oLo[idx] = 1 - coLo[idx]
				else:
					idx = maxidxs[0]
					tmpv = round(oL[idx]*decay, accdig)
					oLo, coLo = complementDecayVals(oL, coL, tau=tau, tidx=idx, stau=stau, tresh=tresh, decaying=False)
					oLo[idx] = tmpv
			return idx, np.round(np.array(oLo),accdig).tolist(), np.round(np.array(coLo),accdig).tolist()		
	return idx, np.round(np.array(oL),accdig).tolist(), np.round(np.array(coL),accdig).tolist()



		
'''		#this is the old version-----
def insertComplDecayVals(oList, coList, tau=0.1, tresh=0.0, idx=-1, maxv=0.9, stau=0.01, accdig=10, duprep=False):
	decay = 1-tau
	subdecay = 1-stau
	if idx >= 0:
		maxoL = max(oList)
		maxcoL = max(coList)
		oL, coL = complementDecayVals(oList, coList, tau, tresh)
		if max(coL) > (1-tresh):
			oL = list(oList)
			coL = list(coList)
			i, oL, coL = maxComplReadoutVals(oL, coL, tau, tresh, stau, accdig)
		if idx < len(oList):
			if (oL[idx] <= 0) and (coL[idx] <= 0):
				oL[idx] = maxv
				coL[idx] = 1-maxv
			else:
				if duprep:
					oL.append(maxv)
					coL.append(1-maxv)
				else:
					oL[idx] = maxoL*subdecay
		else:
			oL.append(maxv)
			coL.append(1-maxv)
		return np.round(np.array(oL),accdig).tolist(), np.round(np.array(coL),accdig).tolist()
	return np.round(np.array(oList),accdig).tolist(), np.round(np.array(coList),accdig).tolist()
'''

def insertComplDecayVals(oList, coList, tau=0.1, tresh=0.0, idx=-1, maxv=0.9, stau=0.01, accdig=10, duprep=False):
	#decay = 1-tau
	#subdecay = 1-stau
	if idx >= 0:
		maxoL = max(oList)
		maxcoL = max(coList)
		oL = copy.deepcopy(oList)
		coL = copy.deepcopy(coList)
		if idx < len(oList):
			if (oL[idx] <= 0) and (coL[idx] <= 0):
				oL, coL = complementDecayVals(oL, coL, tau=tau, tresh=tresh)
				oL[idx] = maxv
				coL[idx] = 1-maxv
			else:
				if duprep:
					oL, coL = complementDecayVals(oL, coL, tau=tau, tresh=tresh)
					oL.append(maxv)
					coL.append(1-maxv)
				else:
					if isComplemented(oL[idx], coL[idx]):
						oL, coL = complementDecayVals(oL, coL, tau=tau, tresh=tresh)
					else:
						oL, coL = complementDecayVals(oL, coL, tau=tau, tidx=idx, stau=stau, tresh=tresh)
					oL[idx] = maxv
		#return np.round(np.array(oL),accdig).tolist(), np.round(np.array(coL),accdig).tolist()
		return np.array(oL).tolist(), np.array(coL).tolist()
	#return np.round(np.array(oList),accdig).tolist(), np.round(np.array(coList),accdig).tolist()				

def insertComplDecayValsX(oList, coList, tau=0.1, tresh=0.0, idx=-1, maxv=0.9, stau=0.01, accdig=10, duprep=False):
	decay = 1-tau
	subdecay = 1-stau
	if idx >= 0:
		maxoL = max(oList)
		maxcoL = max(coList)
		oL = copy.deepcopy(oList)
		coL = copy.deepcopy(coList)
		if idx < len(oList):
			if (oL[idx] <= 0) and (coL[idx] <= 0):
				oL, coL = complementDecayVals(oL, coL, tau=tau, tresh=tresh)
				oL[idx] = maxv
				coL[idx] = 1-maxv
			else:
				if duprep:
					oL, coL = complementDecayVals(oL, coL, tau=tau, tresh=tresh)
					oL.append(maxv)
					coL.append(1-maxv)
				else:
					'''if isComplemented(oL[idx], coL[idx]):
						oL, coL = complementDecayVals(oL, coL, tau=tau, tresh=tresh)
					else:
						oL, coL = complementDecayVals(oL, coL, tau=tau, tidx=idx, stau=stau, tresh=tresh)
					oL[idx] = maxv'''
					oL, coL = complementDecayVals(oL, coL, tau=tau, tresh=tresh)
					if isComplemented(oL[idx], coL[idx]):
						oL[idx] = maxv
						print('complemented: ', oL[idx])
					else:
						oL[idx] *= subdecay
						oL[idx] /= decay**(round(math.log(oL[idx],decay),0) - 1)
						print('non-complemented: ', oL[idx])

						
		return np.round(np.array(oL),accdig).tolist(), np.round(np.array(coL),accdig).tolist()
		#return np.array(oL).tolist(), np.array(coL).tolist()
	return np.round(np.array(oList),accdig).tolist(), np.round(np.array(coList),accdig).tolist()				

		
def competedValues(oList, widx, val=1.0):
	if type(oList).__module__ != 'numpy':
		arrvect = np.array(oList)*0.0
		arrvect = arrvect.tolist()
	else:
		arrvect = oList*0.0
	if widx < len(arrvect):
		arrvect[widx] = val
	return arrvect
	
	
def resetValues(oList):
	return np.array(oList)*0.0
	
def isComplemented(v, cv):
	return (v + cv == 1.0)
		
def modLog(val,base=0.9,accdig=10):
	print('modlog: ', round(math.log(val,base) % 1,accdig))
	#return round(math.log(val,base),accdig) % 1
	return round(math.log(val,base) % 1,accdig)
	#return math.log(val,base) % 1

#default initialize complement value 29082021
def defaultInitComplVal(val=0):
	return val

#default insert complement values in sequence with schema 29082021
def insertSchemaComplVal(fusART=None, schema=None, tau=0, tresh=0, idx=0, maxv=0, stau=0, accdig=0):
	v = schema['val']
	cv = schema['vcompl']
	v, cv = insertComplDecayVals(v, cv, tau=tau, tresh=tresh, idx=idx, maxv=maxv, stau=stau, accdig=accdig)
	fusART.updateF1bySchema([{'name':schema['name'], 'val':v, 'vcompl':cv}],refresh=False) 