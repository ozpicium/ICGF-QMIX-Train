## latest udpate 8 Sept 2021 by Budhitama Subagdja -- adding indicator for perfect mismatch
## latest update 26 Aug 2021 by Budhitama Subagdja -- adding uncommitted check during resSearch to deal with template matching that can't overwrite uncommitted code
## updated 24 Aug 2021 by Budhitama Subagdja -- dealing with overriding schema complement refresh 
## updated 29 July 2021 by Budhitama Subagdja -- dealing with generalized sequence values during readout with schema
## updated 23 June 2021 by Budhitama Subagdja
## updated 12 May 2021 by Budhitama Subagdja
## from 01 Apr 2020
##-------------------------------------------

from ARTfunc import *
import copy
import json

FRACTION = 0.00001

class FusionART:
	def __init__(self,numspace=0,lengths=[],beta=[],alpha=[],gamma=[],rho=[],schema={},numarray=True):
		self.codes=[]
		self.icode = {'F2':0, 'weights':[]}
		self.alpha = list(alpha)
		self.gamma = list(gamma)
		self.rho = list(rho)
		self.pmcriteria = [1.0]*numspace

		if type(beta) is float:
			self.beta = beta
		if type(beta) is list:
			self.beta = list(beta)
		
		self.lastChoice = []
		self.lastMatch = []
		self.lastActRho = []
		
		self.perfmismatch = False #added to enable the indication of perfect mismatch
		
		self.unCommitCheck = True  #added to enable the check of uncommitted node during resonance search for a field 26082021
		
		self.numarray = numarray
		
		if len(schema) <= 0:
			self.icode = self.initCode(numspace,lengths)
			if len(self.icode['weights']) > 0:
				self.codes.append(self.icode)
			self.schema = {}
		else:
			self.activityF1 = []
			self.schema = copy.deepcopy(schema)
			self.F1FieldsSetup(self.schema)

		#Default functions for choice, match (per field), weight update (every field), resonance search and match (all fields).
		#The functions can be programmatically updated or replaced at runtime.
		#------------------------------------------------------------------------ 
		self.choiceFieldAct = []
		self.updWeight = []
		self.resonanceField = []
		self.matchValField = []

		self.refComplSchema = [] #new added functions for dealing with auto complement coding in schema 24082021

		self.initComplFunction = defaultInitComplVal  #new added functions for dealing with initializing complemented values in sequence
		self.insertSchemaSeqComplVal = insertSchemaComplVal  #new added functions for dealing with inserting complemented values in schema for sequential resonance search
		
		self.setChoiceActFunction()
		self.setResonanceFieldFunction()
		self.setMatchValFieldFunction()
		self.setUpdWeightFunction()

		self.setUpdRefComplFunction() #new added functions for overriding the auto complement coding in schema 24082021

		#self.setupInitComplFuction() #new added functions for overriding the initialization value of a complement 29082021
		#------------------------------------------------------------------------ 		

		self.prevF2Sel = 0
		self.prevUncommit = True

		

	#set the choice function to activate category field
	def setChoiceActFunction(self, cfunction=choiceFieldFuncFuzzy, k=-1):
		if (k >= 0) and (len(self.choiceFieldAct) > k):
			self.choiceFieldAct[k] = cfunction
		else:
			self.choiceFieldAct = [cfunction] * len(self.activityF1)
	#set the weight update function
	def setUpdWeightFunction(self, ufunction=updWeightsFuzzy, k=-1):
		if (k >= 0) and (len(self.updWeight) > k):
			self.updWeight[k] = ufunction
		else:
			self.updWeight = [ufunction] * len(self.activityF1)

	#set the resonance search function
	def setResonanceFieldFunction(self, rfunction=resonanceFieldFuzzy, k=-1):
		if (k >= 0) and (len(self.resonanceField) > k):
			self.resonanceField[k] = rfunction
		else:
			self.resonanceField = [rfunction] * len(self.activityF1)

	#set the match function
	def setMatchValFieldFunction(self, mfunction=matchFuncFuzzy, k=-1):
		if (k >= 0) and (len(self.matchValField) > k):
			self.matchValField[k] = mfunction
		else:
			self.matchValField = [mfunction] * len(self.activityF1)

	#set the refresh complement schema function		
	def setUpdRefComplFunction(self, rcfunction=refreshComplSchema, k=-1):
		if (k >= 0) and (len(self.refComplSchema) > k):
			self.refComplSchema[k] = rcfunction
		else:
			self.refComplSchema = [rcfunction] * len(self.activityF1)

	#set the initialize complement value function
	def setupInitComplFuction(self, icfunction=defaultInitComplVal):
		self.initComplFunction = icfunction

	#set the insert complement schema-based sequential value function
	def setupInsertSchemaComplVal(self, iscfunction=insertSchemaComplVal):
		self.insertSchemaSeqComplVal = iscfunction

	#initialize the code in category (F2) field
	def initCode(self,nspace,lengths,ivalue=0.0, wvalue=1.0):
			iicode = {'F2':0, 'weights':[]}
			self.activityF1=[[]]*nspace #old
			if len(lengths) >= nspace:
				wght = []
				for k in range(len(self.activityF1)):
					self.activityF1[k] = [ivalue]*lengths[k]
					wght.append([wvalue]*lengths[k])
				iicode['weights'] = list(wght)
			return copy.deepcopy(iicode)

	#setting up input F1 field based on the schema representation
	def F1FieldsSetup(self, fschemas):
		actTmp = []
		schmTmp = []
		if len(self.activityF1) <= 0:
			lengths = []
			for i in range(len(fschemas)):
				if 'compl' in fschemas[i]:
					if fschemas[i]['compl']:
						lengths.append(len(fschemas[i]['attrib'])*2)
					else:
						lengths.append(len(fschemas[i]['attrib']))
				else:
					lengths.append(len(fschemas[i]['attrib']))
			self.icode = self.initCode(len(fschemas),lengths)
			if len(self.icode['weights']) > 0:
				self.codes.append(self.icode)
			if len(self.alpha) < len(fschemas):
				self.alpha = [1.0]*len(fschemas)
			if len(self.beta) < len(fschemas):
				self.beta = [1.0]*len(fschemas)
			if len(self.gamma) < len(fschemas):
				self.gamma = [1.0]*len(fschemas)
			if len(self.rho) < len(fschemas):
				self.rho = [1.0]*len(fschemas)
			self.pmcriteria = [1.0]*len(fschemas)
		self.setChoiceActFunction()
		self.setResonanceFieldFunction()
		self.setMatchValFieldFunction()
		self.setUpdWeightFunction()
		self.setUpdRefComplFunction()
		#print('lengths: ', len(self.choiceFieldAct))
		for k in range(len(fschemas)):
			fschema = initFieldSchema(fschemas[k])
			schmTmp.append(fschema)
			factivity = getActivityFromField(fschema)
			self.setActivityF1(factivity,kidx=k)
		self.F1Fields = schmTmp
		
	def buttUpAllF1(self):
		if hasattr(self,'F1Fields'):
			for k in range(len(self.F1Fields)):
				self.setActivityF1(getActivityFromField(self.F1Fields[k]),kidx=k)
	

	def updateF1bySchema(self,fschemas,refresh=True):
		for k in range(len(fschemas)):
			if 'name' in fschemas[k]:
				for kf in self.F1Fields:
					if isSchemaWithAtt(kf,'name',fschemas[k]['name']):
						kf.update(fschemas[k])
						if refresh:
							#kf.update(refreshComplSchema(kf))
							kf.update(self.refComplSchema[k](kf))
		self.buttUpAllF1()

	def updateF1byAttVal(self,attvals,kidx=-1,name='',refresh=True):
		if kidx >= 0:
			self.F1Fields[kidx].update(setSchemabyAttVal(self.F1Fields[kidx],attvals))
		if len(name) > 0:
			for kf in self.F1Fields:
				if isSchemaWithAtt(kf,'name',name):
					kf.update(setSchemabyAttVal(kf,attvals))
					if refresh:
						#kf.update(refreshComplSchema(kf))
						kf.update(self.refComplSchema[self.F1Fields.index(kf)](kf))
		self.buttUpAllF1()
		
	def updateF1byVals(self,vals,kidx=-1,name='',refresh=True):
		if kidx >= 0:
			self.F1Fields[kidx].update(setValFieldSchema(self.F1Fields[kidx],vals))
		if len(name) > 0:
			for kf in self.F1Fields:
				if isSchemaWithAtt(kf,'name',name):
					kf.update(setValFieldSchema(kf,vals))
					if refresh:
						#kf.update(refreshComplSchema(kf))
						kf.update(self.refComplSchema[self.F1Fields.index(kf)](kf))
		self.buttUpAllF1()

	
	def buttUpF1(self,fschema,kidx=-1,fname=''):
		if kidx >= 0:
			self.F1Fields[kidx].update(fschema)
		else:
			if len(fname)>0:
				for k in range(len(self.F1Fields)):
					if 'name' in self.F1Fields[k]:
						if self.F1Fields[k]['name'] == fname:
							self.buttUpF1(fschema,k)
							break
			else:
				for k in range(len(fschema)):
					self.buttUpF1(fschema[k],k)
		for k in range(len(self.F1Fields)):
			self.setActivityF1(getActivityFromField(self.F1Fields[k]),kidx=k)

	def TopDownF1(self):
		F1f = []
		if (len(self.activityF1) > 0) and (len(self.F1Fields)>0):
			for k in range(len(self.activityF1)):
				c = False
				if 'compl' in self.F1Fields[k]:
					c = self.F1Fields[k]['compl']
				self.F1Fields[k].update(readOutVectSym(self.activityF1[k], c))
			F1f = copy.deepcopy(self.F1Fields)
		return F1f

	#new function added 23 June 2021
	def clearActivityF1(self, val=0.0, kidx=None):
		if kidx:
			self.setActivityF1([val]*len(self.activityF1[kidx]), kidx=kidx)
		else:
			for k in range(len(self.activityF1)):
				self.setActivityF1([val]*len(self.activityF1[k]), kidx=k)
		if hasattr(self, 'F1Fields'):
			self.TopDownF1()


	def setActivityF1(self,val,kidx=-1,iidx=-1):
		if kidx > -1:
			if iidx > -1:
				self.activityF1[kidx][iidx] = val
			else:
				self.activityF1[kidx] = list(val)
		else:
			self.activityF1 = list(val)

	# new function added 23 June 2021
	def clearActivityF2(self, val=0.0):
		for j in range(len(self.codes)):
			self.setActivityF2(val, jidx=j)

	def setActivityF2(self,val,jidx=-1):
		if jidx > -1:
			self.codes[jidx]['F2'] = val 
		else:
			print("val, self.codes, len(val), len(self.codes): ", val, ' ', self.codes, ' ', len(val), ' ', len(self.codes))
			assert (len(val) == len(self.codes))
			for j in range(len(val)):
				self.codes[j]['F2'] = val[j]

	def setParam(self,param,value,k=-1):
		if param == "beta":
			if type(value) is float:
				self.beta = value
			if type(value) is list:
				if k>0:
					self.beta[k]=value
				else:
					self.beta=list(value)
		if param == "alpha":
			if k>=0:
				self.alpha[k]=value
			else:
				self.alpha=list(value)
		if param == "gamma":
			if k>=0:
				self.gamma[k]=value
			else:
				self.gamma=list(value)
		if param == "rho":
			if k>=0:
				self.rho[k]=value
			else:
				self.rho=list(value)

	#new 27/05/2020: extF1 argument to allow F1 to be provided externally independent from activityF1;
	#updF2 argument to enable changing the activation value of F2 in the network;
	#return the list of vector values correspond to F2 activation values as the output of the choice function
	def compChoice(self, extF1=None, updF2=True):
		a = self.activityF1
		if extF1:
			a = copy.deepcopy(extF1)
		w = listAttVal(self.codes,'weights')
		if self.numarray:
			a = np.array(a, dtype=object) 
			w = np.array(w, dtype=object)
		F2Values = [0.0] * len(w)
		for j in range(len(w)):
			F2Values[j] = np.sum([self.choiceFieldAct[k](a[k],w[j][k],self.alpha[k],self.gamma[k]) for k in range(len(a))])
		if updF2:
			self.codes = attValList(list(F2Values), self.codes, 'F2')
		return copy.deepcopy(F2Values)
		
			
		
	def expandCode(self):
		tw = []
		for k in range(len(self.activityF1)):
			tw.append([1]*len(self.activityF1[k]))
		self.codes.append({'F2':0, 'weights':list(tw)})
		
	def uncommitted(self,idx):
		for k in range(len(self.codes[idx]['weights'])):
			if self.numarray:
				sumw = np.sum(self.codes[idx]['weights'][k])
			else:
				sumw = 0
				for i in range(len(self.codes[idx]['weights'][k])):
					sumw += self.codes[idx]['weights'][k][i]
			#if sumw <= (len(self.codes[idx]['weights'][k])/2):
			if sumw < len(self.codes[idx]['weights'][k]):
				return False
		return True
		
	#new 27/05/2020: extF2 argument allows max selection based on external list instead of the current F2
	def codeCompetition(self, extF2=None):
		maxact = -1
		c = -1
		if extF2:
			c = np.argmax(extF2)
		else:
			c = np.argmax(listAttVal(self.codes,'F2'))
		return c
		
		
	#new 28/05/2020: extF1 argument allows learning based on external input besides its own activityF1	
	def doLearn(self,j, extF1=None):
		activityF1 = self.activityF1
		if extF1:
			activityF1 = copy.deepcopy(extF1)
		for k in range(len(activityF1)):
			if self.numarray:
				self.codes[j]['weights'][k] = self.updWeight[k](self.beta[k], self.codes[j]['weights'][k], activityF1[k])
			else:
				for i in range(len(activityF1[k])):
					self.codes[j]['weights'][k][i] = self.updWeight[k](self.beta[k], self.codes[j]['weights'][k][i], activityF1[k][i])
				
	#new 28/05/2020: extF1 argument allows learning based on external input besides its own activityF1					
	def doOverwrite(self,j, extF1=None):
		activityF1 = self.activityF1
		if extF1:
			activityF1 = self.activityF1
		for k in range(len(activityF1)):
			self.codes[j]['weights'][k] = list(activityF1[k])
			
	#new 28/05/2020: extF1 argument allows learning based on external input besides its own activityF1				
	def autoLearn(self,j,overwrite=False, extF1=None):
		if self.uncommitted(j):
			overwrite=True
			self.expandCode()
		if overwrite:
			self.doOverwrite(j, extF1=extF1)
		else:
			self.doLearn(j, extF1=extF1)
			
			
	def doReadout(self,j,k,overwrite=True, resetnode=False, resetval=0.0):
		if overwrite:
			self.activityF1[k] = list(self.codes[j]['weights'][k]) #fixed suggested by Hu Yue 03-08-2020
		else:
			if self.numarray:
				self.activityF1[k] = np.amin([self.activityF1[k],self.codes[j]['weights'][k]],axis=0)
			else:
				for i in range(len(self.activityF1[k])):
					self.activityF1[k][i] = min(self.activity[k][i],self.codes[j]['weights'][k][i])
		if resetnode:
			self.setActivityF2(resetval,jidx=j)
					
	def doReadoutAllFields(self, j, overwrite=True, resetnode=False, resetval=0.0):
		for k in range(len(self.activityF1)):
			self.doReadout(j,k,overwrite)
		if resetnode:
			self.setActivityF2(resetval,jidx=j)
			
	#new 23/05/2020
	def doRetrieve(self, j, k=-1, overwrite=True):
		outfield = []
		if k >= 0:
			if overwrite:
				outfield = copy.deepcopy(self.codes[j]['weights'][k])
			else:
				outfield = np.amin([self.activityF1[k],self.codes[j]['weights'][k]],axis=0)
		else:
			if overwrite:
				outfield = copy.deepcopy(self.codes[j]['weights'])
			else:
				outfield = [self.doRetrieve(j,ck,overwrite) for ck in range(len(self.activityF1))] 
		return outfield
	
	#new 23/05/2020
	def doReadoutMax(self, k=-1, F2reset=False, resetval=0.0):
		F2Vect = self.getActivityF2()
		maxjv = max(F2Vect)
		if maxjv > 0:
			J = F2Vect.index(max(maxjv))
			if k >= 0:
				self.doReadout(J,k,resetNode=F2reset)
			else:
				self.doReadoutAllFields(J, resetnode=F2reset)
				
	#new 23/05/2020
	def doRetrieveKMax(self, k=-1, kmax=1, resetval=0.0):
		outseq = []
		kc = 0
		F2Vect = self.getActivityF2()
		for km in range(kmax):
			if max(F2Vect) > 0:
				J = F2Vect.index(max(F2Vect))
				outseq.append(self.doRetrieve(J,k))
				F2Vect[J] = resetval
			else:
				break
		return outseq		


	def isResonance(self, j, rhos=[]):
		crhos = list(self.rho)
		if len(rhos)>0:
			crhos = list(rhos)
		w = self.codes[j]['weights']
		if self.numarray:
			w = np.array(w)
		matched = True
		for k in range(len(self.activityF1)):
			if self.resonanceField[k](self.activityF1[k],w[k],self.rho[k]): #bug fix activityF1 -> self.activityF1
				matched = False
		return matched

	#fixed on 23 June 2021		
	def rhotracking(self, m,fraction):
		return min(m+fraction,1)
	
	#new 27/05/2020: extF1 argument to allow F1 to be provided externally independent from activityF1;
	#updRec argument to enable previous/recent activations, matching, selection, and parameter to be recorded;
	#updF2 argument to enable changing the activation value of F2 in the network;  
	def resSearch(self,mtrack=[],rhos=[],F2filter=[], duprep=False, prevSel=[], extF1=None, updRec=True, updF2=True): 
		activityF1 = self.activityF1
		if extF1:
			activityF1 = copy.deepcopy(extF1)
		resetcode = True
		J = -1
		crhos = list(self.rho)
		if len(rhos)>0:
			crhos = list(rhos)
		if updRec:
			self.lastActRho = list(crhos)
			self.perfmismatch = False		#added to indicate perferct mismatch status 06082021
		choiceV = self.compChoice(extF1=activityF1, updF2=updF2)
		if updRec:
			self.lastChoice = listAttVal(self.codes,'F2')
		while resetcode:
			resetcode = False
			J = self.codeCompetition(extF2=choiceV)
			if J >= 0:
				if updF2:
					self.codes[J]['F2'] = 0
				choiceV[J] = 0
				matches = [self.matchValField[k](activityF1[k], self.codes[J]['weights'][k]) for k in range(len(activityF1))]
				if updRec:
					self.lastMatch = list(matches)
				if not duprep and pmismatch(matches,self.pmcriteria):
					if updRec:
						self.perfmismatch = True #added to indicate perferct mismatch status 06082021
					return J
				if(not mresonance(matches,crhos)) or (J in F2filter) :
					if (not self.uncommitted(J)) or (not self.unCommitCheck): #added to handle match values that can't return 1 (e.g ART2)
						resetcode = True
						for m in range(len(mtrack)):
							if crhos[mtrack[m]] < matches[mtrack[m]]:
								crhos[mtrack[m]] = self.rhotracking(matches[mtrack[m]],FRACTION)
				if updRec:
					self.lastActRho = list(crhos)
				
				if duprep and not resetcode:
					if (J in prevSel) and not self.uncommitted(J):
						resetcode = True
		if J >= 0 and updRec:
			self.prevF2sel = J
			self.prevUncommit = self.uncommitted(J)
		return J

	#new 27/05/2020: extF1 argument to allow F1 to be provided externally independent from activityF1;
	#updRec argument to enable previous/recent activations, matching, selection, and parameter to be recorded;
	#updF2 argument to enable changing the activation value of F2 in the network;  	
	def resSearchPredict(self,mtrack=[],rhos=[],F2filter=[],kmax=1, extF1=None, updRec=True, updF2=True):  #new method to predict through resonance search to find k-best (kmax) matching nodes
		activityF1 = self.activityF1
		if extF1:
			activityF1 = copy.deepcopy(extF1)

		resetcode = True
		J = -1
		kcount = 0
		kJ = []
		crhos = list(self.rho)
		if len(rhos)>0:
			crhos = list(rhos)
		if updRec:	
			self.lastActRho = list(crhos)
		choiceV = self.compChoice(extF1=activityF1, updF2=updF2)
		choiceVlast = list(choiceV)
		if updRec:
			self.lastChoice = listAttVal(self.codes,'F2')

		
		lpredict = [0.0] * len(choiceV)
				
		while resetcode:
			resetcode = False
			J = self.codeCompetition(extF2=choiceV)
			if self.codes[J]['F2'] <= 0:
				break
			if J >= 0:
				if updF2:
					self.codes[J]['F2'] = 0
				choiceV[J] = 0
				matches = [self.matchValField[k](activityF1[k], self.codes[J]['weights'][k]) for k in range(len(activityF1))]
				if updRec:
					self.lastMatch = list(matches)
				if pmismatch(matches,self.pmcriteria):
					resetcode = False
				elif(not mresonance(matches,crhos)) or (J in F2filter):
					if (not self.uncommitted(J)) or (not self.unCommitCheck): #added to handle match values that can't return 1 (e.g ART2)
						resetcode = True
						for m in range(len(mtrack)):
							if crhos[mtrack[m]] < matches[mtrack[m]]:
								crhos[mtrack[m]] = self.rhotracking(matches[mtrack[m]],FRACTION)
				if updRec:
					self.lastActRho = list(crhos)
				
				if not resetcode:
					if not self.uncommitted(J):
						#print('J: ', J)
						kcount += 1
						kJ.append(J)
					if kcount < kmax:
						resetcode = True 
		if len(kJ) > 0 and updRec:
			self.prevF2sel = kJ[0]
			self.prevUncommit = self.uncommitted(kJ[0])
			for i in kJ:
				lpredict[i] = self.lastChoice[i]
		if not updRec:
			for i in kJ:
				lpredict[i] = choiceVlast[i]
		return kJ, lpredict 		
		
	
					
	def displayNetwork(self):
		for j in range(len(self.codes)):
			print('Code: ' + str(j) + ' ' + str(self.codes[j]))
		print ('-----------------------------------------')
		print ('F1: ' + str(self.activityF1))
		
	def displayNetParam(self):
		print('alpha: ' + str(self.alpha))
		print('beta: '+ str(self.beta))
		print('gamma: ' + str(self.gamma))
		print('rho: ' + str(self.rho))

	def expandInput(self,idxs=[],quant=1,ivalue=0,wvalue=0,wvalue_uncommit=1):
		for q in range(quant):
			for i in range(len(idxs)):
				self.activityF1[idxs[i]].append(ivalue)
				#for j in range(len(self.weights)):
				for j in range(len(self.codes)):
					if self.uncommitted(j):
						self.codes[j]['weights'][idxs[i]].append(wvalue_uncommit)
					else:
						self.codes[j]['weights'][idxs[i]].append(wvalue)
				

	def expandInputCompl(self, kidx=-1, ivalue=0, wvalue=0, wvalue_uncommit=1):
		if kidx >= 0:
			if kidx < len(self.activityF1):
				if len(self.activityF1[kidx])%2 == 0:
					midx = int(len(self.activityF1[kidx])/2)
					self.activityF1[kidx].insert(midx,ivalue)
					self.activityF1[kidx].append(self.initComplFunction(val=ivalue))
					for j in range(len(self.codes)):
						if self.uncommitted(j):
							self.codes[j]['weights'][kidx].insert(midx,wvalue_uncommit)
							self.codes[j]['weights'][kidx].append(wvalue_uncommit)
						else:
							self.codes[j]['weights'][kidx].insert(midx,wvalue)
							self.codes[j]['weights'][kidx].append(self.initComplFunction(val=wvalue))
						
					

	def expandInputwSchema(self,kidx=-1,name='',ivalue=0,wvalue=0,attname=''):
		if kidx >= 0:
			ffield = self.F1Fields[kidx]
		if len(name) > 0:
			for ki in range(len(self.F1Fields)):
				if isSchemaWithAtt(self.F1Fields[ki],'name',name):
					ffield = self.F1Fields[ki]
					kidx = ki
		compl = False
		if 'compl' in ffield:
			compl = ffield['compl']
		
		if 'attrib' in ffield:
			if len(attname) > 0:
				ffield['attrib'].append(attname)
			else:
				att = "ax" + str(len(ffield['attrib'])+1)
				ffield['attrib'].append(att)
		if compl:	
			ffield['val'].append(ivalue)
			ffield['vcompl'].append(self.initComplFunction(val=ivalue))
			self.expandInputCompl(kidx=kidx,ivalue=ivalue,wvalue=wvalue)
		else:
			ffield['val'].append(ivalue)
			self.expandInput([kidx],ivalue=ivalue,wvalue=wvalue)
		return ffield



					
	def removeInput(self,k=-1,idx=-1):
		if k >= 0:
			if k < len(self.activityF1):
				if idx >= 0:
					self.activityF1[k].remove(self.activityF1[k][idx])
					for j in range(len(self.codes)):
						self.codes[j]['weights'][k].remove(self.codes[j]['weights'][k][idx])
		return idx
						
	
	def removeInputwSchema(self, k=-1, name=''):
		if k >= 0:
			ffield = self.F1Fields[k]
		if len(name) > 0:
			for ki in range(len(self.F1Fields)):
				if isSchemaWithAtt(self.F1Fields[ki],'name',name):
					ffield = self.F1Fields[ki]
					kidx = ki
		compl = False
		
			
			
	def removeCode(self,idx=-1):
		if idx >= 0:
			if idx < len(self.codes):
				retcode = copy.deepcopy(self.codes[idx])
				self.codes.remove(self.codes[idx])
				return retcode
		return {}

	def getActivityF2(self):
		return listAttVal(self.codes,'F2')


	def gradEncActivateF1(self, k=-1, idx=-1, tau=0.1, tresh=0.0):
		if k>=0:
			if k < len(self.activityF1):
				self.setActivityF1(decayVals(self.activityF1[k], tau, tresh),kidx=k)
				if idx >= 0:
					if idx < len(self.activityF1[k]):
						self.setActivityF1(val=1-tresh,kidx=k,iidx=idx)
						
	def gradComplEnvActivateF1(self, k=-1, idx=-1, tau=0.1, tresh=0.0):
		if k>=0:
			return
						
						
	def stackTopFusionART(self, TopfusART):
		self.TopFusionART = TopfusART
		
	def linkF2TopF1BySchema(self, sNameList):
		if hasattr(self,'TopFusionART'):
			self.toplinkedSchemas = []
			for i in range(len(self.TopFusionART.F1Fields)):
				if self.TopFusionART.F1Fields[i]['name'] in sNameList:
					self.toplinkedSchemas.append(self.TopFusionART.F1Fields[i])
					
	def linkF2TopF1(self, F1idxList, cF1idxList):
		if hasattr(self, 'TopFusionART'):
			self.topIdxList = []
			self.topcIdxList = []
			for k in range(len(self.TopFusionART.activityF1)):
				if k in F1idxList:
					self.topIdxList.append(k)
				if k in cF1idxList:
					self.topcIdxList.append(k)
					
	
	def SequentialResSearch(self, mtrack=[],rhos=[],F2filter=[], tau=0.1, tresh=0.0, maxv=0.9, stau=0.0, accdig=10, duprep=True):
		J = self.resSearch(mtrack, rhos, F2filter)
		if hasattr(self, 'TopFusionART'):
			fTop = self.TopFusionART
			for i in range(len(self.topIdxList)):
				if duprep:
					pSel = [x for x in range(len(fTop.activityF1[self.topIdxList[i]])) if fTop.activityF1[self.topIdxList[i]][x] > 0]
					J = self.resSearch(mtrack, rhos, F2filter, duprep=True, prevSel=pSel)				
				if self.uncommitted(J) and (len(self.codes)>1):
					fTop.expandInput(idxs=[self.topIdxList[i]])
					if len(self.topcIdxList) > 0:
						fTop.expandInput(idxs=[self.topcIdxList[i]])
				if len(self.topcIdxList) > 0:
					v = fTop.activityF1[self.topIdxList[i]]
					cv = fTop.activityF1[self.topcIdxList[i]]
					v, cv = insertComplDecayVals(v, cv, tau=tau, tresh=tresh, idx=J, maxv=maxv, stau=stau, accdig=accdig)
					fTop.setActivityF1(v,kidx=self.topIdxList[i])
					fTop.setActivityF1(cv,kidx=self.topcIdxList[i])
				else:
					v = fTop.activityF1[self.topIdxList[i]]
					v = insertDecayVals(v, tau=tau, tresh=tresh, idx=J, maxv=maxv, accdig=accdig)
					fTop.setActivityF1(v,kidx=self.topIdxList[i])
		return J						
	
	
					
	def SchemaBasedSequentialResSearch(self,mtrack=[],rhos=[],F2filter=[], tau=0.1, tresh=0.0, maxv=0.9, stau=0.0, accdig=10, duprep=True):
		J = self.resSearch(mtrack, rhos, F2filter)
		if hasattr(self, 'TopFusionART'):
			fTop = self.TopFusionART			
			for scm in self.toplinkedSchemas:
				if duprep:
					pSel = [i for i in range(len(scm['val'])) if scm['val'][i] > 0]
					J = self.resSearch(mtrack, rhos, F2filter, duprep=True, prevSel=pSel)
				if self.uncommitted(J) and (len(self.codes)>1):
					fTop.expandInputwSchema(name=scm['name'])
				if scm['compl']:
					self.insertSchemaSeqComplVal(fusART=fTop, schema=scm, tau=tau, tresh=tresh, idx=J, maxv=maxv, stau=stau, accdig=0)
					#v = scm['val']
					#cv = scm['vcompl']
					#v, cv = insertComplDecayVals(v, cv, tau=tau, tresh=tresh, idx=J, maxv=maxv, stau=stau, accdig=accdig)
					#fTop.updateF1bySchema([{'name':scm['name'], 'val':v, 'vcompl':cv}],refresh=False)
					#fTop.updateF1bySchema([{'name':scm['name'], 'val':v}])
				else:
					v = insertDecayVals(scm['val'], tau=tau, tresh=tresh, idx=J, maxv=maxv, accdig=accdig)
					fTop.updateF1bySchema([{'name':scm['name'], 'val':v}])					
		return J
		
						

	def seqTopReadoutToF1(self, tau=0.1, tresh=0.0, stau=0.0, accdig=10, queue=True, overwrite=True):
		if hasattr(self, 'TopFusionART'):
			fTop = self.TopFusionART
			J=-1
			for i in self.topIdxList:
				if len(self.topcIdxList) > 0:
					v = fTop.activityF1[self.topIdxList[i]]
					cv = fTop.activityF1[self.topcIdxList[i]]
					J, v, cv = maxComplReadoutVals(v, cv, tau=tau, tresh=tresh, stau=stau, accdig=accdig, queue=queue)
					fTop.setActivityF1(v,kidx=self.topIdxList[i])
					fTop.setActivityF1(cv,kidx=self.topcIdxList[i])
				else:
					v = fTop.activityF1[self.topIdxList[i]]
					J, v = maxReadoutVals(v, tau=tau, tresh=tresh, accdig=accdig, queue=queue)
					fTop.setActivityF1(v,kidx=self.topIdxList[i])
				self.doReadoutAllFields(J, overwrite=overwrite)
				self.TopDownF1()
			return J

	#updated 29 July 2021 -- adding bpasscomp argument as a flag to bypass complemented values during reading out (when True) to handle generalized values in the sequence
	def seqTopReadoutToF1Schema(self, tau=0.1, tresh=0.0, stau=0.0, accdig=10, queue=True, overwrite=True, bpasscomp=True):
		if hasattr(self, 'TopFusionART'):
			fTop = self.TopFusionART
			J=-1
			for scm in self.toplinkedSchemas: 
				if scm['compl']:
					v = scm['val']
					cv = scm['vcompl']
					J, v, cv = maxComplReadoutVals(v, cv, tau=tau, tresh=tresh, stau=stau, accdig=accdig, queue=queue, bpasscomp=bpasscomp)
					fTop.updateF1bySchema([{'name':scm['name'], 'val':v, 'vcompl':cv}],refresh=False)
					#print('topF1 length j: ', len(self.activityF1[0]))
				else:
					v = scm['val']
					J, v = maxReadoutVals(v, tau=tau, tresh=tresh, accdig=accdig, queue=queue)
					fTop.updateF1bySchema([{'name':scm['name'], 'val':v}])
				self.doReadoutAllFields(J, overwrite=overwrite)
				self.TopDownF1()
			return J



def saveFusionARTNetwork(nnet, name='fart.net'):
	fartnet = {'file_name': name, 
			'codes': nnet.codes,
			'alpha': nnet.alpha,
			'beta': nnet.beta,
			'gamma': nnet.gamma,
			'rho': nnet.rho,
			'lastChoice': nnet.lastChoice,
			'lastMatch': nnet.lastMatch,
			'lastActRho': nnet.lastActRho,
			'activityF1': nnet.activityF1
			}
	if hasattr(nnet, 'F1Fields'):
		fartnet['F1Fields'] = nnet.F1Fields
	with open(name, 'w') as outfile:
		json.dump(fartnet, outfile)

def loadFusionARTNetwork(nnet, name='fart.net'):
	with open(name) as json_file:
		fartnet = json.load(json_file)
	nnet.codes = fartnet['codes']
	nnet.alpha = fartnet['alpha']
	nnet.beta = fartnet['beta']
	nnet.gamma = fartnet['gamma']
	nnet.rho = fartnet['rho']
	nnet.lastChoice = fartnet['lastChoice']
	nnet.lastMatch = fartnet['lastMatch']
	nnet.lastActRho = fartnet['lastActRho']
	nnet.activityF1 = fartnet['activityF1']
	if 'F1Fields' in fartnet:
		nnet.F1Fields = fartnet['F1Fields']
			