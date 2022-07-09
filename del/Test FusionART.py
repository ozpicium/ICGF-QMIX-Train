from fusionART import *

"""
A Fusion ART object could be created by specifying the number of F1 fields (numspace), length of each field (lengths), 
beta, alpha, gamma, and rho parameters for every field.
"""
fa = FusionART(numspace=3,
               lengths=[4, 4, 2],
               beta=[1.0, 1.0, 1.0],
               alpha=[0.1, 0.1, 0.1],
               gamma=[1.0, 1.0, 1.0],
               rho=[0.2, 0.2, 0.5])
fa.displayNetwork()

"""setActivityF1 method is to provide the input to every F1 field of fa. It will store the value to activityF1 
property of fa """
fa.setActivityF1([
    [1.0, 0, 0, 1.0],
    [0, 0, 0, 1.0],
    [0.6, 0.4]
])
print("set ActivityF1 to ", fa.activityF1)

"""
resonance search (resSearch) is to select a node in F2. In this case, J is the index of the selected node in F2 
(let's ignore the mtrack argument for a while). 
The node selected can uncommitted which mean it failed to find a matching code in F2. 
It's checked with uncommitted method. Once the node J is selected, the input vectors in F1 can be learned to associate 
with J with autoLearn method. The structure of the network can be shown by displayNetwork method 
"""

print("resonance search: ")
J = fa.resSearch()
print("selected ", J)
if fa.uncommitted(J):
    print('uncommitted')
fa.autoLearn(J)  # learning the parameters of the selected code with the input vector.
fa.displayNetwork()

"""
Different input patterns can be given subsequently to retrieve nodes or to be stored in the network as shown
"""
# Input another to F1 field.
fa.setActivityF1([
    [0, 1.0, 1.0, 0],
    [1.0, 0, 0, 0],
    [0.3, 0.7]
])

print("set ActivityF1 to ", fa.activityF1)
print("resonance search: ")
J = fa.resSearch(mtrack=[0])
print("selected ", J)
if fa.uncommitted(J):
    print('uncommitted')
fa.autoLearn(J)
fa.displayNetwork()

print("")
fa.setActivityF1([
    [1.0, 0, 0, 1.0],
    [0, 0, 0, 1.0],
    [0.6, 0.4]
])
print("set ActivityF1 to ", fa.activityF1)
print("resonance search: ")
J = fa.resSearch(mtrack=[0])
print("selected ", J)
if fa.uncommitted(J):
    print('uncommitted')
fa.autoLearn(J)
fa.displayNetwork()
