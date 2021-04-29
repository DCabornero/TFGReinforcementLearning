import sys, os

mypath = os.getcwd() + '/ItemBandits'
if mypath not in sys.path:
    sys.path.append(mypath)
mypath = os.getcwd() + '/ItemBandits/abstract'
if mypath not in sys.path:
    sys.path.append(mypath)

from EpsilonGreedy import EpsilonGreedy
from Analysis import Analysis

bandit = EpsilonGreedy()
bandit.read_csv('data/ratings.csv')

an = Analysis()
an.execute([bandit],epochs=100000)
an.recall()
