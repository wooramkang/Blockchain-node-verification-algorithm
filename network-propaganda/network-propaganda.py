import numpy as np
from random import *

map = []
scores = [0 for i in range(101)]
learning_rate = 0.5
nodes = [[] for i in range(101)]

def make_random_graph():
    for i in range(100):
        source = randint(0,100)
        destin = randint(0,100)
        map.append([source, destin, 0])
        if destin not in nodes[source]:
            nodes[source].append(destin)
        if source not in nodes[destin]:
            nodes[destin].append(source)

def find_score(target_node):
    return scores[target_node]

def propaganda(score, target_node):

    visited = [target_node]
    queue = [target_node]

    scores[target_node] = scores[target_node] + score

    while(queue):
        vertex = queue.pop()
        score = learning_rate * score
        for i in nodes[vertex]:
            if i not in visited:
                queue.append(i)
                visited.append(i)
                scores[i] = scores[i] + score

    return 0

def score_normalize(scores):

    upper_bound = max(scores)
    lower_bound = min(scores)

    for i in range(len(scores)):
        if scores[i] <= 12:
            scores[i] = 0

    scores = np.add(scores, -lower_bound)
    scores = np.divide(scores, upper_bound)
    scores = np.multiply(scores, 100)
    scores = [int(i) for i in scores]

    return scores


if __name__ == "__main__":
    make_random_graph()
    #print(map)
    #print(nodes)
    for i in range(100):
        t = randint(0, 100)
        propaganda(100, t)
        scores = score_normalize(scores)
        print(scores)