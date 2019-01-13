import numpy as np
import json
import matplotlib.pyplot as plt
import time
import networkx as nx
import pandas as pd

from random import *

map = []
scores = [0 for i in range(101)]
learning_rate = 0.5
nodes = [[] for i in range(101)]
G = nx.Graph()

def make_random_graph():
    for i in range(101):
        seed(time.time())
        source = randint(0,100)
        seed(time.time())
        destin = randint(0,100)
        G.add_node(source,weight=0.5)
        G.add_node(destin)
        map.append([source, destin])
        G.add_edge(source, destin)
        G.add_edge(destin, source)

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
        if scores[i] <= 6:
            scores[i] = 0

    scores = np.add(scores, -lower_bound)
    scores = np.divide(scores, upper_bound)
    scores = np.multiply(scores, 100)
    scores = [int(i) for i in scores]

    return scores


if __name__ == "__main__":
    make_random_graph()
    print(map)
    print(nodes)
    step_scores = {}
    # And a data frame with characteristics for your nodes

    for i in range(30):
        seed(time.time())
        t = randint(0, 100)
        propaganda(100, t)
        print(t)
        scores = score_normalize(scores)
        print(scores)
        step_scores[i] = scores
        carac = pd.DataFrame({'ID': range(101), 'myvalue': scores})

        G.nodes()

        carac = carac.set_index('ID')
        carac = carac.reindex(G.nodes())

        # Plot it, providing a continuous color scale with cmap:
        fig = plt.figure()
        fig.set_facecolor("#7ed102")
        nx.draw(G, with_labels=True, node_color=carac['myvalue'], edge_color ='black', cmap=plt.cm.Blues)
        #plt.show()
        plt.savefig("figure/"+str(i) +"_step.png",facecolor=fig.get_facecolor())
        plt.gca()
        plt.cla()
        plt.close()


    with open("ways.json", "w") as prop:
        json.dump(map,prop)
    with open("nodes.json", "w") as prop:
        json.dump(nodes,prop)
    with open("scores.json", "w") as prop:
        json.dump(step_scores,prop)

