# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Rahul Kunji (rahulsk2@illinois.edu) on 01/16/2019

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import collections
import heapq
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)


class Heuristic:
    def __init__(self,goals, mapsize):
        self.vertices = goals
        self.edges = []
        self.mapsize = mapsize

    def addEdge(self,u,v,w):
        self.edges.append([u,v,w])

    def find(self,node,parent):
        if parent[node] == node:
            return node
        return self.find(parent[node], parent)

    def union(self,u,v,rank, parent):
        uroot = self.find(u, parent)
        vroot = self.find(v, parent)

        if rank[uroot] < rank[vroot]:
            parent[uroot] = vroot
        elif rank[uroot] > rank[vroot]:
            parent[vroot] = uroot
        else:
            parent[uroot] = vroot
            rank[vroot] += 1

    def KruskalMST(self, current_state):
        Vertex = set(self.vertices.copy())
        Vertex.add(current_state)
        queue = []
        rank = {}
        parent = {}
        for n in Vertex:
            parent[n] = n
            rank[n] = float('inf')
            for d in Vertex:
                if n == d:
                    continue
                heapq.heappush(queue,(total_heuristic(n,d, self.mapsize),n,d))

        parent[current_state] = current_state
        rank[current_state] = 0

        while len(self.edges) + 1 < len(Vertex) :
            min= heapq.heappop(queue)
            x = self.find(min[1], parent)
            y = self.find(min[2], parent)
            if x == y:
                continue
            self.addEdge(min[1],min[2],total_heuristic(min[2],min[1], self.mapsize))
            self.union(x, y, rank, parent)
        result = 0
        for i in self.edges:
            result += i[2]
        return result

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
    }.get(searchMethod)(maze)

def getDist(a,b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def new_heuristic(curr,goal):
    a = abs(curr[0] - goal[0])
    b = abs(curr[1] - goal[1])
    return a+b+((a*a+b*b)**0.5)

def second_heuristic(curr,goal):
    a = abs(curr[0] - goal[0])
    b = abs(curr[1] - goal[1])
    return a+b+((a*a+b*b)**0.5)*0.618

def total_heuristic(curr, goal, mapsize):
    if mapsize < 110:
        return getDist(curr, goal)
    if mapsize < 400:
        return second_heuristic(curr, goal)
    return new_heuristic(curr, goal)

def bfs(maze):
    # TODO: Write your code here

    start = maze.getStart()
    goal = maze.getObjectives()

    explored = []

    queue = [[start]]

    if start in goal:
        return [start], 1

    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node not in explored:
            neighbors = maze.getNeighbors(node[0], node[1])
            explored.append(node)
            for neighbor in neighbors:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
                if neighbor in goal:
                    return new_path, len(explored)



def dfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    start = maze.getStart()
    goal = maze.getObjectives()

    if start in goal:
        return [start], 1
    stack = collections.deque([[start]])
    explored = []
    while stack:
        path = stack.pop()
        node = path[-1]
        if node in explored:
            continue
        explored.append(node)
        neighbors = maze.getNeighbors(node[0], node[1])
        for neighbor in neighbors:
            if neighbor not in explored:
                new_path = list(path)
                new_path.append(neighbor)
                stack.append(new_path)

            if neighbor in goal:
                return new_path, len(explored)


def greedy(maze):
    # TODO: Write your code here
    start = maze.getStart()
    goal = maze.getObjectives()
    if start in goal:
        return [start], 1
    explored, queue = [], []
    heapq.heappush(queue, (getDist(start, goal[0]), [start]))
    while queue:
        path = heapq.heappop(queue)[1]
        node = path[-1]
        if node in explored:
            continue
        explored.append(node)
        neighbors = maze.getNeighbors(node[0], node[1])
        for check in neighbors:
            dist = getDist(check, goal[0])
            if check not in explored:
                newpath = list(path)
                newpath.append(check)
                heapq.heappush(queue, (dist, newpath))
            if check in goal:
                return newpath, len(explored)
    # return path, num_states_explored
    # return [], 0

def calTotalDist(current, goals):
    total_distance = 0
    for goal in goals:
        total_distance += getDist(current, goal)
    return total_distance

def calMinDist(current, goals):
    current_min = getDist(current, goals[0])
    for goal in goals:
        if getDist(current, goal) < current_min:
            current_min = getDist(current, goal)
    return current_min

def calState(goals, child, parent_state):
    if child in goals:
        index = goals.index(child)
        bit_map = list(parent_state[1])
        bit_map[index] = '1'
        bit_map = ''.join(bit_map)
    else:
        bit_map = parent_state[1]

    return (child, bit_map)

def getRemainingGoals(goals, bit_map):
    remaining_goals = list()

    for i, n in enumerate(list(bit_map)):
        if n == '0':
            remaining_goals.append(goals[i])

    return remaining_goals


def astar(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    queue = []
    start_node = maze.getStart()
    goals = maze.getObjectives()
    mapsize = maze.getDimensions()[0] * maze.getDimensions()[1]
    #print(mapsize)
    if len(goals) == 1:
        return astar_onegoal(maze)
    start_state = (start_node, '0'*len(goals))         # state (node, bit map of goals)
    heapq.heappush(queue, (0, (start_state, [start_node])))         # heap element (priority, (state, path))
    cost_so_far = {start_state: 0}                  # state -> cost

    explored = set()

    while queue:
        current_heap_elem = heapq.heappop(queue)
        current_state = current_heap_elem[1][0]
        current_path = current_heap_elem[1][1]
        current_node = current_path[-1]
        neighbors = maze.getNeighbors(current_node[0], current_node[1])
        explored.add(current_node)

        for neighbor in neighbors:
            neighbor_state = calState(goals, neighbor, current_state)
            new_cost = cost_so_far[current_state] + 1
            new_path = list(current_path)
            new_path.append(neighbor)

            # print(neighbor_state)

            if neighbor_state[1].find('0') == -1:
                #print("new Path:", new_path)
                return new_path, len(explored)

            if neighbor_state not in cost_so_far or cost_so_far[neighbor_state] > new_cost:
                remaining_goals = getRemainingGoals(goals, neighbor_state[1])
                #print("neighbor:", neighbor)
                #print("remaining_goals:", remaining_goals)
                heuristic = Heuristic(remaining_goals, mapsize)
                #print("heuristic:", heuristic)
                priority = new_cost + heuristic.KruskalMST(neighbor_state[0])
                cost_so_far[neighbor_state] = new_cost
                heapq.heappush(queue, (priority, (neighbor_state, new_path)))

def bigdots(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    queue = []
    start_node = maze.getStart()
    goals = maze.getObjectives()
    start_state = (start_node, '0'*len(goals))         # state (node, bit map of goals)
    heapq.heappush(queue, (0, (start_state, [start_node])))         # heap element (priority, (state, path))
    cost_so_far = {start_state: 0}                  # state -> cost

    explored = set()

    while queue:
        current_heap_elem = heapq.heappop(queue)
        current_state = current_heap_elem[1][0]
        current_path = current_heap_elem[1][1]
        current_node = current_path[-1]
        neighbors = maze.getNeighbors(current_node[0], current_node[1])
        explored.add(current_node)

        for neighbor in neighbors:
            neighbor_state = calState(goals, neighbor, current_state)
            new_cost = cost_so_far[current_state] + 1
            new_path = list(current_path)
            new_path.append(neighbor)

            # print(neighbor_state)

            if neighbor_state[1].find('0') == -1:
                #print("new Path:", new_path)
                return new_path, len(explored)

            if neighbor_state not in cost_so_far or cost_so_far[neighbor_state] > new_cost:
                remaining_goals = getRemainingGoals(goals, neighbor_state[1])
                #print("neighbor:", neighbor)
                #print("remaining_goals:", remaining_goals)
                # heuristic = Heuristic(remaining_goals)
                heuristic = calTotalDist(neighbor_state[0], remaining_goals)
                #print("heuristic:", heuristic)
                # priority = new_cost + heuristic.KruskalMST(neighbor_state[0])
                priority = new_cost + heuristic
                cost_so_far[neighbor_state] = new_cost
                heapq.heappush(queue, (priority, (neighbor_state, new_path)))

def astar_onegoal(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    start = maze.getStart()
    goal = maze.getObjectives()
    if goal in start:
        return [start], 1
    explored, queue = [], []
    heapq.heappush(queue, (getDist(start, goal[0]), [start]))
    while queue:
        path = heapq.heappop(queue)[1]
        node = path[-1]
        if node in explored:
            continue
        explored.append(node)
        neighbors = maze.getNeighbors(node[0], node[1])
        for neighbor in neighbors:
            dist = getDist(neighbor, goal[0])
            if neighbor not in explored:
                newpath = list(path)
                newpath.append(neighbor)
                heapq.heappush(queue, (dist + len(newpath), newpath))
            if neighbor in goal:
                return newpath, len(explored)