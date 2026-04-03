import heapq

class Node:
    def __init__(self, position,parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f


def a_star(start,end,grid):
    open_list = []
    closed_list = set()
    start_node = Node(start)
    end_node = Node(end)

    heapq.heappush(open_list,(start_node.f,start_node))

    while open_list:
        current  = heapq.heappop(open_list)[1]
        closed_list.add(current.position)
        if current.position == end_node.position:
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            return path[::-1]
        x , y = current.position
        neighbors = [(x-1,y), (x+1,y),(x,y-1),(x,y+1)]

        for nx , ny in neighbors:
            if 0<=nx<len(grid) and 0<=ny<len(grid[0]) and grid[nx][ny]==0 and (nx,ny) not in closed_list:
                neighbor_node = Node((nx,ny),current)
                neighbor_node.g = current.g + 1
                neighbor_node.h = abs(nx - end_node.position[0]) + abs(ny - end_node.position[1])
                neighbor_node.f = neighbor_node.g + neighbor_node.h
                heapq.heappush(open_list,(neighbor_node.f,neighbor_node))

    return None   
            
grid = [[0,0,0,0],[1,1,0,1],[0,0,0,0],[0,1,1,0]] 
start = (0,0)
end = (3,3)
path = a_star(start,end,grid)
print("Path: ",path)