import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
from typing import TypeVar
from utils import sort_counterclockwise, draw, angle

Vertices = TypeVar("Vertices", list, np.ndarray)
ANGLE_THRESHOLD = -0.2

img = cv2.imread(r"C:\Users\ASUS\Downloads\vase.jpg", cv2.IMREAD_GRAYSCALE)
cannyimg = cv2.Canny(img, 5, 125)
# d = cv2.cvtColor(cannyimg, cv2.COLOR_GRAY2BGR)

img2 = cannyimg.astype("bool")

SEGMENTED_IMG = np.copy(img2).astype("int")
img5 = cv2.dilate(cannyimg.copy(), (3, 3))
plt.imshow(img5)
plt.show()
contours, _ = cv2.findContours(cv2.dilate(cannyimg.copy(), (3, 3)), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i, c in enumerate(contours):
    x, y, w, h = cv2.boundingRect(c)
    rect = img2[y : y + h, x : x + w]
    if rect.sum() < 10:
        SEGMENTED_IMG[y : y + h, x : x + w][rect] = 0
#     else:

#         SEGMENTED_IMG[y : y + h, x : x + w][rect] = i
#         for p in c:
#             x, y = p[0]
#             SEGMENTED_IMG[y, x] = -1
img2 = SEGMENTED_IMG.astype("bool")

ROWS, COLS = img2.shape
is_boundary = lambda y, x: (y == 0 or y == ROWS - 1) or (x == 0 or x == COLS - 1)
IMAGE_VERTICES = (0, 0), (0, COLS - 1), (ROWS - 1, 0), (ROWS - 1, COLS - 1)
get_index = lambda y, x: y * COLS + x
get_points = lambda v: [VERTICES_LIST[i][0] for i in v[1]]


plt.imshow(SEGMENTED_IMG)
plt.show()


def clear(v: int):
    """delete v from VERTICES_LIST"""
    VERTICES_LIST[v] = []


def create(x: int, y: int):
    """adding y into x's neighbor list"""
    VERTICES_LIST[x][1].append(y)


def remove(x: int, y: int):
    """remove y in x's neighbor list"""
    VERTICES_LIST[x][1].remove(y)



def mesh_construction(image: np.ndarray):
    """create triangles
    o-----------o
    |         / |
    |       /   |
    |    /      |
    | /         |
    o-----------o"""

    for y in range(ROWS):
        for x in range(COLS):
            edge = -1
            is_feature = image[y, x]
            if not (x == COLS - 1 or y == ROWS - 1):
                x2 = x + 1
                y2 = y + 1
                # edge: 0 means top left to bottom right,
                #       1 means top right to botom left
                if is_feature and image[y2, x2]:
                    edge = 1
                elif image[y2, x] and image[y, x2]:
                    edge = 0
                else:
                    edge = np.random.choice([0, 1])
            # print(y, x)
            kernel(y, x, edge, is_feature)
    return


def kernel(y, x, edge, is_feature):
    n = y * COLS + x
    neighbours = []
    # edge 0 means lower right is neighbour
    if edge == 0:
        neighbours.append(n + COLS + 1)

    # Check boundaries
    if y != 0:
        neighbours.append(n - COLS)
    if y != ROWS - 1:
        neighbours.append(n + COLS)
    if x != 0:
        neighbours.append(n - 1)
    if x != COLS - 1:
        neighbours.append(n + 1)

    # checks if upper px has edge 1,
    # if it does it means upper right is neighbour
    if y != 0 and x != COLS - 1 and VERTICES_LIST[n - COLS][-1] == 1:
        neighbours.append(n - COLS + 1)

    # check if upper left px has edge 0,
    # if it does it means upper left is neighbour
    if y != 0 and x != 0 and VERTICES_LIST[n - COLS - 1][-1] == 0:
        neighbours.append(n - COLS - 1)

    # checks if left px has edge 1,
    # if it does it means lower left is neighbour
    if x != 0 and y != ROWS - 1 and VERTICES_LIST[n - 1][-1] == 1:
        neighbours.append(n + COLS - 1)

    VERTICES_LIST[n] = [(y, x), neighbours, is_feature, edge]


def collapse_condition(a: Vertices, b: Vertices):
    """The whole image has four vertices that
    should be fixed in the process of collapse.
    Vertices in the same boundary can collapse to
    each other. In one collapse, if one vertex is in
    boundary and another is not, then it
    cannot be active vertex.

    Two feature vertices with the same feature
    can collapse to each other. However, if
    collapse happens between feature vertices
    and general vertices, feature vertices must
    be passive. General vertices can be active or passive.

    Collapse does not allow two-edge cross which is called
    overlap; the way of judging whether there is overlap or not
    is based on the following criteria (sort-edge method). As
    shown in Figure 4, the authors suppose that vertex a 
    collapses to vertex b. A coordinated system is created 
    with a as the origin. The anticlockwise order of the
    neighbors of a (except passive vertex aka b) is  {g, c, d, e, f}.
    After the collapse, the authors use vertex b as the origin to
    create a coordinate system and to judge if the anticlockwise.
    
    A stricter limit for the collapse is to avoid
    sharp triangles due to their many disadvantages."""

    y1, x1 = a[0]
    y2, x2 = b[0]
    if a[0] in IMAGE_VERTICES:
        return False
    if a[-2] == True and b[-2] == False:
        return False
    # if SEGMENTED_IMG[y1, x1] == -1:
    #     return False
    if is_boundary(y1, x1):
        if not is_boundary(y2, x2):
            return False
        if not (y1 == y2 or x1 == x2):
            return False
    

    ib = get_index(*b[0])
    ia = get_index(*a[0])
    neighbour_a = [VERTICES_LIST[na][0] for na in a[1] if na != ib]

    # if a[-2] == True and b[-2] == True:
        # if SEGMENTED_IMG[y1, x1] != SEGMENTED_IMG[y2, x2]:
        #     return False  
        # if any(math.cos(angle(a[0], b[0], VERTICES_LIST[na][0])) < ANGLE_THRESHOLD for na in a[1]):
        #     return False
    f = sort_counterclockwise(neighbour_a, b[0])
    if f != 0 and f == sort_counterclockwise(neighbour_a, a[0]):
        neighbour_b = list(set(neighbour_a + [VERTICES_LIST[nb][0] for nb in b[1] if nb != ia]))
        # print(neighbour_b, b[0])
        if sort_counterclockwise(neighbour_b, b[0]) == 0:
            # for yy, xx in neighbour_b:
                # plt.plot([b[0][1], xx], [b[0][0], yy], color="blue")
            # plt.plot([x for _, x in counterclockwise_points], [y for y, _ in counterclockwise_points], color="green")
            # plt.show()
            return False
        # print(sort_counterclockwise(neighbour_b, b[0]))
        return True
    return False


REMOVED_VERTICE = 0


def collapse(va: Vertices, vb: Vertices):
    global REMOVED_VERTICE
    """vertices a collapses to vertices b.
    a is active; b is pasive."""

    index_a = get_index(*va[0])
    index_b = get_index(*vb[0])
    remove(index_b, index_a)

    REMOVED_VERTICE += 1

    for na in va[1]:
        if na == index_b:
            continue
        # remove vertice a
        remove(na, index_a)
        if na in vb[1]:
            # this vertice is shared by vertice b
            continue
        # add vertice b to a's neighbour list
        # and vice versa
        create(index_b, na)
        create(na, index_b)

    clear(index_a)


def mesh_simplification(k):
    for i in range(k):
        print(i)
        for j in start_order:
            v = VERTICES_LIST[j]
            if not v:
                continue
            n = VERTICES_LIST[np.random.choice(v[1])]

            if collapse_condition(v, n):
                collapse(v, n)
        print(i)


VERTICES_LIST = np.empty(shape=ROWS * COLS, dtype=object)
print(VERTICES_LIST.shape)

import time

start = time.monotonic()
mesh_construction(img2)
print(time.monotonic() - start)

start_order = np.arange(ROWS * COLS)
# np.random.shuffle(start_order)

mesh_simplification(40)

# sort out empty lists
P = VERTICES_LIST[VERTICES_LIST.astype("bool")]

plt.imshow(img2, cmap="binary")

with np.printoptions(threshold=np.inf):
    print(P)

print(REMOVED_VERTICE)
print(len(VERTICES_LIST))

for i in P:
    y, x = i[0]
    for yy, xx in [VERTICES_LIST[n][0] for n in i[1]]:
        # if img2[yy, xx] and i[-2]:
            # plt.plot([x, xx], [y, yy], color="black")
        # else:
            plt.plot([x, xx], [y, yy], color="blue")



plt.show()
