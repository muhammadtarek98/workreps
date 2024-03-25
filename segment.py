from time import perf_counter

import numpy as np

from tools import check_stacks


def add_to_stack(index, n_donors, donors, stack):
    # This recursive function adds to the stack the donors of an outlet, and then the donors of these
    # donors, etc. Until an entire catchment is added to the stack

    stack.append(index)  # add nodes to the stack

    for k in range(n_donors[index]):  # find donors and add them (if any) to the stack
        add_to_stack(donors[index, k], n_donors, donors, stack)


def add_to_stack_bw(index, delta, Di, stack, local_maximum):
    # This recursive function adds to the stack the donors of an outlet, and then the donors of these
    # donors, etc. Until an entire catchment is added to the stack

    stack.append(index)  # add nodes to the stack

    for k in range(delta[index], delta[index + 1]):  # find donors and add them (if any) to the stack
        if Di[k] != local_maximum:  # avoid infinite loop
            add_to_stack_bw(Di[k], delta, Di, stack, local_maximum)


def philippe_steer_stack_building(receivers, local_maximum_indexes, knn):
    start = perf_counter()
    n_points = len(receivers)

    # identify the donors for each receiver
    ndon = np.zeros(n_points, dtype=int)  # number of donors
    donor = np.zeros((n_points, knn), dtype=int)  # donor list
    for k, receiver in enumerate(receivers):
        if receiver != k:
            ndon[receiver] = ndon[receiver] + 1
            donor[receiver, ndon[receiver] - 1] = k

    # build the stacks
    labels = np.zeros(n_points, dtype=int)
    labelsnpoint = np.zeros(n_points)
    stacks = []

    for k, ij in enumerate(local_maximum_indexes):
        stack = []
        add_to_stack(ij, ndon, donor, stack)  # recursive function
        stacks.append(stack)
        labels[stack] = k
        labelsnpoint[stack] = len(stack)

    end = perf_counter()
    print(f'Philippe Steer {end - start}')

    return stacks, labels, labelsnpoint, ndon


def braun_willett_stack_building(receivers, local_maximum_indexes):
    start = perf_counter()
    n_points = len(receivers)

    # get the number of donors per receiver and build the list of donors per receiver
    di = np.zeros(n_points, dtype=int)  # number of donors
    Dij = [[] for i in range(n_points)]  # lists of donors
    for k, receiver in enumerate(receivers):
        di[receiver] = di[receiver] + 1
        Dij[receiver].append(k)

    # build Di, the list of donors
    Di = np.zeros(n_points, dtype=int)  # list of donors
    idx = 0
    for list_ in Dij:  # build the list of donors
        for point in list_:
            Di[idx] = point
            idx = idx + 1

    # build delta, the index array
    delta = np.zeros(n_points + 1, dtype=int)  # index of the first donor
    delta[n_points] = n_points
    for i in range(n_points - 1, -1, -1):
        delta[i] = delta[i + 1] - di[i]
    stacks = []
    labels = np.zeros(n_points, dtype=int)
    points_per_label = np.zeros(n_points, dtype=int)

    # build the stacks
    for k, ij in enumerate(local_maximum_indexes):
        stack = []
        add_to_stack_bw(ij, delta, Di, stack, ij)  # recursive function
        stacks.append(stack)
        labels[stack] = k
        points_per_label[stack] = len(stack)

    end = perf_counter()
    print(f'Braun Willet {end - start}')

    return stacks, labels, points_per_label, di


def segment_labels(xyz, knn, neighbors_indexes, braun_willett=True):
    print('[segment_labels]')

    x, y, z = np.split(xyz, 3, axis=1)
    n_points = len(xyz)
    dx = x - np.squeeze(x[neighbors_indexes])  
    dy = y - np.squeeze(y[neighbors_indexes])
    dz = z - np.squeeze(z[neighbors_indexes])
    slopes = dz / (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5 
    index_of_min_slope = np.argmin(slopes, axis=1)  
    min_slope = np.amin(slopes, axis=1)  
    receivers = neighbors_indexes[np.arange(n_points), index_of_min_slope]

    local_maximum_indexes = np.where(min_slope >= 0)[0]
    receivers[local_maximum_indexes] = local_maximum_indexes

    if braun_willett:
        stacks, labels, labelsnpoint, ndon = braun_willett_stack_building(receivers, local_maximum_indexes)
    else:
        stacks, labels, labelsnpoint, ndon = philippe_steer_stack_building(receivers, local_maximum_indexes, knn)

    nlabels = len(local_maximum_indexes)
    print(nlabels)
    print(len(stacks))
    #if check_stacks(stacks, len(labels)):
    #    print("[segment_labels] stacks are valid")

    return labels, nlabels, labelsnpoint, stacks, ndon, local_maximum_indexes
