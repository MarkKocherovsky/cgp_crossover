import numpy as np
from numpy import random


def uniform_xover(parents, max_r, p_xov):
    children = []
    retention = []
    d_distro = np.zeros((len(parents), max_r))
    for i in range(0, len(parents), 2):
        if random.random() >= p_xov:
            children.append(parents[i].copy())
            children.append(parents[i + 1].copy())
        for _ in [1, 2]:  #two children
            p1 = parents[i].copy()
            p2 = parents[i + 1].copy()

            inst_counts = [len(p1), len(p2)]  #count instructions
            samples = [random.choice(inst_counts[0], size=(int(len(p1) / 2),), replace=False),
                       random.choice(inst_counts[1], size=(int(len(p2) / 2),), replace=False)]  #get instruction indices
            for s in samples:
                for n in s:
                    d_distro[i, n] += 1
                    d_distro[i+1, n] += 1

            samples_norm = [samples[0] / inst_counts[0], samples[1] / inst_counts[1]]  #normalize indices

            p1_list = samples[0]
            p2_list = samples[1]
            fu_list = np.concatenate((p1_list, p2_list))
            no_list = np.concatenate((samples_norm[0], samples_norm[1]))

            c1 = np.concatenate((p1[p1_list], p2[p2_list]), axis=0)
            #c2 = np.concatenate((p1[-p1_list], p2[-p2_list]), axis = 0)
            #https://stackoverflow.com/questions/9007877/sort-arrays-rows-by-another-array-in-python
            fu_list = no_list.argsort()  #sort by normalized order
            c = c1[fu_list, :]
            retention.append(i)
            children.append(c)
    return children, np.array(retention).astype(np.int32), d_distro


def onepoint_xover(parents, max_r, p_xov, fixed_length):  # 1 point crossover
    children = []
    retention = []
    d_distro = np.zeros((len(parents), max_r))
    for i in range(0, len(parents), 2):
        p1 = parents[i].copy()
        p2 = parents[i + 1].copy()
        if random.random() > p_xov:
            children.append(p1)
            children.append(p2)
            continue
        retention.append(i)
        inst_counts = [len(p1), len(p2)]
        if not fixed_length:  # Normal LGP
            try:
                def select_index(length):
                    return 1 if length == 2 else random.randint(1, length - 1)

                s1 = select_index(p1.shape[0])
                s2 = select_index(p2.shape[0])

                s = [s1, s2]
                d_distro[i, s[0]] += 1
                d_distro[i + 1, s[1]] += 1

            except Exception as e:
                s = [0, 0]
                print("Error in crossover:", e)
                print("p1:", p1)
                print("p2:", p2)
        else:
            s_temp = random.randint(1, max_r)
            d_distro[i, s_temp] += 1
            d_distro[i+1, s_temp] += 1
            s = [s_temp, s_temp]

        p1_list_front = p1[:s[0]].copy()
        p1_list_back = p1[s[0]:].copy()

        p2_list_front = p2[:s[1]].copy()
        p2_list_back = p2[s[1]:].copy()

        c1 = np.concatenate((p1_list_front, p2_list_back), axis=0)
        c2 = np.concatenate((p2_list_front, p1_list_back), axis=0)

        if c1.shape[0] > max_r:  #keep to maximimum rule size!
            idxs = np.array(range(0, max_r))
            to_del = random.choice(idxs, ((c1.shape[0] - max_r),), replace=False)
            c1 = np.delete(c1, to_del, axis=0)
        if c2.shape[0] > max_r:
            idxs = np.array(range(0, max_r))
            to_del = random.choice(idxs, ((c2.shape[0] - max_r),), replace=False)
            c2 = np.delete(c2, to_del, axis=0)
        children.append(c1)
        children.append(c2)
    #children.append(np.concatenate((p1_list_front, p2_list_back), axis = 0))
    #children.append(np.concatenate((p2_list_front, p1_list_back), axis = 0))
    return children, np.array(retention).astype(np.int32), d_distro


def flatten_xover(parents, max_r, p_xov, bank_length):  # 1 point flattened crossover
    children = []
    retention = []
    d_distro = np.zeros(max_r)
    for i in range(0, len(parents), 2):
        p1 = parents[i].copy()
        p2 = parents[i + 1].copy()
        if random.random() > p_xov:
            children.append(p1)
            children.append(p2)
            continue
        inst_length = p1.shape[1]
        p1 = p1.flatten()
        p2 = p2.flatten()
        retention.append(i)
        inst_counts = [len(p1), len(p2)]
        try:
            if p1.shape[0] > 2 and p2.shape[0] > 2:
                s = [random.randint(1, p1.shape[0] - 1), random.randint(1, p2.shape[0] - 1)]
            elif p1.shape[0] == 2:
                s = [1, random.randint(1, p2.shape[0] - 1)]
            elif p2.shape[0] == 2:
                s = [random.randint(1, p1.shape[0] - 1), 1]
            else:
                s = [0, 0]

        except:
            s = [0, 0]
            print(p1, p2)

        p1_list_front = p1[:s[0]].copy()
        p1_list_back = p1[s[0]:].copy()

        p2_list_front = p2[:s[1]].copy()
        p2_list_back = p2[s[1]:].copy()

        c1 = np.concatenate((p1_list_front, p2_list_back))
        c2 = np.concatenate((p2_list_front, p1_list_back))

        if c1.shape[0] > max_r * inst_length:  #keep to maximimum size size!
            idxs = np.array(range(0, max_r * inst_length))
            to_del = random.choice(idxs, ((c1.shape[0] - max_r * inst_length),), replace=False)
            c1 = np.delete(c1, to_del)
        elif c1.shape[0] % inst_length != 0:  #have to keep divisible by instruction length
            idxs = np.array(range(0, c1.shape[0]))
            to_del = random.choice(idxs, ((c1.shape[0] % inst_length),), replace=False)
            c1 = np.delete(c1, to_del)
        elif c1.shape[0] < inst_length:
            try:
                c1 = np.append(c1, random.choice(c1, ((inst_length - c1.shape[0]),), replace=True))
            except:
                print('c1 length == 0')
        if c2.shape[0] > max_r * inst_length:
            idxs = np.array(range(0, max_r * inst_length))
            to_del = random.choice(idxs, ((c2.shape[0] - max_r * inst_length),), replace=False)
            c2 = np.delete(c2, to_del)
        elif c2.shape[0] < inst_length:
            try:
                c2 = np.append(c2, random.choice(c2, ((inst_length - c2.shape[0]),), replace=True))
            except:
                print('c2 length == 0')
        elif c2.shape[0] % inst_length != 0:  #have to keep divisible by instruction length
            idxs = np.array(range(0, c2.shape[0]))
            to_del = random.choice(idxs, ((c2.shape[0] % inst_length),), replace=False)
            c2 = np.delete(c2, to_del)

        c1 = c1.reshape((-1, inst_length))
        for operation in c1:
            if operation[1] > bank_length - 1:
                operation[1] = random.randint(0, 3)
        c2 = c2.reshape((-1, inst_length))
        for operation in c2:
            if operation[1] > bank_length - 1:
                operation[1] = random.randint(0, 3)

        children.append(c1)
        children.append(c2)
    #children.append(np.concatenate((p1_list_front, p2_list_back), axis = 0))
    #children.append(np.concatenate((p2_list_front, p1_list_back), axis = 0))
    return children, np.array(retention).astype(np.int32)


def twopoint_xover(parents, max_r, p_xov):  # 2 point crossover
    children = []
    retention = []
    d_distro = np.zeros((len(parents), max_r))
    for i in range(0, len(parents), 2):
        if random.random() < p_xov:
            c1 = parents[i].copy()
            c2 = parents[i + 1].copy()
        else:
            retention.append(i)
            p1 = parents[i].copy()
            p2 = parents[i + 1].copy()
            cp1 = np.sort(np.random.randint(1, p1.shape[0] - 1, 2)) if p1.shape[0] > 2 else np.array([1, 1])
            cp2 = np.sort(np.random.randint(1, p2.shape[0] - 1, 2)) if p2.shape[0] > 2 else np.array([1, 1])

            d_distro[i, cp1] += 1
            d_distro[i, cp2] += 1
            d_distro[i+1, cp1] += 1
            d_distro[i+1, cp2] += 1
            p1_list_front = p1[:cp1[0]]
            p1_list_mid = p1[cp1[0]:cp1[1]]
            p1_list_end = p1[cp1[1]:]

            p2_list_front = p2[:cp2[0]]
            p2_list_mid = p2[cp2[0]:cp2[1]]
            p2_list_end = p2[cp2[1]:]

            c1 = np.concatenate((p1_list_front, p2_list_mid, p1_list_end), axis=0)
            c2 = np.concatenate((p2_list_front, p1_list_mid, p2_list_end), axis=0)
            if c1.shape[0] > max_r:  #keep to maximimum rule size!
                idxs = np.array(range(0, max_r))
                to_del = random.choice(idxs, ((c1.shape[0] - max_r),), replace=False)
                c1 = np.delete(c1, to_del, axis=0)
            if c2.shape[0] > max_r:
                idxs = np.array(range(0, max_r))
                to_del = random.choice(idxs, ((c2.shape[0] - max_r),), replace=False)
                c2 = np.delete(c2, to_del, axis=0)
        children.append(c1)
        children.append(c2)
    return children, np.array(retention).astype(np.int32), d_distro


def xover(parents, max_r, p_xov=0.5, mode='Uniform', bank_length=4, fixed_length=False):
    if mode == 'Uniform':
        return uniform_xover(parents, max_r, p_xov)
    elif mode == 'OnePoint':
        return onepoint_xover(parents, max_r, p_xov, fixed_length)
    elif mode == 'TwoPoint':
        return twopoint_xover(parents, max_r, p_xov)
    elif mode == 'Flatten':
        return flatten_xover(parents, max_r, p_xov, bank_length)
    else:
        print('Crossover Mode not Recognized, defaulting to Uniform')
        return uniform_xover(parents, max_r, p_xov)
