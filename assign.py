import itertools
import numpy as np
import random
import time

# to build the all the different eligibiity vectors
def xx_builder(i, j):
    # input the name of the boxes
    # output is xx: a list of all possible X's for two boxes
    # x_pp a list containing the probability of each X's in xx
    xx = list(map(list, itertools.product([0, 1], repeat=2)))
    x_pp = []

    for x in xx:
        p_x = 1
        # for ii in range(0,2):
        #     p_x = p_x * x[ii] * p[i-1] + p_x * (1-x[ii])*(1-p[i-1])
        p_x = p_x * x[0] * p_vec[i] + p_x * (1-x[0])*(1-p_vec[i])
        p_x = p_x * x[1] * p_vec[j] + p_x * (1-x[1])*(1-p_vec[j])
        x_pp.append(p_x)

    return xx, x_pp


def two_box_dp(m_i, m_j, x_i, x_j, i, j):
    alpha = .9
    # Name of boxes are 0, 1, 2, ...

    N = np.ones((m_i+1, m_j+1, 2, 2))  # the matrix of DP,
    #  Note: m_i+1 due to zero remaining quotas, last column is for the box that is chosen
    N[0, 0, :, :] = 0  # for 0 remaining quota DP is 0
    #
    for jj in range(1, m_j+1):
        N[0, jj, :, :] = jj/p_vec[j]  # if one of the boxes is full the remaining expected number of balls is j/p_j
    for ii in range(1,m_i+1):
        N[ii, 0, :, :] = ii/p_vec[i]  # if one of the boxes is full the remaining expected number of balls is i/p_i

    xx, x_pp = xx_builder(i, j)

    # the main loop for the DP
    for i_m in range(1,m_i+1):
        for j_m in range(1,m_j+1):
            for i_x in range(0,2):
                for j_x in range(0,2):
                    if i_x + j_x == 0:
                        sum_returns = 1
                        k = 0
                        for x in xx:
                            sum_returns += x_pp[k] * N[i_m, j_m, x[0], x[1]]
                            k += 1
                        N[i_m, j_m, i_x, j_x] = alpha * sum_returns
                    elif i_x == 1 and j_x == 0:
                        sum_returns = 1 # to calculate the recursive equation
                        # xx, x_pp = xx_builder(i, j)
                        k = 0
                        for x in xx:
                            sum_returns += x_pp[k] * N[i_m-1, j_m, x[0], x[1]]
                            k += 1
                        N[i_m, j_m, i_x, j_x] = alpha * sum_returns
                    elif i_x == 0 and j_x == 1:
                        sum_returns = 1
                        # xx, x_pp = xx_builder(i, j)
                        k = 0
                        for x in xx:
                            sum_returns += x_pp[k] * N[i_m, j_m-1, x[0], x[1]]
                            k += 1
                        N[i_m, j_m, i_x, j_x] = alpha * sum_returns
                    elif i_x == 1 and j_x == 1:
                        # xx, x_pp = xx_builder(i, j)
                        sum_returns_putini = 1  # to calculate the value if put in box i
                        k = 0
                        for x in xx:
                            sum_returns_putini += x_pp[k] * N[i_m-1, j_m, x[0], x[1]]
                            k += 1
                        sum_returns_putinj = 1  # to calculate the value if put in box j
                        k = 0
                        for x in xx:
                            sum_returns_putinj += x_pp[k] * N[i_m, j_m-1, x[0], x[1]]
                            k += 1
                        if sum_returns_putini >= sum_returns_putinj:  # take the minimum
                            N[i_m, j_m, i_x, j_x] = alpha * sum_returns_putinj
                        else:
                            N[i_m, j_m, i_x, j_x] = alpha * sum_returns_putini

    winner_box = 0
    if x_i + x_j == 0 or (m_i+m_j == 0):
        winner_box = -1
    elif x_i == 1 and x_j == 0 and (m_i >0):
        winner_box = i
    elif x_i == 0 and x_j == 1 and (m_j >0):
        winner_box = j
    elif x_i == 1 and x_j == 1 and (m_i == 0):
        winner_box = j
    elif x_i == 1 and x_j == 1 and (m_j == 0):
        winner_box = i
    elif x_i == 1 and x_j == 1:
        # xx, x_pp = xx_builder(i, j)
        sum_returns_putini = 1  # to calculate the value if put in box i
        k = 0
        for x in xx:
            sum_returns_putini += x_pp[k] * N[m_i - 1, m_j, x[0], x[1]]
            k += 1
        sum_returns_putinj = 1  # to calculate the value if put in box j
        k = 0
        for x in xx:
            sum_returns_putinj += x_pp[k] * N[m_i, m_j - 1, x[0], x[1]]
            k += 1
            if sum_returns_putini > sum_returns_putinj:  # take the minimum
                winner_box = j
            elif sum_returns_putini < sum_returns_putinj:
                winner_box = i
            else:
                winner_box = random.choice([i, j])

    return winner_box

# print(two_box_dp(4, 4, 1, 1, 0, 1))

max_ite = 20
N_phi_h_tilde = np.zeros(max_ite)  # our approach average number of balls to fill all boxes
random.seed(110)
np.random.seed(110)
t0 = time.time()
for ite in range(max_ite):
    # papers example
    n_box = 9  # number of boxes
    p_vec = [.1, .15, .2, .25, 0.3, .35, .4, .45, .5]  # probability of eligibility for boxes
    m_vec = [5, 8, 10, 12, 15, 18, 20, 23, 26]

    # test example
    # n_box = 3  # number of boxes
    # p_vec = [.1, .15, .2]
    # m_vec = [5, 8, 10]

    # two box
    # n_box = 2  # number of boxes
    # p_vec = [.1, .9]
    # m_vec = [5, 8]

    print("#########ITERATION", ite, ", m_vec", m_vec)
    while sum(m_vec) > 0:
        N_phi_h_tilde[ite] += 1
        X_new = np.maximum(np.sign(p_vec - np.random.rand(n_box)), 0)
        print("X_new", X_new)
        winnings = np.zeros(n_box)
        if sum(X_new) == 1: # just one is eligible
            potential_winner = np.argmax(X_new)
            if m_vec[potential_winner] > 0:
                winner = potential_winner
                m_vec[potential_winner] -= 1
                final_winner = potential_winner
            else:
                winner = -1
                final_winner = -1
        else:
            for first_box in range(n_box):
                if X_new[first_box] == 1:
                    for second_box in range(first_box + 1, n_box):
                        if X_new[second_box] == 1:
                            winner = two_box_dp(m_vec[first_box], m_vec[second_box], X_new[first_box],
                                                X_new[second_box], first_box,
                                                second_box)  # winner box of the two_box game
                            print("winner", winner)
                            if winner == first_box:
                                winnings[first_box] += 1
                            elif winner == second_box:
                                winnings[second_box] += 1
            if sum(winnings) != 0:  # some one won
                final_winner = np.argmax(winnings)  # winner box of the new ball
                m_vec[final_winner] -= 1
            else:
                final_winner = -1  # includes the case of ball ineligible for all the boxes and all the boxes full
        print("winnings", winnings, ", final_winner", final_winner, ", m_vec", m_vec)
    print("ITERATION", ite, ", N", N_phi_h_tilde[ite], "\n\n")
print("Time=", time.time() - t0)
print("Mean N =", np.mean(N_phi_h_tilde), ", Variance of N =", np.var(N_phi_h_tilde)/max_ite)













