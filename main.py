import numpy as np
import random
import matplotlib.pyplot as plt
import math
import copy

class Map:
    def __init__(self, filename, prob_decrease):
        fin = open('input', 'r')
        self.B = 100000
        self.C = 5
        self.I = 1
        self.n_rows = 150
        self.n_cols = 570
        self.col_data = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 480, 510, 540]
        idx = 0
        self.ship = np.zeros([self.n_cols, self.n_rows, 2])
        self.sensors = []
        self.prob_decr = prob_decrease

        for i, line in enumerate(fin):
            if idx >= 2:
                for j, c in enumerate(line):
                    if c == '#':
                        self.ship[j, i - 2, 0] = -1
            idx += 1
        fin.close()

    def score(self):
        P = self.compute_sensor_coverage()
        I_tot = self.compute_sensor_cost()
        tot_cost = len(self.sensors) * self.C + I_tot
        return tot_cost, 100 * P + self.B - tot_cost

    def compute_sensor_coverage(self):
        for sx, sy in self.sensors:
            no_coverage = []
            if self.ship[sx - 1, sy - 1, 0] == -1:
                no_coverage.append([sx - 1, sy - 1])
                no_coverage.append([sx - 1, sy - 2])
                no_coverage.append([sx - 2, sy - 2])
                no_coverage.append([sx - 2, sy - 1])
            if self.ship[sx - 1, sy, 0] == -1:
                no_coverage.append([sx - 1, sy])
                no_coverage.append([sx - 2, sy - 1])
                no_coverage.append([sx - 2, sy + 1])
                no_coverage.append([sx - 2, sy])
            if self.ship[sx - 1, sy + 1, 0] == -1:
                no_coverage.append([sx - 1, sy + 1])
                no_coverage.append([sx - 2, sy + 1])
                no_coverage.append([sx - 2, sy + 2])
                no_coverage.append([sx - 1, sy + 2])
            if self.ship[sx, sy - 1, 0] == -1:
                no_coverage.append([sx - 1, sy - 2])
                no_coverage.append([sx, sy - 1])
                no_coverage.append([sx, sy - 2])
                no_coverage.append([sx + 1, sy - 2])
            if self.ship[sx, sy + 1, 0] == -1:
                no_coverage.append([sx - 1, sy + 2])
                no_coverage.append([sx, sy + 1])
                no_coverage.append([sx, sy + 2])
                no_coverage.append([sx + 1, sy + 2])
            if self.ship[sx + 1, sy - 1, 0] == -1:
                no_coverage.append([sx + 1, sy - 1])
                no_coverage.append([sx + 1, sy - 2])
                no_coverage.append([sx + 2, sy - 1])
                no_coverage.append([sx + 2, sy - 2])
            if self.ship[sx + 1, sy, 0] == -1:
                no_coverage.append([sx + 1, sy])
                no_coverage.append([sx + 2, sy - 1])
                no_coverage.append([sx + 2, sy + 1])
                no_coverage.append([sx + 2, sy])
            if self.ship[sx + 1, sy + 1, 0] == -1:
                no_coverage.append([sx + 1, sy + 1])
                no_coverage.append([sx + 2, sy + 1])
                no_coverage.append([sx + 2, sy + 2])
                no_coverage.append([sx + 1, sy + 2])

            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if not (sx + dx < 0 or sx + dx >= self.n_cols or sy + dy < 0 or sy + dy >= self.n_rows):
                        if not self.ship[sx + dx, sy + dy, 0] == -1:
                            if [sx + dx, sy + dy] not in no_coverage:
                                self.ship[sx + dx, sy + dy, 0] += 1
        p = 0
        for row in self.ship:
            for c in row:
                if c[0] >= 3:
                    p += 1
        return p

    def compute_sensor_cost(self):
        I_tot = 0
        for sy, _ in self.sensors:

            if sy > 540:
                I_tot += self.I * abs(sy - 540)
            else:
                for i, col in enumerate(self.col_data):
                    if sy <= col:
                        if i == 0:
                            col_before = -100000
                        else:
                            col_before = self.col_data[i - 1]
                        #print(f"sy {sy}")
                        #print(col)
                        #print(col_before)
                        if abs(sy - col) < abs(sy - col_before):
                            I_tot += self.I * abs(sy - col)
                            break
                        else:
                            I_tot += self.I * abs(sy - col_before)
                            break

        return I_tot

    def add_sensors(self, prob):
        orig_prob = prob
        prob_decr2 = 1 #math.pow(self.prob_decr, 1/10)
        for sy in range(0, self.n_rows):
            for sx in range(self.n_cols):
                if not self.ship[sx, sy, 0] == -1:
                    if self.ship[sx - 1, sy - 1, 0] == -1:
                        prob = prob*self.prob_decr
                    if self.ship[sx - 1, sy, 0] == -1:
                        prob = prob*self.prob_decr
                    if self.ship[sx - 1, sy + 1, 0] == -1:
                        prob = prob*self.prob_decr
                    if self.ship[sx, sy - 1, 0] == -1:
                        prob = prob*self.prob_decr
                    if self.ship[sx, sy + 1, 0] == -1:
                        prob = prob*self.prob_decr
                    if self.ship[sx + 1, sy - 1, 0] == -1:
                        prob = prob*self.prob_decr
                    if self.ship[sx + 1, sy, 0] == -1:
                        prob = prob*self.prob_decr
                    if self.ship[sx + 1, sy + 1, 0] == -1:
                        prob = prob*self.prob_decr

                    if self.ship[sx - 2, sy - 2, 0] == -1:
                        prob = prob*prob_decr2
                    if self.ship[sx - 2, sy - 1, 0] == -1:
                        prob = prob*prob_decr2
                    if self.ship[sx - 2, sy, 0] == -1:
                        prob = prob*prob_decr2
                    if self.ship[sx - 2, sy + 1, 0] == -1:
                        prob = prob*prob_decr2
                    if self.ship[sx - 2, sy + 2, 0] == -1:
                        prob = prob*prob_decr2
                    if self.ship[sx - 1, sy + 2, 0] == -1:
                        prob = prob*prob_decr2
                    if self.ship[sx, sy + 2, 0] == -1:
                        prob = prob*prob_decr2
                    if self.ship[sx + 1, sy + 2, 0] == -1:
                        prob = prob*prob_decr2
                    if self.ship[sx + 2, sy + 2, 0] == -1:
                        prob = prob*prob_decr2
                    if self.ship[sx + 2, sy + 1, 0] == -1:
                        prob = prob*prob_decr2
                    if self.ship[sx + 2, sy, 0] == -1:
                        prob = prob*prob_decr2
                    if self.ship[sx + 2, sy - 1, 0] == -1:
                        prob = prob*prob_decr2
                    if self.ship[sx + 2, sy - 2, 0] == -1:
                        prob = prob*prob_decr2
                    if self.ship[sx + 1, sy - 2, 0] == -1:
                        prob = prob*prob_decr2
                    if self.ship[sx, sy - 2, 0] == -1:
                        prob = prob*prob_decr2
                    if self.ship[sx - 1, sy - 2, 0] == -1:
                        prob = prob*prob_decr2

                    # if self.ship[sx - 1, sy - 1, 1] == 1:
                    #     prob = prob / prob_decr2
                    # if self.ship[sx, sy - 1, 1] == 1:
                    #     prob = prob / prob_decr2
                    # if self.ship[sx - 1, sy, 1] == 1:
                    #     prob = prob / prob_decr2
                    # if self.ship[sx + 1, sy, 1] == 1:
                    #     prob = prob / prob_decr2
                    # if self.ship[sx, sy + 1, 1] == 1:
                    #     prob = prob / prob_decr2
                    # if self.ship[sx + 1, sy + 1, 1] == 1:
                    #     prob = prob / prob_decr2
                    # if self.ship[sx + 1, sy - 1, 1] == 1:
                    #     prob = prob / prob_decr2
                    # if self.ship[sx - 1, sy + 1, 1] == 1:
                    #     prob = prob / prob_decr2

                    n = random.random()
                    if n < prob:
                        self.sensors.append([sx, sy])
                        self.ship[sx, sy, 1] = 1
                    prob = orig_prob

    def clear(self):
        self.sensors = []

        for i, row in enumerate(self.ship):
            for j, c in enumerate(row):
                if not c == -1:
                    self.ship[i, j, 1] == 0
                    self.ship[i, j, 2] == 0


if __name__ == '__main__':

    sensors_best = []
    prob = 0.4
    prob_decrease = 0.5285714285714286
    score_best = 0

    map_best = Map('input', prob_decrease)
    a = 3
    b = 4
    scores_all = np.zeros([a, b])

    for x, prob in enumerate(np.linspace(0.65, 0.75, a)):
        for y, prob_decrease in enumerate(np.linspace(0.15, 0.17, b)):
            map = Map('input', prob_decrease)
            map.add_sensors(prob)
            tot, score = map.score()
            if tot < 100000:
                scores_all[x, y] = score
            else:
                scores_all[x, y] = 0
            print(f"{prob},{prob_decrease} -> {tot},{score}")
            if tot < 100000 and score > score_best:
                sensors_best = map.sensors
                score_best = score
                map_best = map

    print(f"best score ", score_best)

    fout = open('montagna.txt', 'w')
    for sx, sy in sensors_best:
        fout.write(str(sy) + ' ' + str(sx) + '\n')
    fout.close()

    plt.matshow(map_best.ship[:, :, 1])
    plt.matshow(scores_all)
    plt.show()
