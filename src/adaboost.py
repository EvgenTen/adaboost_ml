import numpy as np
import warnings
# Phyton 3.8

class Adaboost:

    def __init__(self, rules, lines):
        self.rules = rules
        self.lines = lines
        self.weights = []
        for i in range(0, np.size(lines)):  # give equal weights to any data point
            self.weights.append(1 / np.size(lines))
        self.r_weigths = np.empty(np.size(rules))

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    def adaBoost_train(self):
        for i in range(8):
            self.r_error = np.empty(np.size(self.rules))
            for r_index, rule in enumerate(self.rules):
                for dl_index, d_line in enumerate(self.lines):

                    if rule.prophesy(d_line.fun.get_point()) != d_line.lab.get_lab():
                     self.r_error[r_index] += float(self.weights[dl_index])
            best_rule = np.argmin(self.r_error)
            error = self.r_error[best_rule]
            if error==0: error=np.full(0.0001)
            if error>0:
             self.r_weigths[best_rule] = 1 / 2 * np.log((1 - error) / error)

            for pw_index in range(np.size(self.weights)):
                self.weights[pw_index] *= np.exp(
                    - self.r_weigths[best_rule] * self.rules[best_rule].prophesy(self.lines[pw_index].fun.get_point()) *
                    self.lines[pw_index].lab.get_lab())

            self.weights = self.weights / np.sum(self.weights)


    
    def get_H(self, inf):
        best_rules = self.r_weigths.argsort()[-8:][::-1]
        sum_arr = []
        g_F = np.empty(np.size(inf))
        for i in range(8):
            for dline_index, d_line in enumerate(inf):
                sum = 0
                for rule in best_rules[:i]:
                    sum += self.rules[rule].prophesy(d_line.fun.get_point()) * self.r_weigths[rule]
                if sum < 0:
                    g_F[dline_index] = -1
                else:
                    g_F[dline_index] = 1
            positive = 0
            for i in range(np.size(g_F)):
                if g_F[i] != inf[i].lab.get_lab():
                    positive += 1
            sum_arr.append(positive)
        return np.array(sum_arr)
        

    
