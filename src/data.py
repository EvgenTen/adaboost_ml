import numpy as np
import random



class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_y(self):
        return float(self.y)

    def get_x(self):
        return float(self.x)

    def get_point(self):
        return [self.get_x(), self.get_y()]


class Line:

    def __init__(self, p1, p2):
        if p1.get_x() != p2.get_x():
            c = (p2.get_y() - p1.get_y()) / (p2.get_x() - p1.get_x())
            d = p1.get_y() - (c * p1.get_x())
            self.line = RegularLine(c, d)

        else:
            self.line = Special(p1.get_x())


class RegularLine:

    def __init__(self, c, d):
        self.c = c
        self.d = d

    def get_d(self):
        return self.d

    def get_c(self):
        return self.c

    def get_line(self):
        return [self.get_c(), self.get_d()]

    def get_communic(self, point):
        loc = point.get_x() * self.get_c() + self.get_d() - point.get_y()
        if loc > 0:
            return 1
        else:
            return -1


class Special:

    def __init__(self, x):
        self.x = x

    def get_x(self):
        return self.x

    def get_line(self):
        return [self.get_x()]

    def full_str(self):
        return 'x = {0}'.format(self.get_x())

    def get_communic(self, point):
        if self.get_x() > point.get_x():
            return 1
        else:
            return -1


class Data:


    def __init__(self, file_name, separator=None):
        self.lines = []
        self.rules = []
        with open('{0}'.format(file_name), 'r') as f:
            lines = f.readlines()
            for i in range(0, np.size(lines)):
                data = lines[i].split(separator)
                self.pack_line(data)
            self.mix_train()
        for i in range(0, np.size(self.lines)):
            for j in range(i + 1, np.size(self.lines)):
                p1 = self.lines[i].fun.get_point()
                p2 = self.lines[j].fun.get_point()
                self.rules.append(Regular(self.lines, Line(p1, p2)))



    def mix_train(self):
        random.shuffle(self.lines)
        center = int(np.size(self.lines) / 2)
        self.test_data = self.lines[:center]
        self.train_data = self.lines[center:]


class DataLine:

    def __init__(self, fun, lab):
        self.lab = Label(lab)
        self.fun = Fun(fun)



class Fun:

    def __init__(self, fun):
        self.fun = Point(fun[0], fun[1])

    def get_point(self):
        return self.fun


class Label:

    def __init__(self, lab):
        self.lab = lab

    def get_lab(self):
        if self.lab == '1':
            return 1
        elif self.lab == '2':
            return -1
        if self.lab == 'Iris-versicolor':
            return -1
        elif self.lab == 'Iris-virginica':
            return 1


class Regular:

    def __init__(self, d_lines, line):
        self.line = line
        self.proph = self.set_prophesy(d_lines)


    def prophesy(self, point):
        if self.line.line.get_communic(point) < 0:
            return -self.proph
        else:
            return  self.proph

    def set_prophesy(self, d_lines):
        sum = 0
        for i in d_lines:
            sum += self.line.line.get_communic(i.fun.get_point()) * i.lab.get_lab()
        if sum < 0:
            return -1

        else:
            return 1

    def full_str(self):
        return '{0}'.format(self.line.line.full_str())


class Temperature(Data):


    def __init__(self):
        super().__init__('HC_Body_Temperature.txt') 
        

    def pack_line(self, data):
        self.lines.append(DataLine([data[2], data[0]], data[1]))


class Iris(Data):


    def __init__(self):
        super().__init__('iris.data', ',') 


    def pack_line(self, data):
        self.lines.append(DataLine([data[2], data[1]], data[4].rstrip()))