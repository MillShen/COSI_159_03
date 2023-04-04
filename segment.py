import numpy as np
import math
import matplotlib.pyplot as plt
from skimage import color, io
from collections import defaultdict


# The Centroid class which stores the values of the cluster centroids
class Centroid(object):

    def __init__(self, x, y, l, a, b, image=None):
        self.image = image
        self.x = x
        self.y = y
        self.l = l
        self.a = a
        self.b = b
        # list of pixels assigned to a centroid
        self.pixels = []


    @classmethod
    def autoCentroid(cls, x, y, image):
        return cls(x, y, image[y][x][0], image[y][x][1], image[y][x][2], image)

    def update(self, x, y, l, a, b):
        self.x = x
        self.y = y
        self.l = l
        self.a = a
        self.b = b


# The Processor class which executes the SLIC algorithm
class Process(object):

    def __init__(self, image, segments, m=10, threshold=100):
        self.clusters = []
        self.groups = defaultdict(Centroid)


        self.K = segments
        self.M = m
        self.threshold = threshold
        self.image = image
        self.image_h = image.shape[0]
        self.image_w = image.shape[1]
        self.N = self.image_w * self.image_h
        self.S = math.sqrt(self.N / self.K)  # Quick search showed that math library sqrt is much faster than np

        self.distances = np.full((self.image_h, self.image_w), np.inf) # initializing all distances to infinity

    # Evenly chooses ~K, evenly-spaced pixels as centroids and stores them
    def initialize_centroids(self):
        """Simple list comprehension to evenly space centroids across the image plane without edges"""
        start = int(self.S // 2)
        s = round(self.S)
        self.clusters = [Centroid.autoCentroid(x, y, self.image) for y in range(start, self.image_h, s)
                         for x in range(start, self.image_w, s)]

    def calc_dist(self, cluster, x, y):
        """Util Function to calculate Euclidean distance in 5D as per the paper"""
        l, a, b = self.image[y][x]
        Dl = math.sqrt(math.pow(l - cluster.l, 2) +
                       math.pow(a - cluster.a, 2) +
                       math.pow(b - cluster.b, 2))
        Dx = math.sqrt(math.pow(y - cluster.y, 2) +
                       math.pow(x - cluster.x, 2))
        return Dl + (self.M / self.S)*Dx

    # I choose to forgo the gradient step
    # Instead this dives right into the Assignment for a 2S*2S square of pixels around each centroid
    def group(self):
        """Examines a 2S*2S square around each centroid and calculates distance from each pixel in that region
            to the centroid, then assigns the pixel to the nearest centroid. A bit of a slow, backwards
             process, but it works"""
        s = round(self.S)
        for cluster in self.clusters:
            for y in range(0 if cluster.y - s < 0 else cluster.y - s,
                           self.image_h if cluster.y + s > self.image_h else cluster.y + s):
                for x in range(0 if cluster.x - s < 0 else cluster.x - s,
                               self.image_w if cluster.x + s > self.image_w else cluster.x + s):
                    dist = self.calc_dist(cluster, x, y)
                    if dist < self.distances[y][x]:
                        if (y, x) not in self.groups:
                            self.groups[(y, x)] = cluster
                            cluster.pixels.append((y, x))
                        else:
                            self.groups[(y, x)].pixels.remove((y, x))
                            self.groups[(y, x)] = cluster
                            cluster.pixels.append((y, x))
                        self.distances[y][x] = dist

    def eval_threshold(self, new_centroids, threshold):
        """Util Function to evaluate the level of change in distance by the centroids, stopping once below
        a certain threshold"""
        return True if max([self.calc_dist(c, _c.x, _c.y) for c, _c in zip(self.clusters, new_centroids)]) <= threshold\
            else False

    def move_centroids(self):
        """Moves the centroids to the average of its constituent x and y pixel values,
            then updates accordingly"""
        new_centroids = []
        for c in self.clusters:
            sy = sx = num = 0
            for p in c.pixels:
                sy += p[0]
                sx += p[1]
                num += 1
            _y = int(sy // num)
            _x = int(sx // num)
            new_centroids.append(self.calc_dist(c, _x, _y))
            c.update(_y, _x, self.image[_y][_x][0], self.image[_y][_x][1], self.image[_y][_x][2])
        m = max(new_centroids)
        print(m)
        return True if m > self.threshold else False

    def slic_compute(self):
        """Runs through the grouping and rearranging steps until some threshold value is met
            In this case, because the L*a*b* values are not normalized with each other, the threshold is 100"""
        self.initialize_centroids()
        run = True
        while run:
            self.group()
            run = self.move_centroids()

    def save_borders(self):
        """Saves the image as a collection of superpixels.
            I don't know how to mark boundaries so instead each color block represents a different
            superpixel value and shape"""
        image_arr = np.copy(self.image)
        print([(c.x, c.y) for c in self.clusters if c.x > self.image_w or c.y > self.image_h])
        for c in self.clusters:
            for p in c.pixels:
                image_arr[p[0]][p[1]][0] = c.l
                image_arr[p[0]][p[1]][1] = c.a
                image_arr[p[0]][p[1]][2] = c.b
            image_arr[c.x][c.y][0] = 0
            image_arr[c.x][c.y][1] = 0
            image_arr[c.x][c.y][2] = 0
        io.imsave("Home_SLIC.png", color.lab2rgb(image_arr))
