import numpy as np
import matplotlib.pyplot as plt
#import torch

rng = np.random.default_rng()

### Tools :
class circle():
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
    
    def intersectPoints(self, other):
        # Parameters :
        # _ other : the other circle to intersect

        # Returns :
        # _ points : the intersection points (0, 1 or 2)

        x1, y1, x2, y2 = self.center[0], self.center[1], other.center[0], other.center[1]
        r1, r2 = self.radius, other.radius
        
        # if x1 = x2 and y1 = y2, there is no intersection
        if(np.linalg.norm(self.center - other.center) <= np.linalg.norm(self.center)*10**(-6)):
            return []
        
        # if y1 = y2
        if(np.abs(y1 - y2) < 10**(-6)*np.abs(y1)):
            xprime = (x1 + x2)/2 + (r1**2 - r2**2)/(2*(x2 - x1))
            y1prime, y2prime = y1 - np.sqrt(r1**2 - (xprime - x1)**2), y1 + np.sqrt(r1**2 - (xprime - x1)**2)
            return [[xprime, y1prime], [xprime, y2prime]]
        
        A = np.linalg.norm(self.center - other.center)**2
        Bint = (x2**2 - x1**2 + r1**2 - r2**2 + (y2 - y1)**2)
        B = -Bint*(x2-x1) - 2*x1*(y2-y1)**2
        C = (y2 - y1)**2*(x1**2 - r1**2) + (Bint/2)**2

        # solve A x^2 + B x + C = 0
        discrim = B**2 - 4*A*C
        #print(A, " , ", B, " , ", C, " , ", discrim)
        if(discrim > 10**(-5)):
            x1prime, x2prime = (-B-np.sqrt(discrim))/(2*A), (-B+np.sqrt(discrim))/(2*A)
            y1prime, y2prime = y1 + (Bint - 2*x1prime*(x2 - x1))/(2*(y2 - y1)), y1 + (Bint - 2*x2prime*(x2 - x1))/(2*(y2 - y1))
            return [[x1prime, y1prime], [x2prime, y2prime]]
        elif(discrim > -10**(-5)):
            xprime = -B/(2*A)
            yprime = y1 + (Bint - 2*xprime*(x2 - x1))/(2*(y2 - y1))
            return [[xprime, yprime]]
        else:
            return []
        
    def isIn(self, point):
        # Parameters :
        # _ point : point to check

        # Returns True if the point is in the disk, False otherwise
        return np.linalg.norm(point - self.center) <= self.radius
        
    def plot(self, colour='blue'):
        x = np.linspace(self.center[0] - self.radius + 0.0001*self.radius, self.center[0] + self.radius - 0.0001*self.radius, 1000)
        plt.plot(x, self.center[1] + np.sqrt(self.radius**2 - (x - self.center[0])**2), color=colour)
        plt.plot(x, self.center[1] - np.sqrt(self.radius**2 - (x - self.center[0])**2), color=colour)
        

class sillon():
    def __init__(self, architecture, speedlaw = (775/9, 25/9), mindurationbwPR = 50, maxdurationbwPR = 200, cross_deviation = 40, nominal_deviation = 10):
        self.branches = []
        self.durations = []
        self.speedmax = 0.0
        
        mean, std = speedlaw[0], speedlaw[1]
        N = len(architecture)

        self.angles = np.zeros(N) # only to ease the creating and modifying process

        # Generation of a random realistic sillon respecting the architecture
        for k in range(N):
            branch = architecture[k]
            n = branch[0]
            links = [branch[i] for i in range(1, n+1)]
            numberOfPoints = branch[n+1]
            durations = rng.integers(mindurationbwPR, maxdurationbwPR, size=numberOfPoints-1)
            speeds = np.random.normal(mean, std, numberOfPoints-1)
            linkspeeds = np.random.normal(mean, std, n)
            angles = (2.0*np.random.random(numberOfPoints-1)-1)*nominal_deviation*np.pi/180

            initialAngle = 0.0
            if(n >= 1):
                initialAngle = np.mean([self.angles[links[i]] for i in range(len(links))])
            initialAngle += (2.0*np.random.random()-1)*cross_deviation*np.pi/180
            
            initialPoint = np.zeros(2)
            if(n >= 1):
                initialPoint = np.mean([self.branches[links[i]][1][-1] for i in range(len(links))], axis=0)
                initialPoint = initialPoint + mean*(maxdurationbwPR + mindurationbwPR)/2.0*np.array([np.cos(initialAngle), np.sin(initialAngle)])
            
            linkdurations = [np.linalg.norm(self.branches[links[i]][1][-1] - initialPoint)/linkspeeds[i] for i in range(n)]

            points = initialPoint[None, :] + np.zeros((numberOfPoints, 2))
            angles[0] += initialAngle
            for i in range(numberOfPoints-1):
                if(i < numberOfPoints-2):
                    angles[i+1] += angles[i]
                points[i+1] = points[i] + np.array([np.cos(angles[i]), np.sin(angles[i])])*speeds[i]*durations[i]

            self.branches.append([links, points])
            self.durations.append([linkdurations, durations])
            self.angles[k] = angles[numberOfPoints-2]

    def show(self):
        N = len(self.branches)
        minP, maxP = np.zeros(2), np.zeros(2)
        for k in range(N):
            links, points = self.branches[k][0], self.branches[k][1]
            x, y = [points[0][0]], [points[0][1]]
            minP[0], minP[1], maxP[0], maxP[1] = min(minP[0], x[0]), min(minP[1], y[0]), max(maxP[0], x[0]), max(maxP[1], y[0]) 
            for link in links:
                xlink, ylink = self.branches[link][1][-1]
                plt.plot([xlink, x[0]], [ylink, y[0]], color='blue')
                minP[0], minP[1], maxP[0], maxP[1] = min(minP[0], xlink), min(minP[1], ylink), max(maxP[0], xlink), max(maxP[1], ylink)
            
            for i in range(1, len(points)):
                x.append(points[i][0])
                y.append(points[i][1])
                minP[0], minP[1], maxP[0], maxP[1] = min(minP[0], x[i]), min(minP[1], y[i]), max(maxP[0], x[i]), max(maxP[1], y[i])
            
            plt.plot(x, y, color='blue', linestyle=':')
            plt.scatter(x, y, color='blue')

        plt.xlim(minP[0], maxP[0])
        plt.ylim(minP[1], maxP[1])
        plt.show()

    def removePoint(self, numBranch, indPoint):
        links, points = self.branches[numBranch][0], self.branches[numBranch][1]
        if(len(points) > indPoint + 1):
            self.branches[numBranch][1][indPoint] = self.branches[numBranch][1][indPoint + 1]
        else:
            self.branches[numBranch][1][indPoint] = self.branches[numBranch][1][indPoint - 1]

    def neighbourhood(self, numBranch, indPoint):
        # Parameters :
        # _ numBranch : number fo the branch
        # _ indPoint : point index

        # Returns :
        # _ amontPoints : points before point
        # _ amontDurations : travelling time from point to points in amontPoints
        # _ avalPoints : points after point
        # _ avalDurations : travelling time from point to points in avalPoints

        links, points = self.branches[numBranch][0], self.branches[numBranch][1]
        linkdurations, durations = self.durations[numBranch][0], self.durations[numBranch][1]
        n = len(points)
        if(indPoint == 0):
            amontPoints, amontDurations = [self.branches[links[i]][1][-1] for i in range(len(links))], linkdurations
            avalPoints, avalDurations = [points[1]], [durations[1]]
        elif(indPoint == n-1):
            amontPoints, amontDurations = [points[n-2]], [durations[n-2]]
            avalPoints, avalDurations = [], []
            for k in range(numBranch+1, len(self.branches)):
                examlinks, exampoints = self.branches[k][0], self.branches[k][1]
                
                for j in range(len(examlinks)):
                    if(examlinks[j] == numBranch):
                        avalPoints.append(exampoints[0])
                        avalDurations.append(self.durations[k][0][j])
                        break
                
        else:
            amontPoints, amontDurations = [points[indPoint-1]], [durations[indPoint-1]]
            avalPoints, avalDurations = [points[indPoint+1]], [durations[indPoint+1]]

        return amontPoints, amontDurations, avalPoints, avalDurations
    
    def maxSpeed(self, ignoredPoints=[]):
        speedmax = 0.0
        for k in range(len(self.branches)):
            links, points = self.branches[k][0], self.branches[k][1]
            linkdurations, durations = self.durations[k][0], self.durations[k][1]
            
            for j in range(len(links)):
                if((([k,0] in ignoredPoints) == False) and (([links[j], len(self.branches[links[j]][1])-1] in ignoredPoints) == False)):
                    speedmax = max(speedmax, np.linalg.norm(self.branches[links[j]][1][-1] - points[0])/linkdurations[j])

            for i in range(len(points)-1):
                if((([k,i] in ignoredPoints) == False) and (([k, i+1] in ignoredPoints) == False)):
                    speedmax = max(speedmax, np.linalg.norm(points[i+1] - points[i])/durations[i])
        return speedmax

class Interpolator():
    def __init__(self, method=0, params=[]):
        self.points = np.array([])
        self.durations = np.array([])
        self.params = params
        self.speedmax = 0.0
        self.method = method

    def update(self, sillon, numBranch, indPoint):
        self.speedmax = sillon.maxSpeed(ignoredPoints=[[numBranch,indPoint]])
        amontPoints, amontDurations, avalPoints, avalDurations = sillon.neighbourhood(numBranch, indPoint)
        self.points = np.concatenate((amontPoints, avalPoints), axis=0)
        self.durations = np.concatenate((amontDurations, avalDurations))

    def __interpolate(self):
        # Returns a bunch of realistic points regarding travelling times and positions

        maxDomains = []
        nDomains, _ = self.points.shape
        for i in range(nDomains):
            dom = circle(self.points[i], self.durations[i]*self.speedmax)
            maxDomains.append(dom)

        realisticPoints = []
        for i in range(nDomains):
            for j in range(i+1, nDomains):
                points = np.array(maxDomains[i].intersectPoints(maxDomains[j]))
                for k in range(len(points)):
                    for l in range(nDomains):
                        if(maxDomains[l].isIn(points[k]) == False and l != i and l != j):
                            break
                        elif(l == nDomains - 1):
                            realisticPoints.append(points[k])
        realisticPoints = np.array(realisticPoints)
        return maxDomains, realisticPoints

    def retrieve(self):
        # Returns :
        # _ Doms : circles limiting acceptable domains
        # _ retrievedPoints : realistic points retrieved

        if(self.method == 0):
            Doms, retrievedPoints = self.__interpolate()
            while(retrievedPoints.size == 0):
                self.speedmax *= 1.0 + self.params[0]
                Doms, retrievedPoints = self.__interpolate()

            return Doms, retrievedPoints
        elif(self.method == 1):
            dist = self.params[1]+1
            Doms, retrievedPoints = [], np.zeros(0)
            while(self.params[1] < dist):
                Doms, retrievedPoints = self.__interpolate()
                N = retrievedPoints.size
                while(N == 0):
                    self.speedmax *= 1.0 + self.params[0]
                    Doms, retrievedPoints = self.__interpolate()
                    N = retrievedPoints.size
                
                retrievedPointsRed = retrievedPoints - np.mean(retrievedPoints, axis=0)[None, :]
                dist = np.max(np.linalg.norm(retrievedPointsRed, axis=1))
                self.speedmax *= 1.0 - self.params[0]
            
            self.speedmax /= 1.0 - self.params[0]

            return Doms, retrievedPoints
        
    def plot(self):
        Doms, retrievedPoints = self.retrieve()
        plt.plot(retrievedPoints[:, 0], retrievedPoints[:, 1], color='red')
        for i in range(len(Doms)):
            Doms[i].plot('black')
        plt.scatter(retrievedPoints[:,0], retrievedPoints[:,1], color='red')
        final_point = np.mean(retrievedPoints, axis=0)
        plt.scatter([final_point[0]], [final_point[1]], color='green')

### Tests

# sillon
def testSillon():
    Stest = sillon([[0, 5], [1, 0, 3], [1, 0, 6], [1, 1, 5], [1, 1, 4]])
    Stest.show()

def testSillon0():
    Stest = sillon([[0, 5], [1, 0, 3], [1, 0, 6], [1, 1, 5], [1, 1, 4]])
    interpol = Interpolator(0, [0.05])
    interpol.update(Stest, 1, 2)
    interpol.plot()
    Stest.show()

def testSillon1():
    Stest = sillon([[0, 5], [1, 0, 3], [1, 0, 6], [1, 1, 5], [1, 1, 4]])
    interpol = Interpolator(1, [0.05, 100.0])
    interpol.update(Stest, 1, 2)
    interpol.plot()
    Stest.show()

def testcircle1():
    c1 = circle(np.array([1.0, 2.0]), 3.0)
    c2 = circle(np.array([0.0, 5.0]), 4.0)
    points = np.array(c1.intersectPoints(c2))
    print(points)
    plt.scatter(points[:,0], points[:,1], color='red')
    c1.plot()
    c2.plot()
    plt.show()

#testcircle1()
testSillon0()