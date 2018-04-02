from itertools import combinations_with_replacement, permutations
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pprint import pprint

class World(object):
	def __init__(self, dimensions, particles, seed='zeros'):
		self.dimensions = dimensions # (x, y, z)
		self.locations = np.prod(self.dimensions) # amount of locations in the world
		self.particles = particles # amount of types of partile in the world
		if seed == 'zeros':
			self.grid = np.zeros(self.dimensions)
		elif seed == 'rand':
			self.grid = np.random.randint(particles, size=self.dimensions)
		else:
			os.exit('invalid seed argument')
		self.grid = np.zeros(self.dimensions)
		self.mapping = {}
		[[self.mapping.update({j: randint(0, particles-1)}) for j in permutations(i)] for i in combinations_with_replacement(range(particles), 2*len(dimensions))] # [(state of adjacent nodes, new state of central square)]

	def next(self, *kwargs): # calculate the next state of the world and return it
		next_grid = np.copy(self.grid)
		pos = tuple([0 for i in range(len(self.dimensions))])
		for i in range(self.locations):
			next_grid[pos] = self.mapping[self.neighbors(pos)]
			pos = self.next_pos(pos)
		self.grid = next_grid
		return next_grid

	def animate(self, *args):
		return plt.imshow(self.next()/(self.particles-1))
	
	def next_pos(self, pos): # advance to the next position in the world
		pos = list(pos)
		for i, j in enumerate(pos):
			if j == self.dimensions[i]-1:
				pos[i] = 0
			else:
				pos[i] += 1
				break
		return tuple(pos)

	def neighbors(self, pos):
		n = []
		for i in range(len(pos)):
			_pos = list(pos)
			
			_pos[i] = pos[i] + 1
			if _pos[i] < self.dimensions[i]:
				n.append(self.grid[tuple(_pos)])
			else:
				n.append(0)
			
			_pos[i] = pos[i] - 1
			if _pos[i] >= 0:
				n.append(self.grid[tuple(_pos)])
			else:
				n.append(0)
		return tuple(n)

particles = 3
world = World(dimensions=(150, 150), particles=particles, seed='zeros')

fig, ax = plt.subplots()

pprint(world.mapping) # the rules of the world

ani = animation.FuncAnimation(fig, func=world.animate, interval=500)

plt.show()