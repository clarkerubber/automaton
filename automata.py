"""Create and animate n-dimensional cellular automata with i-states"""
from itertools import combinations_with_replacement, permutations
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pprint import pprint
import argparse

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("--particles", "-p", dest="particles",
                    default=2, type=int, help="how many states can each grid occupy")

parser.add_argument("--seed", "-s", dest="seed",
                    default='one', type=str, help="how should the world be initialised")

parser.add_argument("--asymmetric", "-a", dest="asymmetric",
                    default=False, const=True, type=bool, nargs='?', help="should the rules be asymmetric")

parser.add_argument("--conway", dest="conway",
                    default=False, const=True, type=bool, nargs='?', help="should the rules be conway's game of life")

parser.add_argument("--time", "-t", dest="time",
					default=50, type=int, help="how many frames should the animation be run for")

parser.add_argument("--dimensions", "-d", dest="dimensions",
                    default=(101,101), nargs='+', type=int, help="size of the world grid")

settings = parser.parse_args()

class World(object):
	def __init__(self, dimensions, particles, seed='one', symmetric=True, duration=50, conway=False):
		self.dimensions = dimensions # (x, y, z)
		self.particles = particles # amount of types of partile in the world
		self.seed = seed
		self.symmetric = symmetric
		self.duration = duration
		self.conway = conway
		self.epoch = 0

		self.grid = World.create_grid(self.dimensions, self.particles, self.seed)

		self.locations = []
		pos = [0 for i in range(len(self.dimensions))]
		for i in range(np.prod(self.dimensions)):
			self.locations.append(pos)
			pos = self.next_pos(pos)

		self.locations = np.array(self.locations)

		self.neighbor_combinations = []
		[[self.neighbor_combinations.append(j) for j in permutations(i)] for i in combinations_with_replacement([-1,0,1], len(self.dimensions))]
		self.neighbor_combinations = np.array(list(filter(lambda a: a != tuple([0 for i in self.dimensions]), list(set(self.neighbor_combinations)))))
		pprint(self.neighbor_combinations)

		self.mapping = self.create_mapping()

	def update_pos(self, pos):
		self.next_grid[pos] = self.mapping[self.neighbors(pos)]
		return 1

	def next(self, *kwargs): # calculate the next state of the world and return it
		next_grid = np.copy(self.grid)
		for pos in self.locations:
			tpos = tuple([int(i) for i in pos])
			next_grid[tpos] = self.mapping[self.neighbors(pos)]
		self.grid = next_grid
		return self.grid

	def animate(self, *args):
		print(self.epoch)
		self.epoch += 1
		if self.epoch >= self.duration:
			self.reset()
		return plt.imshow(self.next()/(self.particles-1))

	@staticmethod
	def create_grid(dimensions, particles, seed):
		if seed == 'zeros':
			return np.zeros(dimensions)
		elif seed == 'rand':
			return np.random.randint(particles, size=dimensions)
		elif seed == 'one':
			grid = np.zeros(dimensions)
			grid[tuple([int(i/2) for i in dimensions])] = 1
			return grid
		else:
			os.exit('invalid seed argument')

	def create_mapping(self):
		mapping = {}
		if self.conway:
			return {
				(0,0,0,0,0,0,0,0,0): 0,
				(0,0,0,0,0,0,0,0,1): 0,
				(0,0,0,0,0,0,0,1,0): 0,
				(0,0,0,0,0,0,0,1,1): 0,
				(0,0,0,0,0,0,1,1,0): 0,
				(0,0,0,0,0,0,1,1,1): 1,
				(0,0,0,0,0,1,1,1,0): 1,
				(0,0,0,0,0,1,1,1,1): 1,
				(0,0,0,0,1,1,1,1,0): 0,
				(0,0,0,0,1,1,1,1,1): 0,
				(0,0,0,1,1,1,1,1,0): 0,
				(0,0,0,1,1,1,1,1,1): 0,
				(0,0,1,1,1,1,1,1,0): 0,
				(0,0,1,1,1,1,1,1,1): 0,
				(0,1,1,1,1,1,1,1,0): 0,
				(0,1,1,1,1,1,1,1,1): 0,
				(1,1,1,1,1,1,1,1,0): 0,
				(1,1,1,1,1,1,1,1,1): 0,
			}
		if self.symmetric:
			[[mapping.update({i+(j,): randint(0, self.particles-1)}) for j in range(self.particles)] for i in combinations_with_replacement(range(self.particles), len(self.neighbor_combinations))]
			#pprint(list(mapping.keys()))
			mapping[tuple([0 for i in list(mapping.keys())[0]])] = 0 # ground state (0,) => 0
		else:
			[[mapping.update({j: randint(0, self.particles-1)}) for j in permutations(i)] for i in combinations_with_replacement(range(self.particles), len(self.neighbor_combinations))] # [(state of adjacent nodes, new state of central square)]
		return mapping
	
	def next_pos(self, pos): # advance to the next position in the world
		pos = list(pos)
		for i, j in enumerate(tuple(list(pos))):
			if int(j) == self.dimensions[i]-1:
				pos[i] = 0
			else:
				pos[i] += 1
				break
		return pos

	def neighbors(self, pos):
		if self.symmetric:
			return tuple(sorted([self.grid[tuple([int(j%self.dimensions[i]) for i, j in enumerate(p)])] for p in self.neighbor_combinations + pos])) + (self.grid[tuple(pos)],)
		else:
			return tuple([self.grid[tuple([int(j%self.dimensions[i]) for i, j in enumerate(p)])] for p in self.neighbor_combinations + pos])

	def reset(self):
		self.grid = World.create_grid(self.dimensions, self.particles, self.seed)
		self.mapping = self.create_mapping()
		pprint(self.mapping)
		self.epoch = 0

world = World(
	dimensions=tuple(settings.dimensions),
	particles=settings.particles,
	seed=settings.seed,
	symmetric=(not settings.asymmetric),
	duration=settings.time,
	conway=settings.conway)

fig, ax = plt.subplots()

pprint(world.mapping) # the rules of the world

ani = animation.FuncAnimation(fig, func=world.animate, interval=50)
plt.show()