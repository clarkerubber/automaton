"""Create and animate n-dimensional cellular automata with i-states"""
from itertools import combinations_with_replacement, permutations
from random import randint, choice
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

parser.add_argument("--mappingseed", "-m", dest="mappingseed",
                    default='normal', type=str, help="how should the world's rules be initialised")

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
	def __init__(self, dimensions, particles, seed='one', symmetric=True, duration=50, _mapping=None, mapping_seed='normal'):
		self.dimensions = dimensions # (x, y, z)
		self.particles = particles # amount of types of partile in the world
		self.seed = seed
		self.symmetric = symmetric
		self.duration = duration
		self.mapping_seed = mapping_seed
		self._mapping = _mapping
		self.epoch = 0

		self.grid = World.create_grid(self.dimensions, self.particles, self.seed)
		self.init_grid = np.copy(self.grid)

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

		if _mapping is not None:
			self.mapping = _mapping
		else:
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
		grid = self.next()
		if (grid==np.zeros(self.dimensions)).all() or (grid==self.init_grid).all():
			self.reset()
		return plt.imshow(grid/(self.particles-1))

	@staticmethod
	def create_grid(dimensions, particles, seed):
		if seed == 'zeros':
			return np.zeros(dimensions)
		elif seed == 'rand':
			return np.random.randint(particles, size=dimensions)
		elif seed == 'sparse':
			return np.random.choice(np.arange(particles), size=dimensions, p=[0.75] + (particles-1)*[0.25/(particles-1)])
		elif seed == 'one':
			grid = np.zeros(dimensions)
			grid[tuple([int(i/2) for i in dimensions])] = 1
			return grid
		elif seed == 'border' and len(dimensions) == 2:
			grid = np.zeros(dimensions)
			for i in range(dimensions[0]):
				grid[(i, 0)] = 1
				grid[(i, dimensions[1]-1)] = 1
			for i in range(dimensions[1]):
				grid[(0,i)] = 1
				grid[(dimensions[0]-1, i)] = 1
			return grid
		else:
			os.exit('invalid seed argument')

	def create_mapping(self):
		mapping = {}
		if self.symmetric:
			[[mapping.update({i+(j,): self.random_particle()}) for j in range(self.particles)] for i in combinations_with_replacement(range(self.particles), len(self.neighbor_combinations))]
			#pprint(list(mapping.keys()))
			mapping[tuple([0 for i in list(mapping.keys())[0]])] = 0 # ground state (0,) => 0
		else:
			[[mapping.update({j: self.random_particle()}) for j in permutations(i)] for i in combinations_with_replacement(range(self.particles), len(self.neighbor_combinations))] # [(state of adjacent nodes, new state of central square)]
		return mapping

	def random_particle(self):
		return choice(self.particles*2*[0] + list(range(self.particles))) if self.mapping_seed == 'sparse' else randint(0, self.particles-1)
	
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
		self.grid = World.create_grid(
			dimensions=self.dimensions,
			particles=self.particles,
			seed=self.seed)
		self.init_grid = np.copy(self.grid)
		if self._mapping is not None:
			self.mapping = self._mapping
		else:
			self.mapping = self.create_mapping()
		pprint(self.mapping)
		self.epoch = 0

conway = {
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

world = World(
	dimensions=tuple(settings.dimensions),
	particles=settings.particles,
	seed=settings.seed,
	symmetric=(not settings.asymmetric),
	duration=settings.time,
	_mapping=conway if settings.conway else None,
	mapping_seed=settings.mappingseed)

fig, ax = plt.subplots()

pprint(world.mapping) # the rules of the world

ani = animation.FuncAnimation(fig, func=world.animate, interval=50, frames=100, save_count=10000)
plt.show()
#ani.save('animations/animation.gif', writer='imagemagick', fps=4)
print("saving")
print("end")