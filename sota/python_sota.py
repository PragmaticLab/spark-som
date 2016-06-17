import treelib 
import numpy as np
import random
import operator


nodeName = lambda parent, isLeft: "P_%d_%s" % (parent.identifier, "L" if isLeft else "R")

def nodeGetParent(node, tree):
	return tree.get_node(node.bpointer)

def nodeGetChildren(node, tree):
	return [tree.get_node(identifier) for identifier in node.fpointer]

def nodeGetSister(node, tree):
	parent = nodeGetParent(node, tree)
	if not parent:
		return None
	if not parent.fpointer:
		return None
	sisterList = [tree.get_node(identifier) for identifier in parent.fpointer if identifier != node.identifier]
	return sisterList[0] if sisterList else None

def getWinner(elem, tree):
	leaves = tree.leaves()
	distance_map = {}
	for leaf in leaves:
		distance_map[leaf] = np.linalg.norm(elem-leaf.data)
	winner = min(distance_map.iteritems(), key=operator.itemgetter(1))[0]
	return winner

def updateWinner(elem, winner, tree, wcell, pcell, scell):
	parent = nodeGetParent(winner, tree)
	sister = nodeGetSister(winner, tree)
	winner.data += wcell * (elem - winner.data)
	if sister:
		parent.data += pcell * (elem - parent.data)
		sister.data += scell * (elem - sister.data)


class SOTA:
	def __init__(self, wcell=0.01, pcell=0.005, scell=0.001, min_improvement_percent=0.001):
		self.wcell = wcell
		self.pcell = pcell
		self.scell = scell
		self.min_improvement_percent = min_improvement_percent
		self.tree = treelib.Tree()
		self.total_trees = 0

	def show(self, show_architecture=True):
		if show_architecture:
			self.tree.show()
		for node in self.tree.all_nodes():
			print "tag: %s, id: %d, parent_id: %d, is_leaf: %r, data: %s" % (node.tag, node.identifier, node.bpointer if node.bpointer is not None else -1, node.is_leaf(), str(node.data))

	def _getNextTreeId(self):
		num = self.total_trees
		self.total_trees += 1
		return num

	def _nodeReplicate(self, node):
		leftNode = self.tree.create_node(nodeName(node, True), self._getNextTreeId(), parent=node.identifier, data=node.data.copy())
		rightNode = self.tree.create_node(nodeName(node, False), self._getNextTreeId(), parent=node.identifier, data=node.data.copy())

	def getLeafNodes(self):
		return self.tree.leaves()

	def initialize(self, data):
		rootNode = self.tree.create_node("root", self._getNextTreeId(), data=np.mean(data, axis=0))
		self._nodeReplicate(rootNode)

	def train(self, data, sample=10000, num_iter=5):

		for i in range(num_iter):
			sampled_data = data[np.random.choice(data.shape[0], sample)]
			for elem in sampled_data:
				winner = getWinner(elem, self.tree)
				updateWinner(elem, winner, self.tree, self.wcell, self.pcell, self.scell)
			h = self.getHeterogeneity(sampled_data)
			cellToDivide = self.tree.get_node(h[:,0][np.argmax(h[:,2])])
			self._nodeReplicate(cellToDivide)

	def getHeterogeneity(self, data):
		leafNodes = self.getLeafNodes()
		for node in leafNodes:
			node.contents = []
		for elem in data:
			winner = getWinner(elem, self.tree)
			winner.contents += [elem]
		results = []
		for node in leafNodes:
			diffs = [np.linalg.norm(content - node.data) for content in node.contents]
			average = sum(diffs) / len(diffs)
			results += [(node.identifier, len(node.contents), average)]
		results = np.array(results)
		# clear the chart
		for node in leafNodes:
			node.contents = []
		return results


rgb = np.load("../data/generated_rgb.np")
# rgb /= 255.0

sota = SOTA(wcell=0.01, pcell=0.005, scell=0.001)
sota.initialize(rgb)

sota.train(rgb, sample=10000, num_iter=4)
sota.show()

