"""
spark-submit --conf spark.driver.maxResultSize=1g --conf spark.yarn.executor.memoryOverhead=800 --num-executors 6 --executor-cores 1 --driver-memory 500m  --executor-memory 500m spark_sota.py 
"""
import treelib 
import numpy as np
import random
import operator
import copy


def show(tree):
	print tree
	for node in tree.all_nodes():
		print "tag: %s, id: %d, parent_id: %d, is_leaf: %r, data: %s" % (node.tag, node.identifier, node.bpointer if node.bpointer is not None else -1, node.is_leaf(), str(node.data))
	print "\n"

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
	if sister and sister.is_leaf():
		parent.data += pcell * (elem - parent.data)
		sister.data += scell * (elem - sister.data)

def iterateOnData(data, tree, wcell, pcell, scell):
	for elem in data:
		winner = getWinner(elem, tree)
		updateWinner(elem, winner, tree, wcell, pcell, scell)


class SOTA:
	def __init__(self, wcell=0.01, pcell=0.005, scell=0.001, min_improvement_percent=0.001):
		self.wcell = wcell
		self.pcell = pcell
		self.scell = scell
		self.min_improvement_percent = min_improvement_percent
		self.tree = treelib.Tree()
		self.total_trees = 0

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

	def train(self, data, num_iter=5, partitions=12):
		from pyspark import SparkContext
		sc = SparkContext()
		dataRDD = sc.parallelize(data).cache()

		for i in range(num_iter):
			print "iter: %d" % i
			# show(self.tree)
			randomizedRDD = dataRDD.repartition(partitions)
			iterateOnData(randomizedRDD.take(500), self.tree, self.wcell, self.pcell, self.scell) # this helps stablize the new identical children
			treeBC = sc.broadcast(self.tree)

			def train_partition(partition_data):
				localTree = treeBC.value
				iterateOnData(partition_data, localTree, self.wcell, self.pcell, self.scell)
				return [localTree]
			resultTreeRDD = randomizedRDD.mapPartitions(train_partition)

			def add_trees(treeA, treeB):
				treeSum = copy.deepcopy(treeA)
				for leaf in treeA.all_nodes():
					identifier = leaf.identifier
					treeSum.get_node(identifier).data = treeA.get_node(identifier).data + treeB.get_node(identifier).data
				return treeSum
			sumTree = resultTreeRDD.reduce(add_trees)

			def average_tree(tree, count):
				newTree = copy.deepcopy(tree)
				for leaf in newTree.all_nodes():
					leaf.data = leaf.data / float(count)
				return newTree
			newTree = average_tree(sumTree, partitions)
			self.tree = newTree
			# show(self.tree)
			h = self.getHeterogeneity(randomizedRDD.take(1000))
			cellToDivide = self.tree.get_node(h[:,0][np.argmax(h[:,2])])
			if i != num_iter - 1:
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

sota.train(rgb, 4, partitions=12)
show(sota.tree)
