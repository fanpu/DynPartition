# tree object from stanfordnlp/treelstm
class Tree:
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.gold_label = None  # node label for SST
        self.output = None  # output node for SST

        # used by Math Functions only
        self.layer = None  # layer of the node in the tree
        self.name = None  # name of the node

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __repr__(self):
        return f"{self.layer} : {self.name}"
