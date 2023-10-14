import math

class FibonacciHeapNode:
    def __init__(self, wrap, key):
        self.wrap = wrap    
        self.key = key
        self.parent = None
        self.child = None 
        self.left = self.right = self
        self.degree = 0 
        self.mark = False 
    
    def suppress_neighbors(self):
        self.left.right = self.right
        self.right.left = self.left 
        self.right = self.left = self
    
    def link_to_the_left(self, node):
        if self.right != self:
            raise Exception("The node has at least one neighbor.")
        
        node.left.right = self 
        self.left = node.left 
        self.right = node
        node.left = self 

    def link_as_child(self, parent):
        if self != parent: 
            self.suppress_neighbors()

            if parent.child is not None:
                child = parent.child
                self.link_to_the_left(child)
            else:
                parent.child = self

            self.parent = parent
            parent.degree += 1
    
    def remove_from_child_list(self, node):
        if node.parent != self:
            return
        
        if self.child == self.child.right:
            self.child = None

        elif self.child == node:
            self.child = node.right
            node.right.self = self

        node.left.right = node.right
        node.right.left = node.left

    def merge_with_child_list(self, node):
        if self.child is None:
            self.child = node

        else:
            node.right = self.child.right
            node.left = self.child
            self.child.right.left = node
            self.child.right = node

class FibonacciHeap:
    def __init__(self):
        self.min_node = None
        self.root_list = None
        self.nb_nodes = 0
    
    def insertion(self, wrap, key):
        new_node = FibonacciHeapNode(wrap, key)
        self.merge_with_root_list(new_node)

        if self.min_node is None or new_node.key < self.min_node.key:
            self.min_node = new_node

        self.nb_nodes += 1
    
    def merge_with_root_list(self, node):
        if self.root_list is None:
            self.root_list = node

        else:
            node.right = self.root_list.right
            node.left = self.root_list
            self.root_list.right.left = node
            self.root_list.right = node

    def update_min_node(self, node):
        if node.key < self.min_node.key:
            self.min_node = node

    def have_wrap(self, wrap):
        if self.min_node is None:
            return False
        
        return dfs_node(self.min_node, wrap) is not None

    def extract_min(self):
        extract_node = self.min_node

        if extract_node is not None:
            if extract_node.child is not None:
                child = extract_node.child

                while True:
                    other_child = child.right
                    self.merge_with_root_list(child)
                    child.parent = None

                    if other_child == extract_node.child:
                        break

                    child = other_child

            self.remove_from_root_list(extract_node)

            if extract_node == extract_node.right:
                self.min_node = self.root_list = None

            else:
                self.min_node = extract_node.right
                self.consolidate()

            self.nb_nodes -= 1
            
            return extract_node.wrap
    
    def remove_from_root_list(self, node):
        if node == self.root_list:
            self.root_list = node.right

        node.left.right = node.right
        node.right.left = node.left

    def _extract_only_node(self):
        extracted_node = self.min_node
        self.min_node = None
        self.nb_nodes = 0  

        return extracted_node.wrap
    
    def consolidate(self):
        MAX_DEGREE = 2 * int(math.log2(self.nb_nodes)) + 1
        root_list = [None] * (MAX_DEGREE + 1)

        node = self.min_node
        while True:
            degree = node.degree

            while root_list[degree] is not None:
                neighbour = root_list[degree]

                if neighbour.key < node.key:
                    neighbour, node = node, neighbour
                
                self.heap_link(node, neighbour)
                root_list[degree] = None
                degree += 1

            root_list[degree] = node

            if node == self.min_node:
                break

            node = node.right

        for node in root_list:
            if node is not None:
                if node.key < self.min_node.key:
                    self.min_node = node

    def decrease_key(self, wrap, new_key):
        node = self._get_node_by_wrap(wrap)

        if new_key >= node.key:
            return 
        
        node.key = new_key
        parent = node.parent
        
        if (parent is not None) and node.key < parent.key:
            self._cut(node, parent)
            self._cascade_cut(parent)

        if node.key < self.min_node.key:
            self.min_node = node

    def _get_node_by_wrap(self, wrap):
        if self.min_node is None:
            return None 
        
        return dfs_node(self.min_node, wrap)
    
    def _cut(self, node, parent):
        parent.remove_from_child_list(node)
        parent.degree -= 1
        self.merge_with_root_list(node)
        node.mark = False 
        node.parent = None  

    def _cascade_cut(self, node):
        parent = node.parent

        if parent is not None:
            if not node.mark :
                node.mark = True

            else:
                self._cut(node, parent)
                self._cascade_cut(parent)

    def heap_link(self, parent, child):
        self.remove_from_root_list(child)
        child.left = child.right = child
        parent.merge_with_child_list(child)
        parent.degree += 1
        child.parent = parent
        child.mark = False
    
def dfs_node(node, wrap):
    stack = [node]
    visited = set()

    while stack:
        current_node = stack.pop()

        if current_node not in visited:
            visited.add(current_node)

            if current_node.wrap == wrap:
                return current_node

            if current_node.child:
                stack.append(current_node.child)

            if current_node.right != current_node:
                stack.append(current_node.right)

    return None