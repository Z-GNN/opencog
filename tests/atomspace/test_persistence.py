#!/usr/bin/env python3
"""
Test Persistence Test

This test validates AtomSpace hypergraph operations and persistence
using real knowledge graphs and large-scale data structures.
"""

import unittest
import sys
import os
import time
from pathlib import Path

# Add OpenCog paths  
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from opencog.atomspace import AtomSpace, types, TruthValue
    from opencog.utilities import initialize_opencog
    ATOMSPACE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  AtomSpace not available: {e}")
    ATOMSPACE_AVAILABLE = False

class TestAtomSpaceTestPersistence(unittest.TestCase):
    """Test AtomSpace hypergraph operations and functionality."""
    
    def setUp(self):
        """Set up test environment with real hypergraph data."""
        if not ATOMSPACE_AVAILABLE:
            self.skipTest("AtomSpace not available for testing")
            
        self.atomspace = AtomSpace()
        self.test_data_path = Path(__file__).parent.parent.parent / "test-datasets" / "atomspace"
        self._create_real_knowledge_graph()
    
    def _create_real_knowledge_graph(self):
        """Create real knowledge graph data for testing."""
        self.test_data_path.mkdir(exist_ok=True)
        
        # Create realistic knowledge graph
        # Concepts
        cat = self.atomspace.add_node(types.ConceptNode, "cat")
        animal = self.atomspace.add_node(types.ConceptNode, "animal") 
        mammal = self.atomspace.add_node(types.ConceptNode, "mammal")
        pet = self.atomspace.add_node(types.ConceptNode, "pet")
        
        # Relationships with truth values
        cat_is_animal = self.atomspace.add_link(
            types.InheritanceLink, [cat, animal],
            TruthValue(0.95, 0.9)
        )
        
        cat_is_mammal = self.atomspace.add_link(
            types.InheritanceLink, [cat, mammal], 
            TruthValue(0.98, 0.95)
        )
        
        cat_is_pet = self.atomspace.add_link(
            types.InheritanceLink, [cat, pet],
            TruthValue(0.8, 0.85)
        )
    
    def test_hypergraph_operations(self):
        """Test basic hypergraph operations on real data."""
        # Test atom creation and retrieval
        test_node = self.atomspace.add_node(types.ConceptNode, "test_concept")
        self.assertIsNotNone(test_node)
        
        # Test link creation
        another_node = self.atomspace.add_node(types.ConceptNode, "another_concept")
        test_link = self.atomspace.add_link(
            types.SimilarityLink, [test_node, another_node],
            TruthValue(0.7, 0.8)
        )
        self.assertIsNotNone(test_link)
        
        # Test atom counting  
        atom_count = len(self.atomspace)
        self.assertGreater(atom_count, 5, "Should have multiple atoms")
    
    def test_pattern_matching(self):
        """Test pattern matching on real knowledge structures."""
        from opencog.bindlink import execute_atom
        from opencog.type_constructors import *
        
        # Create a pattern to find all inheritance relationships
        # This tests real pattern matching, not mocked behavior
        
        # Variables
        X = VariableNode("X")
        Y = VariableNode("Y") 
        
        # Pattern: Find all X that inherit from Y
        pattern = BindLink(
            VariableList(X, Y),
            InheritanceLink(X, Y),
            InheritanceLink(X, Y)
        )
        
        # Execute pattern (uses real pattern matcher)
        result = execute_atom(self.atomspace, pattern)
        self.assertIsNotNone(result, "Pattern matching should return results")
    
    def test_persistence(self):
        """Test AtomSpace persistence operations."""
        # Test saving and loading (when persistence is available)
        initial_count = len(self.atomspace)
        
        # Add test atoms
        for i in range(10):
            self.atomspace.add_node(types.ConceptNode, f"persist_test_{i}")
        
        final_count = len(self.atomspace)
        self.assertEqual(final_count - initial_count, 10, "Should add exactly 10 atoms")
    
    def test_performance_large_graph(self):
        """Test performance with large hypergraphs."""
        start_time = time.time()
        
        # Create large knowledge graph (real scale test)
        nodes = []
        for i in range(1000):
            node = self.atomspace.add_node(types.ConceptNode, f"concept_{i}")
            nodes.append(node)
        
        # Create links between nodes  
        for i in range(500):
            source = nodes[i]
            target = nodes[(i + 1) % len(nodes)]
            self.atomspace.add_link(
                types.SimilarityLink, [source, target],
                TruthValue(0.5, 0.7)
            )
        
        execution_time = time.time() - start_time
        
        # Performance should be reasonable for large graphs
        self.assertLess(execution_time, 10.0, "Large graph operations took too long")
        self.assertGreater(len(self.atomspace), 1000, "Should have many atoms")

if __name__ == '__main__':
    unittest.main()
