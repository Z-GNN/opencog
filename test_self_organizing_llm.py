#!/usr/bin/env python3
"""
Test Suite for Self-Organizing LLM

This test suite validates the core functionality of the self-organizing LLM
system implementing distributed agentic cognitive grammar with GGML kernels.
"""

import asyncio
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np

# Import the components we're testing
from self_organizing_llm import (
    SelfOrganizingLLM, CognitiveGrammarToken, HypergraphFragment,
    SchemeAdapter, TensorFragmentArchitecture, ECANResourceKernel,
    SymbolicNeuralKernel, GrammarInferenceKernel, CognitiveMeshAPI,
    EmbodimentLayer, MetaCognitivePathway, EvolutionaryOptimizer
)

from ggml_cognitive_integration import TensorShape, GGMLTensorType, CognitiveTensor


class TestCognitiveGrammarParsing(unittest.TestCase):
    """Test cognitive grammar parsing and hypergraph fragment creation."""
    
    def setUp(self):
        self.adapter = SchemeAdapter()
        self.tensor_arch = TensorFragmentArchitecture()
    
    def test_basic_grammar_parsing(self):
        """Test basic agentic grammar parsing."""
        grammar_text = """
        FRAGMENT: test_fragment
        NODE: node1
        NODE: node2
        EDGE: TestLink(node1, node2)
        TAG: PERCEPTION
        TAG: ACTION
        """
        
        fragments = self.adapter.parse_agentic_grammar(grammar_text)
        
        self.assertEqual(len(fragments), 1)
        fragment = fragments[0]
        
        self.assertEqual(fragment.fragment_id, "test_fragment")
        self.assertEqual(len(fragment.nodes), 2)
        self.assertEqual(len(fragment.edges), 1)
        self.assertIn(CognitiveGrammarToken.PERCEPTION, fragment.semantic_tags)
        self.assertIn(CognitiveGrammarToken.ACTION, fragment.semantic_tags)
    
    def test_multiple_fragment_parsing(self):
        """Test parsing multiple fragments in one grammar."""
        grammar_text = """
        FRAGMENT: fragment1
        NODE: node1
        TAG: MEMORY
        
        FRAGMENT: fragment2
        NODE: node2
        TAG: REASONING
        """
        
        fragments = self.adapter.parse_agentic_grammar(grammar_text)
        
        self.assertEqual(len(fragments), 2)
        self.assertEqual(fragments[0].fragment_id, "fragment1")
        self.assertEqual(fragments[1].fragment_id, "fragment2")
    
    def test_tensor_shape_assignment(self):
        """Test tensor shape assignment based on fragment characteristics."""
        fragment = HypergraphFragment(
            fragment_id="test",
            nodes=["n1", "n2", "n3"],
            edges=[("EdgeType", ["n1", "n2"])],
            semantic_tags={CognitiveGrammarToken.PERCEPTION, CognitiveGrammarToken.NEURAL},
            tensor_signature=TensorShape([1], GGMLTensorType.FLOAT32, "temp")
        )
        
        tensor_shape = self.tensor_arch.assign_tensor_shape(fragment)
        
        self.assertEqual(len(tensor_shape.dimensions), 5)
        self.assertEqual(tensor_shape.dimensions[3], 3)  # Number of nodes
        self.assertEqual(tensor_shape.dimensions[4], 1)  # Number of edges
        self.assertIsInstance(fragment.prime_factorization, dict)
    
    def test_atomese_conversion(self):
        """Test conversion to Atomese representation."""
        fragment = HypergraphFragment(
            fragment_id="test",
            nodes=["concept1", "concept2"],
            edges=[("InheritanceLink", ["concept1", "concept2"])],
            semantic_tags=set(),
            tensor_signature=TensorShape([1], GGMLTensorType.FLOAT32, "temp")
        )
        
        atomese = fragment.to_atomese()
        
        self.assertIn('ConceptNode "concept1"', atomese)
        self.assertIn('ConceptNode "concept2"', atomese)
        self.assertIn('InheritanceLink', atomese)


class TestECANResourceAllocation(unittest.TestCase):
    """Test ECAN-style resource allocation and attention management."""
    
    def setUp(self):
        self.resource_kernel = ECANResourceKernel(max_agents=10, resource_budget=1000.0)
    
    def test_resource_allocation_basic(self):
        """Test basic resource allocation functionality."""
        # Create mock agent states [STI, LTI, VLTI, activation, priority]
        agent_states = np.array([
            [100, 50, 25, 0.8, 0.9],  # High priority agent
            [50, 30, 15, 0.4, 0.5],   # Medium priority agent
            [20, 10, 5, 0.2, 0.1]     # Low priority agent
        ], dtype=np.float32)
        
        input_tensor = CognitiveTensor(
            name="test_states",
            shape=TensorShape([3, 5], GGMLTensorType.FLOAT32, "Agent States"),
            data=agent_states,
            metadata={}
        )
        
        result = self.resource_kernel.forward([input_tensor])
        
        # Check that resources were allocated
        self.assertEqual(result.data.shape, (3, 3))  # 3 agents, 3 attention values
        
        # Check that total allocation is within budget
        total_allocated = result.metadata['total_allocated']
        self.assertLessEqual(total_allocated, self.resource_kernel.resource_budget)
        
        # Check that allocation efficiency is reasonable
        efficiency = result.metadata['allocation_efficiency']
        self.assertGreater(efficiency, 0.0)
        self.assertLessEqual(efficiency, 1.0)
    
    def test_resource_conservation(self):
        """Test that resource allocation respects budget constraints."""
        # Create high-demand scenario
        agent_states = np.array([
            [1000, 500, 250, 1.0, 1.0],  # Very high demand
            [1000, 500, 250, 1.0, 1.0],  # Very high demand
            [1000, 500, 250, 1.0, 1.0]   # Very high demand
        ], dtype=np.float32)
        
        input_tensor = CognitiveTensor(
            name="high_demand_states",
            shape=TensorShape([3, 5], GGMLTensorType.FLOAT32, "High Demand States"),
            data=agent_states,
            metadata={}
        )
        
        result = self.resource_kernel.forward([input_tensor])
        
        # Total allocation should not exceed budget
        total_allocated = result.metadata['total_allocated']
        self.assertLessEqual(total_allocated, self.resource_kernel.resource_budget * 1.01)  # Small tolerance


class TestNeuralSymbolicSynthesis(unittest.TestCase):
    """Test neural-symbolic synthesis functionality."""
    
    def setUp(self):
        self.synthesis_kernel = SymbolicNeuralKernel(symbolic_dim=64, neural_dim=128)
    
    def test_symbolic_neural_fusion(self):
        """Test symbolic-neural fusion process."""
        # Create mock symbolic and neural inputs
        symbolic_input = np.random.randn(64).astype(np.float32)
        neural_input = np.random.randn(128).astype(np.float32)
        
        symbolic_tensor = CognitiveTensor(
            name="symbolic_test",
            shape=TensorShape([64], GGMLTensorType.FLOAT32, "Symbolic"),
            data=symbolic_input,
            metadata={}
        )
        
        neural_tensor = CognitiveTensor(
            name="neural_test",
            shape=TensorShape([128], GGMLTensorType.FLOAT32, "Neural"),
            data=neural_input,
            metadata={}
        )
        
        result = self.synthesis_kernel.forward([symbolic_tensor, neural_tensor])
        
        # Check output dimensions
        self.assertEqual(result.data.shape, (192,))  # 64 + 128
        
        # Check metadata
        self.assertIn('symbolic_norm', result.metadata)
        self.assertIn('neural_norm', result.metadata)
        self.assertIn('fusion_entropy', result.metadata)
        
        # Check that fusion entropy is reasonable
        entropy = result.metadata['fusion_entropy']
        self.assertGreater(entropy, 0.0)
    
    def test_fusion_weights_normalization(self):
        """Test that fusion weights are properly normalized."""
        # The softmax operation should ensure weights sum to 1
        weights = self.synthesis_kernel.fusion_weights
        normalized_weights = self.synthesis_kernel._softmax(weights)
        
        # Sum should be approximately 1
        self.assertAlmostEqual(np.sum(normalized_weights), 1.0, places=5)
        
        # All weights should be positive
        self.assertTrue(np.all(normalized_weights >= 0))


class TestGrammarInference(unittest.TestCase):
    """Test grammar inference functionality."""
    
    def setUp(self):
        self.inference_kernel = GrammarInferenceKernel(max_patterns=10, pattern_dim=32)
    
    def test_pattern_confidence_scoring(self):
        """Test pattern confidence scoring."""
        # Create mock pattern candidates
        pattern_candidates = np.random.randn(10, 32).astype(np.float32)
        
        input_tensor = CognitiveTensor(
            name="pattern_candidates",
            shape=TensorShape([10, 32], GGMLTensorType.FLOAT32, "Pattern Candidates"),
            data=pattern_candidates,
            metadata={}
        )
        
        result = self.inference_kernel.forward([input_tensor])
        
        # Check output shape
        self.assertEqual(result.data.shape, (10,))
        
        # Check that all confidence scores are valid
        self.assertTrue(np.all(result.data >= 0.0))
        self.assertTrue(np.all(result.data <= 1.0))
        
        # Check metadata
        self.assertIn('high_confidence_patterns', result.metadata)
        self.assertIn('average_confidence', result.metadata)
        self.assertIn('pattern_diversity', result.metadata)


class TestCognitiveMeshAPI(unittest.TestCase):
    """Test cognitive mesh API functionality."""
    
    def setUp(self):
        self.mesh_api = CognitiveMeshAPI()
    
    async def test_agent_registration(self):
        """Test agent registration in cognitive mesh."""
        agent_spec = {
            'capabilities': ['reasoning', 'perception'],
            'resources': {'memory': '4GB', 'compute': 'CPU'}
        }
        
        agent_id = await self.mesh_api.register_agent(agent_spec)
        
        self.assertTrue(agent_id.startswith('agent_'))
        self.assertIn(agent_id, self.mesh_api.agent_registry)
        
        registered_agent = self.mesh_api.agent_registry[agent_id]
        self.assertEqual(registered_agent['capabilities'], ['reasoning', 'perception'])
        self.assertEqual(registered_agent['status'], 'active')
    
    async def test_cognitive_request_processing(self):
        """Test cognitive request processing."""
        # First register an agent
        agent_spec = {'capabilities': ['processing']}
        await self.mesh_api.register_agent(agent_spec)
        
        # Process a request
        request = {'input': 'test cognitive task', 'type': 'reasoning'}
        result = await self.mesh_api.process_cognitive_request(request)
        
        self.assertIn('request_id', result)
        self.assertIn('agent_id', result)
        self.assertEqual(result['status'], 'completed')
        self.assertIn('result', result)


class TestMetaCognition(unittest.TestCase):
    """Test meta-cognitive analysis and adaptation."""
    
    def setUp(self):
        self.meta_pathway = MetaCognitivePathway()
    
    def test_performance_analysis_insufficient_data(self):
        """Test performance analysis with insufficient data."""
        # Mock agent for testing
        class MockAgent:
            agent_id = "test_agent"
        
        mock_agent = MockAgent()
        metrics = {'accuracy': 0.8, 'efficiency': 0.7}
        
        analysis = self.meta_pathway.analyze_cognitive_performance(mock_agent, metrics)
        
        self.assertIn('performance_analysis', analysis)
        self.assertEqual(analysis['performance_analysis']['trend'], 'insufficient_data')
        self.assertIn('meta_confidence', analysis)
    
    def test_adaptation_suggestions(self):
        """Test adaptation suggestions based on performance trends."""
        # Mock agent
        class MockAgent:
            agent_id = "test_agent"
        
        mock_agent = MockAgent()
        
        # Add enough performance history to enable trend analysis
        for i in range(15):
            metrics = {
                'accuracy': 0.9 - i * 0.02,  # Declining accuracy trend
                'efficiency': 0.8 - i * 0.01  # Declining efficiency trend
            }
            self.meta_pathway.analyze_cognitive_performance(mock_agent, metrics)
        
        # Latest analysis should suggest adaptations
        final_metrics = {'accuracy': 0.6, 'efficiency': 0.65}
        analysis = self.meta_pathway.analyze_cognitive_performance(mock_agent, final_metrics)
        
        adaptations = analysis['suggested_adaptations']
        self.assertGreater(len(adaptations), 0)
        
        # Should suggest learning rate adjustment for declining accuracy
        learning_rate_adaptations = [a for a in adaptations if a['type'] == 'learning_rate_adjustment']
        self.assertGreater(len(learning_rate_adaptations), 0)


class TestSelfOrganizingLLMIntegration(unittest.TestCase):
    """Integration tests for the complete self-organizing LLM system."""
    
    async def test_full_processing_pipeline(self):
        """Test the complete processing pipeline from grammar to results."""
        soll = SelfOrganizingLLM("test_agent")
        await soll.initialize()
        
        # Test grammar
        test_grammar = """
        FRAGMENT: test_cognition
        NODE: input
        NODE: process
        NODE: output
        EDGE: ProcessLink(input, process)
        EDGE: OutputLink(process, output)
        TAG: REASONING
        TAG: NEURAL
        """
        
        result = await soll.process_agentic_grammar(test_grammar)
        
        # Validate results structure
        self.assertIn('processing_id', result)
        self.assertIn('parsed_fragments', result)
        self.assertIn('processed_fragments', result)
        self.assertIn('attention_allocation', result)
        self.assertIn('meta_analysis', result)
        
        # Check that fragments were processed
        self.assertEqual(result['parsed_fragments'], 1)
        self.assertEqual(len(result['processed_fragments']), 1)
        
        fragment_result = result['processed_fragments'][0]
        self.assertEqual(fragment_result['fragment_id'], 'test_cognition')
        self.assertEqual(fragment_result['nodes'], 3)
        self.assertEqual(fragment_result['edges'], 2)
        self.assertIn('reasoning', fragment_result['semantic_tags'])
        self.assertIn('neural', fragment_result['semantic_tags'])
    
    async def test_autonomous_mode_operation(self):
        """Test autonomous mode startup and shutdown."""
        soll = SelfOrganizingLLM("autonomous_test")
        await soll.initialize()
        
        # Start autonomous mode
        await soll.start_autonomous_mode()
        self.assertTrue(soll.autonomous_mode)
        
        # Let it run briefly
        await asyncio.sleep(0.5)
        
        # Stop autonomous mode
        await soll.stop_autonomous_mode()
        self.assertFalse(soll.autonomous_mode)
    
    def test_cognitive_architecture_export(self):
        """Test cognitive architecture export functionality."""
        soll = SelfOrganizingLLM("export_test")
        
        architecture = soll.export_cognitive_architecture()
        
        # Validate export structure
        self.assertIn('agent_id', architecture)
        self.assertIn('system_status', architecture)
        self.assertIn('cognitive_components', architecture)
        self.assertIn('performance_metrics', architecture)
        self.assertIn('meta_cognitive_data', architecture)
        self.assertIn('hypergraph_statistics', architecture)
        self.assertIn('timestamp', architecture)
        
        # Check component registration
        components = architecture['cognitive_components']
        expected_components = [
            'scheme_adapter', 'tensor_architecture', 'resource_kernel',
            'symbolic_neural_kernel', 'grammar_inference_kernel'
        ]
        
        for component in expected_components:
            self.assertIn(component, components)


class AsyncTestCase(unittest.TestCase):
    """Base class for async test cases."""
    
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        self.loop.close()
    
    def async_test(self, coro):
        """Helper to run async tests."""
        return self.loop.run_until_complete(coro)


class TestAsyncOperations(AsyncTestCase):
    """Test async operations of the system."""
    
    def test_mesh_api_operations(self):
        """Test mesh API async operations."""
        async def run_mesh_tests():
            mesh_api = CognitiveMeshAPI()
            
            # Test agent registration
            agent_spec = {
                'capabilities': ['reasoning', 'perception'],
                'resources': {'memory': '4GB', 'compute': 'CPU'}
            }
            
            agent_id = await mesh_api.register_agent(agent_spec)
            
            self.assertTrue(agent_id.startswith('agent_'))
            self.assertIn(agent_id, mesh_api.agent_registry)
            
            registered_agent = mesh_api.agent_registry[agent_id]
            self.assertEqual(registered_agent['capabilities'], ['reasoning', 'perception'])
            self.assertEqual(registered_agent['status'], 'active')
            
            # Test cognitive request processing
            request = {'input': 'test cognitive task', 'type': 'reasoning'}
            result = await mesh_api.process_cognitive_request(request)
            
            self.assertIn('request_id', result)
            self.assertIn('agent_id', result)
            self.assertEqual(result['status'], 'completed')
            self.assertIn('result', result)
        
        self.async_test(run_mesh_tests())
    
    def test_full_pipeline(self):
        """Test full async pipeline."""
        async def run_pipeline_test():
            soll = SelfOrganizingLLM("test_agent")
            await soll.initialize()
            
            # Test grammar
            test_grammar = """
            FRAGMENT: test_cognition
            NODE: input
            NODE: process
            NODE: output
            EDGE: ProcessLink(input, process)
            EDGE: OutputLink(process, output)
            TAG: REASONING
            TAG: NEURAL
            """
            
            result = await soll.process_agentic_grammar(test_grammar)
            
            # Validate results structure
            self.assertIn('processing_id', result)
            self.assertIn('parsed_fragments', result)
            self.assertIn('processed_fragments', result)
            self.assertIn('attention_allocation', result)
            self.assertIn('meta_analysis', result)
            
            # Check that fragments were processed
            self.assertEqual(result['parsed_fragments'], 1)
            self.assertEqual(len(result['processed_fragments']), 1)
            
            fragment_result = result['processed_fragments'][0]
            self.assertEqual(fragment_result['fragment_id'], 'test_cognition')
            self.assertEqual(fragment_result['nodes'], 3)
            self.assertEqual(fragment_result['edges'], 2)
            self.assertIn('reasoning', fragment_result['semantic_tags'])
            self.assertIn('neural', fragment_result['semantic_tags'])
        
        self.async_test(run_pipeline_test())
    
    def test_autonomous_operation(self):
        """Test autonomous mode operation."""
        async def run_autonomous_test():
            soll = SelfOrganizingLLM("autonomous_test")
            await soll.initialize()
            
            # Start autonomous mode
            await soll.start_autonomous_mode()
            self.assertTrue(soll.autonomous_mode)
            
            # Let it run briefly
            await asyncio.sleep(0.5)
            
            # Stop autonomous mode
            await soll.stop_autonomous_mode()
            self.assertFalse(soll.autonomous_mode)
        
        self.async_test(run_autonomous_test())


def run_test_suite():
    """Run the complete test suite."""
    
    print("🧪 Running Self-Organizing LLM Test Suite")
    print("=" * 50)
    
    # Create test suite
    test_classes = [
        TestCognitiveGrammarParsing,
        TestECANResourceAllocation,
        TestNeuralSymbolicSynthesis,
        TestGrammarInference,
        TestMetaCognition,
        TestSelfOrganizingLLMIntegration,
        TestAsyncOperations
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nTest Result: {'✅ PASSED' if success else '❌ FAILED'}")
    
    return success


if __name__ == "__main__":
    run_test_suite()