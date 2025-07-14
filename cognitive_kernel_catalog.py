#!/usr/bin/env python3
"""
Cognitive Kernel Catalog: Dynamic Dictionary of OpenCog Primitives

This module implements a dynamic catalog system that discovers and documents
all cognitive primitives across OpenCog subsystems, including their tensor
shape signatures and adaptation patterns.
"""

import os
import sys
import json
import inspect
import importlib
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re

@dataclass
class TensorSignature:
    """Represents tensor shape requirements for a cognitive primitive."""
    input_shape: List[str]
    output_shape: List[str]
    required_memory: str
    complexity: str = "O(n)"

@dataclass
class AdaptationPattern:
    """Describes the learning and adaptation behavior of a primitive."""
    learning_type: str  # supervised, unsupervised, reinforcement, none
    convergence_rate: str  # fast, medium, slow, unknown
    resource_requirements: str  # low, medium, high
    adaptation_mechanism: str = "unknown"

@dataclass
class PerformanceMetrics:
    """Performance characteristics of a cognitive primitive."""
    cpu_complexity: str
    memory_complexity: str
    benchmark_results: Dict[str, Any]
    typical_execution_time: str = "unknown"

@dataclass
class CognitivePrimitive:
    """Complete specification of a cognitive primitive."""
    primitive_id: str
    name: str
    subsystem: str
    description: str
    tensor_signature: TensorSignature
    adaptation_pattern: AdaptationPattern
    performance_metrics: PerformanceMetrics
    dependencies: List[str]
    status: str  # active, deprecated, experimental, broken
    file_path: str
    line_number: int
    last_updated: str

class CognitiveKernelCatalog:
    """Main catalog system for cognitive primitives discovery and management."""
    
    def __init__(self, opencog_root: str):
        self.opencog_root = Path(opencog_root)
        self.catalog: Dict[str, CognitivePrimitive] = {}
        self.subsystems = [
            "atomspace", "pln", "moses", "asmoses", "attention", 
            "relex", "ure", "learn", "spacetime", "cogutil",
            "agents", "cogserver", "vision", "sensory"
        ]
        
    def discover_primitives(self) -> Dict[str, CognitivePrimitive]:
        """Discover all cognitive primitives across OpenCog subsystems."""
        print("🔍 Discovering cognitive primitives across OpenCog subsystems...")
        
        discovered_count = 0
        
        for subsystem in self.subsystems:
            subsystem_path = self.opencog_root / subsystem
            if not subsystem_path.exists():
                print(f"⚠️  Subsystem {subsystem} not found at {subsystem_path}")
                continue
                
            print(f"📂 Scanning {subsystem}...")
            primitives = self._scan_subsystem(subsystem, subsystem_path)
            discovered_count += len(primitives)
            
            for primitive in primitives:
                self.catalog[primitive.primitive_id] = primitive
                
        print(f"✅ Discovered {discovered_count} cognitive primitives")
        return self.catalog
    
    def _scan_subsystem(self, subsystem: str, path: Path) -> List[CognitivePrimitive]:
        """Scan a specific subsystem for cognitive primitives."""
        primitives = []
        
        # Scan Python files
        for py_file in path.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            primitives.extend(self._analyze_python_file(subsystem, py_file))
        
        # Scan Scheme files for cognitive functions
        for scm_file in path.rglob("*.scm"):
            primitives.extend(self._analyze_scheme_file(subsystem, scm_file))
            
        # Scan C++ files for exported functions
        for cpp_file in path.rglob("*.{cc,cpp,h,hpp}"):
            primitives.extend(self._analyze_cpp_file(subsystem, cpp_file))
            
        return primitives
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Determine if a file should be skipped during analysis."""
        skip_patterns = [
            "test", "tests", "__pycache__", ".git", "build", 
            "CMakeFiles", "node_modules", "example"
        ]
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _analyze_python_file(self, subsystem: str, file_path: Path) -> List[CognitivePrimitive]:
        """Analyze a Python file for cognitive primitive definitions."""
        primitives = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content, filename=str(file_path))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    primitive = self._create_primitive_from_function(
                        subsystem, file_path, node, content
                    )
                    if primitive:
                        primitives.append(primitive)
                        
                elif isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            primitive = self._create_primitive_from_method(
                                subsystem, file_path, node, item, content
                            )
                            if primitive:
                                primitives.append(primitive)
                                
        except Exception as e:
            print(f"⚠️  Error analyzing {file_path}: {e}")
            
        return primitives
    
    def _analyze_scheme_file(self, subsystem: str, file_path: Path) -> List[CognitivePrimitive]:
        """Analyze a Scheme file for cognitive function definitions."""
        primitives = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple regex to find define statements
            define_pattern = r'\(define\s+\(([^)]+)\)'
            matches = re.finditer(define_pattern, content)
            
            for match in matches:
                func_signature = match.group(1)
                func_name = func_signature.split()[0] if func_signature.split() else "unknown"
                
                primitive = CognitivePrimitive(
                    primitive_id=f"{subsystem}:scheme:{func_name}",
                    name=func_name,
                    subsystem=subsystem,
                    description=f"Scheme function from {file_path.name}",
                    tensor_signature=TensorSignature(["unknown"], ["unknown"], "unknown"),
                    adaptation_pattern=AdaptationPattern("none", "unknown", "unknown"),
                    performance_metrics=PerformanceMetrics("unknown", "unknown", {}),
                    dependencies=[],
                    status="active",
                    file_path=str(file_path),
                    line_number=content[:match.start()].count('\n') + 1,
                    last_updated=datetime.now().isoformat()
                )
                primitives.append(primitive)
                
        except Exception as e:
            print(f"⚠️  Error analyzing Scheme file {file_path}: {e}")
            
        return primitives
    
    def _analyze_cpp_file(self, subsystem: str, file_path: Path) -> List[CognitivePrimitive]:
        """Analyze a C++ file for exported cognitive functions."""
        primitives = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple regex for function definitions
            func_pattern = r'(\w+\s+)(\w+)\s*\([^{]*\)\s*{'
            matches = re.finditer(func_pattern, content)
            
            for match in matches:
                func_name = match.group(2)
                
                # Skip common non-cognitive functions
                if func_name in ['main', 'init', 'cleanup', 'test', 'setup']:
                    continue
                    
                primitive = CognitivePrimitive(
                    primitive_id=f"{subsystem}:cpp:{func_name}",
                    name=func_name,
                    subsystem=subsystem,
                    description=f"C++ function from {file_path.name}",
                    tensor_signature=TensorSignature(["unknown"], ["unknown"], "unknown"),
                    adaptation_pattern=AdaptationPattern("none", "unknown", "unknown"),
                    performance_metrics=PerformanceMetrics("unknown", "unknown", {}),
                    dependencies=[],
                    status="active",
                    file_path=str(file_path),
                    line_number=content[:match.start()].count('\n') + 1,
                    last_updated=datetime.now().isoformat()
                )
                primitives.append(primitive)
                
        except Exception as e:
            print(f"⚠️  Error analyzing C++ file {file_path}: {e}")
            
        return primitives
    
    def _create_primitive_from_function(self, subsystem: str, file_path: Path, 
                                      node: ast.FunctionDef, content: str) -> Optional[CognitivePrimitive]:
        """Create a cognitive primitive from a Python function definition."""
        
        # Skip private and test functions
        if node.name.startswith('_') or 'test' in node.name.lower():
            return None
            
        # Extract docstring
        docstring = ast.get_docstring(node) or "No description available"
        
        # Analyze function complexity
        complexity = self._estimate_complexity(node)
        
        # Determine adaptation pattern based on function characteristics
        adaptation = self._infer_adaptation_pattern(node, docstring)
        
        primitive = CognitivePrimitive(
            primitive_id=f"{subsystem}:python:{node.name}",
            name=node.name,
            subsystem=subsystem,
            description=docstring[:200] + "..." if len(docstring) > 200 else docstring,
            tensor_signature=TensorSignature(
                ["inferred"], ["inferred"], 
                self._estimate_memory_usage(node)
            ),
            adaptation_pattern=adaptation,
            performance_metrics=PerformanceMetrics(
                complexity, complexity, {}
            ),
            dependencies=self._extract_dependencies(node),
            status="active",
            file_path=str(file_path),
            line_number=node.lineno,
            last_updated=datetime.now().isoformat()
        )
        
        return primitive
    
    def _create_primitive_from_method(self, subsystem: str, file_path: Path,
                                    class_node: ast.ClassDef, method_node: ast.FunctionDef,
                                    content: str) -> Optional[CognitivePrimitive]:
        """Create a cognitive primitive from a class method."""
        
        # Skip private methods and common non-cognitive methods
        if (method_node.name.startswith('_') or 
            method_node.name in ['__init__', '__str__', '__repr__']):
            return None
            
        docstring = ast.get_docstring(method_node) or "No description available"
        complexity = self._estimate_complexity(method_node)
        adaptation = self._infer_adaptation_pattern(method_node, docstring)
        
        primitive = CognitivePrimitive(
            primitive_id=f"{subsystem}:python:{class_node.name}.{method_node.name}",
            name=f"{class_node.name}.{method_node.name}",
            subsystem=subsystem,
            description=docstring[:200] + "..." if len(docstring) > 200 else docstring,
            tensor_signature=TensorSignature(
                ["inferred"], ["inferred"],
                self._estimate_memory_usage(method_node)
            ),
            adaptation_pattern=adaptation,
            performance_metrics=PerformanceMetrics(
                complexity, complexity, {}
            ),
            dependencies=self._extract_dependencies(method_node),
            status="active",
            file_path=str(file_path),
            line_number=method_node.lineno,
            last_updated=datetime.now().isoformat()
        )
        
        return primitive
    
    def _estimate_complexity(self, node: ast.AST) -> str:
        """Estimate computational complexity of a function."""
        complexity_indicators = {
            'for': 'O(n)', 'while': 'O(n)', 'nested_loop': 'O(n²)',
            'recursive': 'O(exp)', 'sort': 'O(n log n)'
        }
        
        # Count loops and recursive calls
        loops = 0
        recursive_calls = False
        
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)):
                loops += 1
            elif isinstance(child, ast.Call) and hasattr(child.func, 'id'):
                if hasattr(node, 'name') and child.func.id == node.name:
                    recursive_calls = True
                    
        if recursive_calls:
            return "O(exp)"
        elif loops > 1:
            return "O(n²)"
        elif loops == 1:
            return "O(n)"
        else:
            return "O(1)"
    
    def _estimate_memory_usage(self, node: ast.AST) -> str:
        """Estimate memory usage of a function."""
        # Simple heuristic based on data structure usage
        memory_indicators = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.List):
                memory_indicators.append("list")
            elif isinstance(child, ast.Dict):
                memory_indicators.append("dict")
                
        if len(memory_indicators) > 2:
            return "high"
        elif len(memory_indicators) > 0:
            return "medium"
        else:
            return "low"
    
    def _infer_adaptation_pattern(self, node: ast.AST, docstring: str) -> AdaptationPattern:
        """Infer adaptation pattern from function characteristics."""
        
        # Keywords that suggest learning types
        learning_keywords = {
            'supervised': ['train', 'fit', 'predict', 'supervised'],
            'unsupervised': ['cluster', 'reduce', 'discover', 'unsupervised'],
            'reinforcement': ['reward', 'policy', 'action', 'reinforcement'],
            'none': ['utility', 'helper', 'format', 'parse']
        }
        
        docstring_lower = docstring.lower()
        learning_type = 'none'
        
        for ltype, keywords in learning_keywords.items():
            if any(keyword in docstring_lower for keyword in keywords):
                learning_type = ltype
                break
        
        # Infer convergence rate
        if 'fast' in docstring_lower or 'quick' in docstring_lower:
            convergence = 'fast'
        elif 'slow' in docstring_lower or 'gradual' in docstring_lower:
            convergence = 'slow'
        else:
            convergence = 'medium'
            
        return AdaptationPattern(
            learning_type=learning_type,
            convergence_rate=convergence,
            resource_requirements='medium'
        )
    
    def _extract_dependencies(self, node: ast.AST) -> List[str]:
        """Extract function dependencies from AST node."""
        dependencies = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and hasattr(child.func, 'id'):
                dependencies.append(child.func.id)
                
        # Return unique dependencies, excluding built-ins
        builtins = {'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict'}
        return list(set(dep for dep in dependencies if dep not in builtins))
    
    def export_catalog(self, output_path: str, format: str = 'json') -> None:
        """Export the catalog to various formats."""
        
        if format == 'json':
            self._export_json(output_path)
        elif format == 'markdown':
            self._export_markdown(output_path)
        elif format == 'html':
            self._export_html(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_json(self, output_path: str) -> None:
        """Export catalog as JSON."""
        catalog_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_primitives': len(self.catalog),
                'subsystems_scanned': self.subsystems
            },
            'primitives': {pid: asdict(primitive) for pid, primitive in self.catalog.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(catalog_data, f, indent=2)
            
        print(f"📄 JSON catalog exported to: {output_path}")
    
    def _export_markdown(self, output_path: str) -> None:
        """Export catalog as Markdown documentation."""
        
        md_content = f"""# Cognitive Kernel Catalog

Generated: {datetime.now().isoformat()}
Total Primitives: {len(self.catalog)}

## Subsystem Overview

"""
        
        # Group by subsystem
        by_subsystem = {}
        for primitive in self.catalog.values():
            if primitive.subsystem not in by_subsystem:
                by_subsystem[primitive.subsystem] = []
            by_subsystem[primitive.subsystem].append(primitive)
        
        for subsystem, primitives in sorted(by_subsystem.items()):
            md_content += f"### {subsystem.title()} ({len(primitives)} primitives)\n\n"
            
            for primitive in primitives:
                md_content += f"#### {primitive.name}\n"
                md_content += f"- **Status**: {primitive.status}\n"
                md_content += f"- **Learning Type**: {primitive.adaptation_pattern.learning_type}\n"
                md_content += f"- **Complexity**: {primitive.performance_metrics.cpu_complexity}\n"
                md_content += f"- **File**: `{primitive.file_path}:{primitive.line_number}`\n"
                md_content += f"- **Description**: {primitive.description}\n\n"
        
        with open(output_path, 'w') as f:
            f.write(md_content)
            
        print(f"📝 Markdown catalog exported to: {output_path}")
    
    def _export_html(self, output_path: str) -> None:
        """Export catalog as interactive HTML."""
        
        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>OpenCog Cognitive Kernel Catalog</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .primitive {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; }}
        .subsystem {{ background-color: #f5f5f5; padding: 10px; margin: 20px 0; }}
        .status-active {{ color: green; }}
        .status-deprecated {{ color: orange; }}
        .status-experimental {{ color: blue; }}
    </style>
</head>
<body>
    <h1>OpenCog Cognitive Kernel Catalog</h1>
    <p>Generated: {datetime.now().isoformat()}</p>
    <p>Total Primitives: {len(self.catalog)}</p>
    
    <div id="catalog">
"""
        
        # Group by subsystem for HTML
        by_subsystem = {}
        for primitive in self.catalog.values():
            if primitive.subsystem not in by_subsystem:
                by_subsystem[primitive.subsystem] = []
            by_subsystem[primitive.subsystem].append(primitive)
        
        for subsystem, primitives in sorted(by_subsystem.items()):
            html_template += f'<div class="subsystem"><h2>{subsystem.title()}</h2>\n'
            
            for primitive in primitives:
                html_template += f'''
                <div class="primitive">
                    <h3>{primitive.name}</h3>
                    <p><strong>Status:</strong> <span class="status-{primitive.status}">{primitive.status}</span></p>
                    <p><strong>Learning Type:</strong> {primitive.adaptation_pattern.learning_type}</p>
                    <p><strong>Complexity:</strong> {primitive.performance_metrics.cpu_complexity}</p>
                    <p><strong>File:</strong> <code>{primitive.file_path}:{primitive.line_number}</code></p>
                    <p><strong>Description:</strong> {primitive.description}</p>
                </div>
                '''
            
            html_template += '</div>\n'
        
        html_template += """
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html_template)
            
        print(f"🌐 HTML catalog exported to: {output_path}")

def main():
    """Main entry point for the cognitive kernel catalog."""
    
    if len(sys.argv) < 2:
        print("Usage: python cognitive_kernel_catalog.py <opencog_root_directory>")
        print("Example: python cognitive_kernel_catalog.py /home/runner/work/opencog/opencog")
        return
    
    opencog_root = sys.argv[1]
    
    print("🧠 OpenCog Cognitive Kernel Catalog")
    print("=" * 50)
    
    catalog = CognitiveKernelCatalog(opencog_root)
    primitives = catalog.discover_primitives()
    
    # Export in multiple formats
    base_name = "cognitive_kernel_catalog"
    catalog.export_catalog(f"{base_name}.json", "json")
    catalog.export_catalog(f"{base_name}.md", "markdown") 
    catalog.export_catalog(f"{base_name}.html", "html")
    
    print(f"\n✅ Catalog generation complete!")
    print(f"📊 Summary:")
    print(f"   - Total primitives discovered: {len(primitives)}")
    
    # Subsystem breakdown
    by_subsystem = {}
    for primitive in primitives.values():
        subsystem = primitive.subsystem
        if subsystem not in by_subsystem:
            by_subsystem[subsystem] = 0
        by_subsystem[subsystem] += 1
    
    for subsystem, count in sorted(by_subsystem.items()):
        print(f"   - {subsystem}: {count} primitives")

if __name__ == "__main__":
    main()