import networkx as nx
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple, Any
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import random

logger = logging.getLogger(__name__)

@dataclass
class PruningGroup:
    """Represents a group of parameters that must be pruned together"""
    id: str
    layer_idx: int
    parameters: Dict[str, torch.Tensor]
    group_type: str  # 'attention_head' or 'mlp_channel'
    dependencies: Set[str] = field(default_factory=set)
    size: Optional[int] = None
    memory_size: Optional[float] = None
    importance_score: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize group metrics and validate parameters"""
        self._calculate_metrics()
        self._validate_parameters()
    
    def _calculate_metrics(self):
        """Calculate comprehensive metrics for the group"""
        # Calculate size metrics
        self.size = sum(p.numel() for p in self.parameters.values())
        self.memory_size = sum(p.numel() * p.element_size() 
                             for p in self.parameters.values()) / (1024 * 1024)  # MB
        
        # Calculate parameter statistics
        param_tensors = [p.flatten() for p in self.parameters.values()]
        if param_tensors:
            all_params = torch.cat(param_tensors)
            
            self.metrics.update({
                'mean': float(torch.mean(all_params)),
                'std': float(torch.std(all_params)),
                'l1_norm': float(torch.norm(all_params, p=1)),
                'l2_norm': float(torch.norm(all_params, p=2)),
                'sparsity': float(torch.mean((all_params == 0).float())),
                'max_abs': float(torch.max(torch.abs(all_params)))
            })
    
    def _validate_parameters(self):
        """Validate parameter consistency"""
        if not self.parameters:
            raise ValueError(f"Group {self.id} has no parameters")
        
        for name, param in self.parameters.items():
            if not isinstance(param, torch.Tensor):
                raise ValueError(f"Parameter {name} in group {self.id} is not a tensor")
            if not param.requires_grad:
                logger.warning(f"Parameter {name} in group {self.id} does not require gradients")
    
    def get_shapes(self) -> Dict[str, torch.Size]:
        """Get shapes of all parameters in group"""
        return {name: p.shape for name, p in self.parameters.items()}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert group to dictionary format"""
        return {
            'id': self.id,
            'layer_idx': self.layer_idx,
            'group_type': self.group_type,
            'size': self.size,
            'memory_size': self.memory_size,
            'importance_score': self.importance_score,
            'dependencies': list(self.dependencies),
            'metrics': self.metrics,
            'shapes': {name: list(shape) for name, shape in self.get_shapes().items()}
        }
    
class DependencyGraphBuilder:
    def __init__(self, 
                 model: nn.Module,
                 config: Dict[str, Any]):
        """
        Initialize dependency graph builder for last 20% of layers
        
        Args:
            model: The model to analyze
            config: Configuration dictionary with graph building parameters
        """
        self.model = model
        self.config = config
        self.groups: List[PruningGroup] = []
        self.graph = nx.DiGraph()
        
        # Track parameter mapping
        self.param_to_group: Dict[str, str] = {}
        
        # Calculate layer range for last 20%
        total_layers = len(self.model.model.layers)
        start_idx = int(total_layers * 0.92)  # Start at 80% mark
        self.layer_range = range(start_idx, total_layers)
        
        logger.info(f"Initializing DependencyGraphBuilder for layers {start_idx} to {total_layers-1}")
    
    def build(self) -> Tuple[List[PruningGroup], nx.DiGraph]:
        """Build complete dependency graph for last 20% of layers"""
        logger.info("Building dependency graph for last 20% of layers...")
        
        try:
            with tqdm(total=6, desc="Building Graph") as pbar:
                # Step 1: Create base groups
                print("Creating attention groups...")
                self._create_attention_groups()
                pbar.update(1)
                
                print("Creating MLP groups...")
                self._create_mlp_groups()
                pbar.update(1)
                
                # Step 2: Add dependencies
                print("Adding structural dependencies...")
                self._add_structural_dependencies()
                pbar.update(1)
                
                print("Adding dimensional dependencies...")
                # self._add_dimensional_dependencies()
                # pbar.update(1)
                
                dependency_config = self.config.get('pruning', {}).get('dependency', {})
                include_skip = dependency_config.get('include_skip_connections', False)
                optimize_graph = dependency_config.get('optimize_graph', True)
            
                # if include_skip:
                #     print("Adding skip connection dependencies...")
                #     self._add_skip_connection_dependencies()
                # pbar.update(1)
                
                # Step 3: Validate and optimize graph
                print("Validating and optimizing graph...")
                self._validate_graph()
                if optimize_graph:
                    self._optimize_graph()
                pbar.update(1)
            
            logger.info(f"Built dependency graph with {len(self.groups)} groups "
                       f"and {self.graph.number_of_edges()} dependencies")
            
            return self.groups, self.graph
            
        except Exception as e:
            logger.error(f"Error building dependency graph: {str(e)}")
            raise
    
    def _create_attention_groups(self):
        """Create groups for attention heads (last 20% of layers only)"""
        for layer_idx in tqdm(self.layer_range, desc="Creating attention groups"):
            attention = self.model.model.layers[layer_idx].self_attn
            num_heads = attention.num_heads
            head_dim = attention.head_dim
            
            for head_idx in range(num_heads):
                start_idx = head_idx * head_dim
                end_idx = (head_idx + 1) * head_dim
                
                parameters = {
                    f'q_proj_{head_idx}': attention.q_proj.weight[start_idx:end_idx],
                    f'k_proj_{head_idx}': attention.k_proj.weight[start_idx:end_idx],
                    f'v_proj_{head_idx}': attention.v_proj.weight[start_idx:end_idx],
                    f'o_proj_{head_idx}': attention.o_proj.weight[:, start_idx:end_idx]
                }
                
                group = PruningGroup(
                    id=f"attn_l{layer_idx}_h{head_idx}",
                    layer_idx=layer_idx,
                    parameters=parameters,
                    group_type="attention_head"
                )
                
                self.groups.append(group)
                self.graph.add_node(group.id)
                
                for param_name in parameters.keys():
                    full_name = f"layer_{layer_idx}_attention_{param_name}"
                    self.param_to_group[full_name] = group.id
    
    def _create_mlp_groups(self):
        """Create groups for MLP channels (last 20% of layers only)"""
        for layer_idx in tqdm(self.layer_range, desc="Creating MLP groups"):
            mlp = self.model.model.layers[layer_idx].mlp
            hidden_size = mlp.gate_proj.weight.shape[0]
            
            for channel_idx in range(hidden_size):
                parameters = {
                    f'gate_proj_{channel_idx}': mlp.gate_proj.weight[channel_idx],
                    f'up_proj_{channel_idx}': mlp.up_proj.weight[channel_idx],
                    f'down_proj_{channel_idx}': mlp.down_proj.weight[:, channel_idx]
                }
                
                group = PruningGroup(
                    id=f"mlp_l{layer_idx}_c{channel_idx}",
                    layer_idx=layer_idx,
                    parameters=parameters,
                    group_type="mlp_channel"
                )
                
                self.groups.append(group)
                self.graph.add_node(group.id)
                
                for param_name in parameters.keys():
                    full_name = f"layer_{layer_idx}_mlp_{param_name}"
                    self.param_to_group[full_name] = group.id

    def _add_structural_dependencies(self):
        """Vectorized implementation of structural dependency analysis"""
        logger.info("Starting structural dependency analysis...")
        
        # Step 1: Create efficient data structures
        layer_groups = self._create_layer_groups_lookup()
        
        # Step 2: Process within-layer dependencies (faster path)
        self._process_within_layer_dependencies(layer_groups)
        
        # Step 3: Process cross-layer dependencies (vectorized)
        self._process_cross_layer_dependencies(layer_groups)
        
        logger.info(f"Completed dependency analysis with {self.graph.number_of_edges()} edges")

    def _create_layer_groups_lookup(self):
        """Create efficient lookup structure for groups by layer"""
        layer_groups = {}
        
        # Pre-allocate dictionaries for all layers
        for layer_idx in self.layer_range:
            layer_groups[layer_idx] = {'attention': [], 'mlp': []}
        
        # Single pass grouping
        for group in self.groups:
            group_type = 'attention' if group.group_type == 'attention_head' else 'mlp'
            layer_groups[group.layer_idx][group_type].append(group)
        
        return layer_groups

    def _process_within_layer_dependencies(self, layer_groups):
        """Process dependencies within each layer using vectorized operations"""
        dependencies_to_add = []
        
        for layer_idx in tqdm(self.layer_range, desc="Processing within-layer dependencies"):
            attn_groups = layer_groups[layer_idx]['attention']
            mlp_groups = layer_groups[layer_idx]['mlp']
            
            # Create direct connections (attention -> mlp)
            for attn_group in attn_groups:
                for mlp_group in mlp_groups:
                    dependencies_to_add.append((attn_group.id, mlp_group.id))
                    attn_group.dependencies.add(mlp_group.id)
        
        # Batch add dependencies
        if dependencies_to_add:
            self.graph.add_edges_from(dependencies_to_add)

    def _process_cross_layer_dependencies(self, layer_groups):
        """Process cross-layer dependencies using batched operations"""
        print("\nProcessing cross-layer dependencies...")
        layer_indices = list(self.layer_range)
        batch_size = 1000  # Adjustable batch size
        
        # Pre-compute dimension information
        dim_cache = self._create_dimension_cache()
        
        for i in range(len(layer_indices) - 1):
            curr_layer_idx = layer_indices[i]
            next_layer_idx = layer_indices[i + 1]
            
            # Get groups for current and next layer
            curr_groups = (layer_groups[curr_layer_idx]['attention'] + 
                          layer_groups[curr_layer_idx]['mlp'])
            next_groups = (layer_groups[next_layer_idx]['attention'] + 
                          layer_groups[next_layer_idx]['mlp'])
            
            # Process in batches
            dependencies = []
            for idx in range(0, len(curr_groups), batch_size):
                batch_curr = curr_groups[idx:idx + batch_size]
                self._process_dependency_batch(
                    batch_curr, next_groups, dependencies, dim_cache
                )
                
                # Add batch of dependencies
                if len(dependencies) >= batch_size:
                    self.graph.add_edges_from(dependencies)
                    dependencies = []
            
            # Add remaining dependencies
            if dependencies:
                self.graph.add_edges_from(dependencies)
            
            print(f"Processed layer {curr_layer_idx} → {next_layer_idx}")

    def _create_dimension_cache(self):
        """Create cached dimension information for all groups"""
        dim_cache = {}
        
        for group in self.groups:
            input_dims = set()
            output_dims = set()
            
            # Process each parameter's dimensions
            for param in group.parameters.values():
                if param.dim() >= 2:  # Only process 2D+ tensors
                    input_dims.add(param.size(0))
                    output_dims.add(param.size(-1))
            
            dim_cache[group.id] = {
                'input': input_dims,
                'output': output_dims
            }
        
        return dim_cache

    def _process_dependency_batch(self, curr_batch, next_groups, dependencies, dim_cache):
        """Process a batch of dependencies efficiently"""
        for curr_group in curr_batch:
            curr_dims = dim_cache[curr_group.id]
            
            # Find matching dimensions
            for next_group in next_groups:
                next_dims = dim_cache[next_group.id]
                
                # Quick set intersection check
                if curr_dims['output'] & next_dims['input']:
                    dependencies.append((curr_group.id, next_group.id))
                    curr_group.dependencies.add(next_group.id)

    def _check_structural_dependency(self, group1: PruningGroup, group2: PruningGroup) -> bool:
        """Optimized structural dependency check using cached dimensions"""
        # Early exit for non-adjacent layers
        layer_diff = abs(group1.layer_idx - group2.layer_idx)
        if layer_diff > 1:
            return False
        
        # Within same layer: only attention -> mlp is valid
        if layer_diff == 0:
            return (group1.group_type == 'attention_head' and 
                    group2.group_type == 'mlp_channel')
        
        # Use cached dimensions for cross-layer check
        dims1 = {p.size(-1) for p in group1.parameters.values()}
        dims2 = {p.size(0) for p in group2.parameters.values()}
        
        return bool(dims1 & dims2)

    def _add_dimensional_dependencies(self):
        """Add dependencies based on shared dimensions (strictly within last 20% of layers)"""
        total_groups = len(self.groups)
        with tqdm(total=total_groups, desc="Adding dimensional dependencies") as pbar:
            # Only consider groups within the last 20% of layers
            valid_groups = [g for g in self.groups if g.layer_idx in self.layer_range]
            
            for i, g1 in enumerate(valid_groups):
                for g2 in valid_groups[i+1:]:
                    if self._check_dimensional_dependency(g1, g2):
                        self.graph.add_edge(g1.id, g2.id)
                        g1.dependencies.add(g2.id)
                pbar.update(1)

    def _add_skip_connection_dependencies(self):
        """Add dependencies from skip connections (strictly within last 20% of layers)"""
        with tqdm(total=len(self.layer_range), desc="Adding skip connection dependencies") as pbar:
            layer_indices = list(self.layer_range)
            for i in range(len(layer_indices)):
                curr_layer_idx = layer_indices[i]
                
                if hasattr(self.model.model.layers[curr_layer_idx], 'skip_connection'):
                    # Only consider skip connections within the last 20%
                    if i >= 2:  # Make sure we have enough previous layers within our range
                        curr_groups = [g for g in self.groups if g.layer_idx == curr_layer_idx]
                        prev_groups = [g for g in self.groups if g.layer_idx == layer_indices[i-2]]
                        
                        for curr_group in curr_groups:
                            for prev_group in prev_groups:
                                self.graph.add_edge(curr_group.id, prev_group.id)
                                curr_group.dependencies.add(prev_group.id)
                pbar.update(1)
        
    def _check_structural_dependency(self, group1: PruningGroup, group2: PruningGroup) -> bool:
        """
        Check if two groups have structural dependency.
        Optimized version with early returns and dimension caching.
        
        Args:
            group1: First pruning group
            group2: Second pruning group
            
        Returns:
            bool: True if groups have structural dependency
        """
        # Quick check for same layer (attention -> mlp only)
        if group1.layer_idx == group2.layer_idx:
            if group1.group_type == "attention_head" and group2.group_type == "mlp_channel":
                return True
            if group1.group_type == "mlp_channel" and group2.group_type == "attention_head":
                return True
            return False
        
        # For cross-layer dependencies, only check adjacent layers
        if abs(group1.layer_idx - group2.layer_idx) != 1:
            return False
        
        # Cache shapes for both groups
        shapes1 = list(group1.get_shapes().values())
        shapes2 = list(group2.get_shapes().values())
        
        # Get dimensions of interest
        dims1 = {(shape[0], shape[-1]) for shape in shapes1}  # Input and output dimensions
        dims2 = {(shape[0], shape[-1]) for shape in shapes2}
        
        # Check for any matching dimensions
        for dim1_in, dim1_out in dims1:
            for dim2_in, dim2_out in dims2:
                # Forward dependency: output of group1 feeds into input of group2
                if dim1_out == dim2_in:
                    return True
                # Backward dependency: output of group2 feeds into input of group1
                if dim2_out == dim1_in:
                    return True
        
        return False
    
    def _check_dimensional_dependency(self, group1: PruningGroup, group2: PruningGroup) -> bool:
        """Check if two groups share dimensions"""
        shapes1 = group1.get_shapes()
        shapes2 = group2.get_shapes()
        
        for shape1 in shapes1.values():
            for shape2 in shapes2.values():
                if any(d1 == d2 for d1, d2 in zip(shape1, shape2)):
                    return True
        return False
    
    def _optimize_graph(self):
        """Optimized graph structure cleanup"""
        print("Optimizing graph structure...")
        edges_before = self.graph.number_of_edges()
        
        # Instead of checking each edge individually, use batch processing
        def remove_transitive_edges_batch():
            # Get transitive closure once
            closure = nx.transitive_closure(self.graph)
            
            # Find all direct edges that can be removed
            edges_to_remove = []
            
            # Process in layers to identify redundant edges
            for node in self.graph.nodes():
                successors = list(self.graph.successors(node))
                for succ in successors:
                    # Check if there's an indirect path (length > 1) between node and successor
                    indirect_paths = list(nx.all_simple_paths(self.graph, node, succ))
                    if any(len(path) > 2 for path in indirect_paths):
                        edges_to_remove.append((node, succ))
            
            # Batch remove edges
            print(f"Removing {len(edges_to_remove)} transitive edges...")
            self.graph.remove_edges_from(edges_to_remove)
            
            return len(edges_to_remove)

        # Alternative optimization approach using strongly connected components
        def optimize_using_scc():
            print("Optimizing using strongly connected components...")
            sccs = list(nx.strongly_connected_components(self.graph))
            
            # Collapse strongly connected components
            mapping = {}
            for i, scc in enumerate(sccs):
                for node in scc:
                    mapping[node] = f"scc_{i}"
            
            # Create reduced graph
            reduced = nx.condensation(self.graph, scc=sccs)
            
            # Rebuild graph with minimal edges
            new_graph = nx.DiGraph()
            new_graph.add_nodes_from(self.graph.nodes())
            
            # Add essential edges back
            for u, v in reduced.edges():
                original_u = [n for n in sccs[u]]
                original_v = [n for n in sccs[v]]
                # Add only one edge between components
                new_graph.add_edge(original_u[0], original_v[0])
            
            return new_graph

        # Choose optimization strategy based on graph size
        if self.graph.number_of_edges() > 100000:
            print("Large graph detected, using SCC-based optimization...")
            self.graph = optimize_using_scc()
        else:
            print("Using batch transitive edge removal...")
            edges_removed = remove_transitive_edges_batch()
            print(f"Removed {edges_removed} transitive edges")
        
        edges_after = self.graph.number_of_edges()
        reduction = (edges_before - edges_after) / edges_before * 100
        print(f"Reduced edges by {reduction:.1f}% ({edges_before} → {edges_after})")
        
        # Update group dependencies to match optimized graph
        print("Updating group dependencies...")
        for group in tqdm(self.groups):
            group.dependencies = set(self.graph.predecessors(group.id))

    def _validate_graph(self):
        """Validate graph structure with early stopping"""
        print("Validating graph structure...")
        
        # Quick validation checks
        if not nx.is_directed_acyclic_graph(self.graph):
            # Find a single cycle instead of all cycles
            cycle = next(nx.simple_cycles(self.graph), None)
            if cycle:
                raise ValueError(f"Dependency graph contains cycle: {cycle}")
        
        # Verify layer range for nodes (sample-based validation)
        sample_size = min(1000, len(self.graph.nodes()))
        nodes = random.sample(list(self.graph.nodes()), sample_size)
        
        for node in nodes:
            layer_idx = int(node.split('_l')[1].split('_')[0])
            if layer_idx not in self.layer_range:
                raise ValueError(f"Found node from outside target layer range: {node}")
        
        # Check for isolated nodes
        isolated = list(nx.isolates(self.graph))
        if isolated:
            logger.warning(f"Found {len(isolated)} isolated groups")
        
        print("Validating parameter assignments...")
        # Validate parameters (sampling-based approach for large graphs)
        model_params = {name: param for name, param in self.model.named_parameters()
                      if any(f"layers.{idx}" in name for idx in self.layer_range)}
        
        grouped_params = set().union(*[set(g.parameters.keys()) for g in self.groups])
        ungrouped = set(model_params.keys()) - grouped_params
        
        if ungrouped:
            logger.warning(f"Found {len(ungrouped)} ungrouped parameters in target layers")
            
            print("Validating parameter assignments...")
            # Only consider parameters from the last 20% of layers
            model_params = {name: param for name, param in self.model.named_parameters()
                        if any(f"layers.{idx}" in name for idx in self.layer_range)}
            grouped_params = set().union(*[set(g.parameters.keys()) for g in self.groups])
            ungrouped = set(model_params.keys()) - grouped_params
            if ungrouped:
                logger.warning(f"Found {len(ungrouped)} ungrouped parameters in target layers")

    def visualize(self, output_path: str):
        """Visualize dependency graph with non-interactive backend"""
        print("Generating graph visualization...")
        
        # Set non-interactive backend before importing pyplot
        import matplotlib
        matplotlib.use('Agg')  # Use Agg backend for non-interactive environments
        import matplotlib.pyplot as plt
        
        try:
            # Create figure with specified size
            plt.figure(figsize=(20, 20))
            
            print("Computing layout...")
            # Use spring layout with optimized parameters for large graphs
            pos = nx.spring_layout(self.graph, k=1.5, iterations=50)
            
            # Draw nodes with different colors for attention and MLP
            print("Drawing nodes...")
            node_colors = ['lightblue' if g.group_type == 'attention_head' else 'lightgreen'
                          for g in self.groups]
            
            nx.draw_networkx_nodes(self.graph, pos,
                                node_color=node_colors,
                                node_size=500)
            
            print("Drawing edges...")
            nx.draw_networkx_edges(self.graph, pos,
                                edge_color='gray',
                                arrows=True,
                                alpha=0.5)
            
            print("Adding labels...")
            # Use smaller subset of labels for large graphs
            if len(self.groups) > 1000:
                sample_size = min(1000, len(self.groups))
                sampled_groups = random.sample(self.groups, sample_size)
                labels = {g.id: f"{g.id}\n{g.memory_size:.1f}MB" for g in sampled_groups}
            else:
                labels = {g.id: f"{g.id}\n{g.memory_size:.1f}MB" for g in self.groups}
                
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
            
            # Add title and legend
            plt.title("Parameter Group Dependencies (Last 20% of Layers)", fontsize=16, pad=20)
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=c, label=l, markersize=10)
                            for c, l in [('lightblue', 'Attention Head'),
                                        ('lightgreen', 'MLP Channel')]]
            plt.legend(handles=legend_elements, loc='upper right')
            
            print(f"Saving visualization to {output_path}...")
            # Ensure the output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with high DPI but limit size for very large graphs
            if len(self.groups) > 5000:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
            else:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                
            plt.close()  # Explicitly close the figure
            logger.info(f"Saved dependency graph visualization to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            plt.close()  # Ensure figure is closed even if there's an error
        
    def save_graph(self, output_path: str):
        """Save graph and groups to file"""
        print("Preparing graph data for saving...")
        save_data = {
            'groups': [g.to_dict() for g in tqdm(self.groups, desc="Processing groups")],
            'edges': list(self.graph.edges()),
            'param_to_group': self.param_to_group,
            'config': self.config
        }
        
        print(f"Saving graph to {output_path}...")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_data, output_path)
        logger.info(f"Saved dependency graph to {output_path}")
    
    @classmethod
    def load_graph(cls, 
                  model: nn.Module,
                  load_path: str) -> Tuple['DependencyGraphBuilder', List[PruningGroup], nx.DiGraph]:
        """Load graph and groups from file"""
        print(f"Loading graph from {load_path}...")
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"No saved graph found at {load_path}")
        
        save_data = torch.load(load_path)
        
        # Create instance
        print("Creating graph builder instance...")
        instance = cls(model, save_data['config'])
        
        # Recreate groups
        print("Recreating groups...")
        instance.groups = []
        for group_dict in tqdm(save_data['groups'], desc="Processing groups"):
            # Get parameters for this group
            parameters = {}
            for param_name, shape in group_dict['shapes'].items():
                # Find parameter in model
                param_path = param_name.split('.')
                current = model
                for comp in param_path:
                    current = getattr(current, comp)
                parameters[param_name] = current
            
            # Create group
            group = PruningGroup(
                id=group_dict['id'],
                layer_idx=group_dict['layer_idx'],
                parameters=parameters,
                group_type=group_dict['group_type'],
                dependencies=set(group_dict['dependencies']),
                importance_score=group_dict['importance_score']
            )
            instance.groups.append(group)
        
        print("Recreating graph structure...")
        # Recreate graph
        instance.graph = nx.DiGraph()
        instance.graph.add_nodes_from([g.id for g in instance.groups])
        instance.graph.add_edges_from(save_data['edges'])
        
        # Restore parameter mapping
        instance.param_to_group = save_data['param_to_group']
        
        logger.info(f"Loaded dependency graph with {len(instance.groups)} groups "
                   f"and {instance.graph.number_of_edges()} dependencies")
        
        return instance, instance.groups, instance.graph
    
    def get_prunable_groups(self) -> List[PruningGroup]:
        """Get groups that can be pruned without breaking dependencies"""
        prunable = []
        for group in tqdm(self.groups, desc="Finding prunable groups"):
            # Check if all dependencies are satisfied
            if not group.dependencies or all(dep in self.param_to_group 
                                          for dep in group.dependencies):
                prunable.append(group)
        return prunable
    
    def get_group_by_id(self, group_id: str) -> Optional[PruningGroup]:
        """Get group by ID"""
        for group in self.groups:
            if group.id == group_id:
                return group
        return None
    
    def get_dependent_groups(self, group: PruningGroup) -> List[PruningGroup]:
        """Get all groups that depend on the given group"""
        print(f"Finding groups dependent on {group.id}...")
        dependent_ids = list(self.graph.successors(group.id))
        return [self.get_group_by_id(gid) for gid in tqdm(dependent_ids, desc="Processing dependent groups")]
    
    def get_dependency_groups(self, group: PruningGroup) -> List[PruningGroup]:
        """Get all groups that the given group depends on"""
        print(f"Finding dependencies for {group.id}...")
        dependency_ids = list(self.graph.predecessors(group.id))
        return [self.get_group_by_id(gid) for gid in tqdm(dependency_ids, desc="Processing dependency groups")]
    
    def analyze_graph(self) -> Dict[str, Any]:
        """Analyze graph properties"""
        print("Analyzing graph properties...")
        analysis = {
            'num_groups': len(self.groups),
            'num_edges': self.graph.number_of_edges(),
            'num_attention_heads': len([g for g in tqdm(self.groups, desc="Counting attention heads") 
                                     if g.group_type == 'attention_head']),
            'num_mlp_channels': len([g for g in tqdm(self.groups, desc="Counting MLP channels") 
                                   if g.group_type == 'mlp_channel']),
            'avg_dependencies': np.mean([len(g.dependencies) for g in tqdm(self.groups, desc="Calculating dependencies")]),
            'max_dependencies': max(len(g.dependencies) for g in self.groups),
            'isolated_groups': len(list(nx.isolates(self.graph))),
            'strongly_connected_components': len(list(nx.strongly_connected_components(self.graph))),
            'avg_path_length': nx.average_shortest_path_length(self.graph) 
                             if nx.is_strongly_connected(self.graph) else None,
            'diameter': nx.diameter(self.graph) 
                       if nx.is_strongly_connected(self.graph) else None,
            'density': nx.density(self.graph),
            'layer_distribution': self._get_layer_distribution()
        }
        
        return analysis
    
    def _get_layer_distribution(self) -> Dict[int, Dict[str, int]]:
        """Get distribution of groups across layers"""
        print("Analyzing layer distribution...")
        distribution = {}
        for group in tqdm(self.groups, desc="Processing groups"):
            if group.layer_idx not in distribution:
                distribution[group.layer_idx] = {
                    'attention_heads': 0,
                    'mlp_channels': 0
                }
            
            if group.group_type == 'attention_head':
                distribution[group.layer_idx]['attention_heads'] += 1
            else:
                distribution[group.layer_idx]['mlp_channels'] += 1
        
        return distribution
    
    def get_memory_distribution(self) -> Dict[str, float]:
        """Get memory distribution across group types"""
        print("Calculating memory distribution...")
        memory_dist = {
            'attention_heads': sum(g.memory_size for g in tqdm(
                [g for g in self.groups if g.group_type == 'attention_head'],
                desc="Processing attention heads"
            )),
            'mlp_channels': sum(g.memory_size for g in tqdm(
                [g for g in self.groups if g.group_type == 'mlp_channel'],
                desc="Processing MLP channels"
            ))
        }
        memory_dist['total'] = memory_dist['attention_heads'] + memory_dist['mlp_channels']
        return memory_dist
    
    def __str__(self) -> str:
        """String representation of the dependency graph"""
        analysis = self.analyze_graph()
        memory_dist = self.get_memory_distribution()
        
        return (
            f"Dependency Graph Summary (Last 20% of Layers):\n"
            f"  - Layers: {min(self.layer_range)} to {max(self.layer_range)}\n"
            f"  - Total Groups: {analysis['num_groups']}\n"
            f"  - Attention Heads: {analysis['num_attention_heads']}\n"
            f"  - MLP Channels: {analysis['num_mlp_channels']}\n"
            f"  - Total Dependencies: {analysis['num_edges']}\n"
            f"  - Average Dependencies: {analysis['avg_dependencies']:.2f}\n"
            f"  - Total Memory: {memory_dist['total']:.2f} MB\n"
            f"    - Attention Memory: {memory_dist['attention_heads']:.2f} MB\n"
            f"    - MLP Memory: {memory_dist['mlp_channels']:.2f} MB"
        )