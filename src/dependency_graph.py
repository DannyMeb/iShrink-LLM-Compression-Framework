import networkx as nx
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple, Any
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path

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
        Initialize dependency graph builder
        
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
        
        logger.info("Initializing DependencyGraphBuilder")
    
    def build(self) -> Tuple[List[PruningGroup], nx.DiGraph]:
        """Build complete dependency graph"""
        logger.info("Building dependency graph...")
        
        try:
            # Step 1: Create base groups
            self._create_attention_groups()
            self._create_mlp_groups()
            
            # Step 2: Add dependencies
            self._add_structural_dependencies()
            self._add_dimensional_dependencies()
            if self.config['dependency']['include_skip_connections']:
                self._add_skip_connection_dependencies()
            
            # Step 3: Validate graph
            self._validate_graph()
            
            # Step 4: Optimize graph
            if self.config['dependency']['optimize_graph']:
                self._optimize_graph()
            
            logger.info(f"Built dependency graph with {len(self.groups)} groups "
                       f"and {self.graph.number_of_edges()} dependencies")
            
            return self.groups, self.graph
            
        except Exception as e:
            logger.error(f"Error building dependency graph: {str(e)}")
            raise
    
    def _create_attention_groups(self):
        """Create groups for attention heads"""
        for layer_idx, layer in enumerate(self.model.model.layers):
            attention = layer.self_attn
            num_heads = attention.num_heads
            head_dim = attention.head_dim
            
            for head_idx in range(num_heads):
                # Calculate head slices
                start_idx = head_idx * head_dim
                end_idx = (head_idx + 1) * head_dim
                
                # Collect parameters for this head
                parameters = {
                    f'q_proj_{head_idx}': attention.q_proj.weight[start_idx:end_idx],
                    f'k_proj_{head_idx}': attention.k_proj.weight[start_idx:end_idx],
                    f'v_proj_{head_idx}': attention.v_proj.weight[start_idx:end_idx],
                    f'o_proj_{head_idx}': attention.o_proj.weight[:, start_idx:end_idx]
                }
                
                # Create group
                group = PruningGroup(
                    id=f"attn_l{layer_idx}_h{head_idx}",
                    layer_idx=layer_idx,
                    parameters=parameters,
                    group_type="attention_head"
                )
                
                self.groups.append(group)
                self.graph.add_node(group.id)
                
                # Update parameter mapping
                for param_name in parameters.keys():
                    full_name = f"layer_{layer_idx}_attention_{param_name}"
                    self.param_to_group[full_name] = group.id
    
    def _create_mlp_groups(self):
        """Create groups for MLP channels"""
        for layer_idx, layer in enumerate(self.model.model.layers):
            mlp = layer.mlp
            hidden_size = mlp.gate_proj.weight.shape[0]
            
            for channel_idx in range(hidden_size):
                # Collect parameters for this channel
                parameters = {
                    f'gate_proj_{channel_idx}': mlp.gate_proj.weight[channel_idx],
                    f'up_proj_{channel_idx}': mlp.up_proj.weight[channel_idx],
                    f'down_proj_{channel_idx}': mlp.down_proj.weight[:, channel_idx]
                }
                
                # Create group
                group = PruningGroup(
                    id=f"mlp_l{layer_idx}_c{channel_idx}",
                    layer_idx=layer_idx,
                    parameters=parameters,
                    group_type="mlp_channel"
                )
                
                self.groups.append(group)
                self.graph.add_node(group.id)
                
                # Update parameter mapping
                for param_name in parameters.keys():
                    full_name = f"layer_{layer_idx}_mlp_{param_name}"
                    self.param_to_group[full_name] = group.id
    
    def _add_structural_dependencies(self):
        """Add dependencies based on model structure"""
        # Within-layer dependencies
        for layer_idx in range(len(self.model.model.layers)):
            layer_groups = [g for g in self.groups if g.layer_idx == layer_idx]
            
            # Connect attention heads to MLP inputs
            attn_groups = [g for g in layer_groups if g.group_type == "attention_head"]
            mlp_groups = [g for g in layer_groups if g.group_type == "mlp_channel"]
            
            for attn_group in attn_groups:
                for mlp_group in mlp_groups:
                    if self._check_structural_dependency(attn_group, mlp_group):
                        self.graph.add_edge(mlp_group.id, attn_group.id)
                        mlp_group.dependencies.add(attn_group.id)
        
        # Cross-layer dependencies
        for layer_idx in range(1, len(self.model.model.layers)):
            curr_groups = [g for g in self.groups if g.layer_idx == layer_idx]
            prev_groups = [g for g in self.groups if g.layer_idx == layer_idx - 1]
            
            for curr_group in curr_groups:
                for prev_group in prev_groups:
                    if self._check_structural_dependency(curr_group, prev_group):
                        self.graph.add_edge(curr_group.id, prev_group.id)
                        curr_group.dependencies.add(prev_group.id)
    
    def _add_dimensional_dependencies(self):
        """Add dependencies based on shared dimensions"""
        for g1 in self.groups:
            for g2 in self.groups:
                if g1.id != g2.id:
                    if self._check_dimensional_dependency(g1, g2):
                        self.graph.add_edge(g1.id, g2.id)
                        g1.dependencies.add(g2.id)
    
    def _add_skip_connection_dependencies(self):
        """Add dependencies from skip connections"""
        for layer_idx in range(len(self.model.model.layers)):
            if hasattr(self.model.model.layers[layer_idx], 'skip_connection'):
                curr_groups = [g for g in self.groups if g.layer_idx == layer_idx]
                prev_groups = [g for g in self.groups if g.layer_idx == layer_idx - 2]
                
                for curr_group in curr_groups:
                    for prev_group in prev_groups:
                        self.graph.add_edge(curr_group.id, prev_group.id)
                        curr_group.dependencies.add(prev_group.id)
    
    def _check_structural_dependency(self, group1: PruningGroup, group2: PruningGroup) -> bool:
        """Check if two groups have structural dependency"""
        shapes1 = group1.get_shapes()
        shapes2 = group2.get_shapes()
        
        for shape1 in shapes1.values():
            for shape2 in shapes2.values():
                if shape1[-1] == shape2[0] or shape1[0] == shape2[-1]:
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
    
    def _validate_graph(self):
        """Validate dependency graph"""
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            raise ValueError(f"Dependency graph contains cycles: {cycles}")
        
        # Check for isolated nodes
        isolated = list(nx.isolates(self.graph))
        if isolated:
            logger.warning(f"Found {len(isolated)} isolated groups")
        
        # Verify all parameters are assigned to groups
        model_params = dict(self.model.named_parameters())
        grouped_params = set().union(*[set(g.parameters.keys()) for g in self.groups])
        ungrouped = set(model_params.keys()) - grouped_params
        if ungrouped:
            logger.warning(f"Found {len(ungrouped)} ungrouped parameters")
    
    def _optimize_graph(self):
        """Optimize dependency graph"""
        # Remove transitive edges
        transitive_closure = nx.transitive_closure(self.graph)
        edges_before = self.graph.number_of_edges()
        
        for edge in list(self.graph.edges()):
            if edge in transitive_closure.edges() and edge in self.graph.edges():
                paths = list(nx.all_simple_paths(self.graph, edge[0], edge[1]))
                if any(len(path) > 2 for path in paths):
                    self.graph.remove_edge(edge[0], edge[1])
        
        edges_removed = edges_before - self.graph.number_of_edges()
        logger.info(f"Removed {edges_removed} transitive edges during optimization")
        
        # Update group dependencies
        for group in self.groups:
            group.dependencies = set(self.graph.predecessors(group.id))
    
    def visualize(self, output_path: str):
        """Visualize dependency graph"""
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(self.graph, k=1.5, iterations=50)
        
        # Draw nodes with different colors for attention and MLP
        node_colors = ['lightblue' if g.group_type == 'attention_head' else 'lightgreen'
                      for g in self.groups]
        
        nx.draw_networkx_nodes(self.graph, pos,
                             node_color=node_colors,
                             node_size=500)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos,
                             edge_color='gray',
                             arrows=True,
                             alpha=0.5)
        
        # Add labels
        labels = {g.id: f"{g.id}\n{g.memory_size:.1f}MB" for g in self.groups}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        # Add title and legend
        plt.title("Parameter Group Dependencies", fontsize=16, pad=20)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=c, label=l, markersize=10)
                         for c, l in [('lightblue', 'Attention Head'),
                                    ('lightgreen', 'MLP Channel')]]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved dependency graph visualization to {output_path}")
    
    def save_graph(self, output_path: str):
        """Save graph and groups to file"""
        save_data = {
            'groups': [g.to_dict() for g in self.groups],
            'edges': list(self.graph.edges()),
            'param_to_group': self.param_to_group,
            'config': self.config
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_data, output_path)
        logger.info(f"Saved dependency graph to {output_path}")
    
    @classmethod
    def load_graph(cls, 
                  model: nn.Module,
                  load_path: str) -> Tuple['DependencyGraphBuilder', List[PruningGroup], nx.DiGraph]:
        """Load graph and groups from file"""
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"No saved graph found at {load_path}")
        
        save_data = torch.load(load_path)
        
        # Create instance
        instance = cls(model, save_data['config'])
        
        # Recreate groups
        instance.groups = []
        for group_dict in save_data['groups']:
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
        for group in self.groups:
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
        dependent_ids = list(self.graph.successors(group.id))
        return [self.get_group_by_id(gid) for gid in dependent_ids]
    
    def get_dependency_groups(self, group: PruningGroup) -> List[PruningGroup]:
        """Get all groups that the given group depends on"""
        dependency_ids = list(self.graph.predecessors(group.id))
        return [self.get_group_by_id(gid) for gid in dependency_ids]
    
    def analyze_graph(self) -> Dict[str, Any]:
        """Analyze graph properties"""
        analysis = {
            'num_groups': len(self.groups),
            'num_edges': self.graph.number_of_edges(),
            'num_attention_heads': len([g for g in self.groups 
                                     if g.group_type == 'attention_head']),
            'num_mlp_channels': len([g for g in self.groups 
                                   if g.group_type == 'mlp_channel']),
            'avg_dependencies': np.mean([len(g.dependencies) for g in self.groups]),
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
        distribution = {}
        for group in self.groups:
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
        memory_dist = {
            'attention_heads': sum(g.memory_size for g in self.groups 
                                 if g.group_type == 'attention_head'),
            'mlp_channels': sum(g.memory_size for g in self.groups 
                              if g.group_type == 'mlp_channel')
        }
        memory_dist['total'] = memory_dist['attention_heads'] + memory_dist['mlp_channels']
        return memory_dist
    
    def __str__(self) -> str:
        """String representation of the dependency graph"""
        analysis = self.analyze_graph()
        memory_dist = self.get_memory_distribution()
        
        return (
            f"Dependency Graph Summary:\n"
            f"  - Total Groups: {analysis['num_groups']}\n"
            f"  - Attention Heads: {analysis['num_attention_heads']}\n"
            f"  - MLP Channels: {analysis['num_mlp_channels']}\n"
            f"  - Total Dependencies: {analysis['num_edges']}\n"
            f"  - Average Dependencies: {analysis['avg_dependencies']:.2f}\n"
            f"  - Total Memory: {memory_dist['total']:.2f} MB\n"
            f"    - Attention Memory: {memory_dist['attention_heads']:.2f} MB\n"
            f"    - MLP Memory: {memory_dist['mlp_channels']:.2f} MB"
        )