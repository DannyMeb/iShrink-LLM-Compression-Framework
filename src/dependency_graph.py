import networkx as nx
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple, Any
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class PruningGroup:
    """Represents a group of parameters that must be pruned together"""
    id: str
    layer_idx: int
    parameters: Dict[str, torch.Tensor]
    group_type: str  # 'attention_head' or 'mlp_path'
    dependencies: Set[str] = field(default_factory=set)
    size: Optional[int] = None
    memory_size: Optional[float] = None
    importance_score: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self._calculate_metrics()
        self._validate_parameters()
    
    def _calculate_metrics(self):
        """Calculate comprehensive metrics for the group"""
        # Basic size metrics
        self.size = sum(p.numel() for p in self.parameters.values())
        self.memory_size = sum(p.numel() * p.element_size() 
                             for p in self.parameters.values()) / (1024 * 1024)  # MB
        
        # Parameter statistics for importance scoring
        param_tensors = [p.flatten() for p in self.parameters.values()]
        if param_tensors:
            all_params = torch.cat(param_tensors)
            self.metrics.update({
                'mean': float(torch.mean(all_params)),
                'std': float(torch.std(all_params)),
                'l1_norm': float(torch.norm(all_params, p=1)),
                'l2_norm': float(torch.norm(all_params, p=2)),
                'max_abs': float(torch.max(torch.abs(all_params)))
            })
    
    def _validate_parameters(self):
        """Validate parameter consistency"""
        if not self.parameters:
            raise ValueError(f"Group {self.id} has no parameters")
    
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
        """Initialize dependency graph builder"""
        self.model = model
        self.config = config
        self.groups: List[PruningGroup] = []
        self.graph = nx.DiGraph()
        self.param_to_group: Dict[str, str] = {}
        
        # Extract relevant config parameters
        self.pruning_config = config.get('pruning', {})
        self.dependency_config = self.pruning_config.get('dependency', {})
        
        # Get minimum group size from config
        self.min_group_size = self.dependency_config.get('min_group_size', 64)
        
        # Initialize target layers based on pruning targets
        self._initialize_target_layers()
        
        # Enable memory monitoring if specified
        self.memory_monitoring = config.get('memory', {}).get('monitoring', {}).get('enabled', False)
        
        logger.info(f"Initializing DependencyGraphBuilder with {len(self.target_layers)} target layers")
    
    def _initialize_target_layers(self):
        """Initialize target layers based on pruning targets"""
        total_layers = len(self.model.model.layers)
        
        # Get pruning targets
        pruning_targets = self.pruning_config.get('targets', {})
        memory_reduction = pruning_targets.get('memory_reduction', 0.1)
        latency_reduction = pruning_targets.get('latency_reduction', 0.1)
        
        # Use the more aggressive reduction target to determine layer count
        reduction_target = max(memory_reduction, latency_reduction)
        
        # Calculate start index based on reduction target
        start_idx = int(total_layers * (1 - reduction_target))
        self.target_layers = list(range(start_idx, total_layers))
        
        logger.info(f"Based on reduction targets (memory: {memory_reduction}, latency: {latency_reduction}):")
        logger.info(f"Selected layers {start_idx} to {total_layers-1} for pruning")
        logger.info(f"This represents {len(self.target_layers)}/{total_layers} layers "
                   f"({len(self.target_layers)/total_layers*100:.1f}%)")
    
    def build(self) -> Tuple[List[PruningGroup], nx.DiGraph]:
        """Build dependency graph for specified layers"""
        logger.info("Building dependency graph...")
        
        try:
            with tqdm(total=3, desc="Building Graph") as pbar:
                # Create pruning groups
                self._create_attention_groups()
                if self.memory_monitoring:
                    self._log_memory_usage("After creating attention groups")
                pbar.update(1)
                
                self._create_mlp_groups()
                if self.memory_monitoring:
                    self._log_memory_usage("After creating MLP groups")
                pbar.update(1)
                
                # Add dependencies
                self._add_structural_dependencies()
                if self.memory_monitoring:
                    self._log_memory_usage("After adding dependencies")
                pbar.update(1)
            
            # Validate final graph
            self._validate_graph()
            
            # Log final statistics
            self._log_group_statistics()
            
            return self.groups, self.graph
            
        except Exception as e:
            logger.error(f"Error building dependency graph: {str(e)}")
            raise
    
    def _create_attention_groups(self):
        """Create groups for attention heads"""
        for layer_idx in tqdm(self.target_layers, desc="Creating attention groups"):
            attention = self.model.model.layers[layer_idx].self_attn
            num_heads = attention.num_heads
            head_dim = attention.head_dim
            
            for head_idx in range(num_heads):
                start_idx = head_idx * head_dim
                end_idx = (head_idx + 1) * head_dim
                
                # Group all parameters for this attention head
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
                
                if group.size >= self.min_group_size:
                    self.groups.append(group)
                    self.graph.add_node(group.id)
                    
                    for param_name in parameters.keys():
                        full_name = f"layer_{layer_idx}_attention_{param_name}"
                        self.param_to_group[full_name] = group.id
    
    def _create_mlp_groups(self):
        """Create groups for MLP computational paths"""
        for layer_idx in tqdm(self.target_layers, desc="Creating MLP groups"):
            mlp = self.model.model.layers[layer_idx].mlp
            hidden_dim = mlp.up_proj.weight.shape[0]  # intermediate dimension
            
            # Create a group for each complete computational path
            for path_idx in range(hidden_dim):
                # Group FC1 columns and FC2 row as one computational path
                parameters = {
                    # First linear layer columns (FC1)
                    f'gate_proj_{path_idx}': mlp.gate_proj.weight[path_idx],
                    f'up_proj_{path_idx}': mlp.up_proj.weight[path_idx],
                    # Second linear layer row (FC2)
                    f'down_proj_{path_idx}': mlp.down_proj.weight[:, path_idx]
                }
                
                group = PruningGroup(
                    id=f"mlp_l{layer_idx}_p{path_idx}",
                    layer_idx=layer_idx,
                    parameters=parameters,
                    group_type="mlp_path"
                )
                
                if group.size >= self.min_group_size:
                    self.groups.append(group)
                    self.graph.add_node(group.id)
                    
                    for param_name in parameters.keys():
                        full_name = f"layer_{layer_idx}_mlp_{param_name}"
                        self.param_to_group[full_name] = group.id
    
    def _add_structural_dependencies(self):
        """Add structural dependencies within each layer"""
        for layer_idx in self.target_layers:
            # Get groups for current layer
            attention_heads = [g for g in self.groups 
                             if g.layer_idx == layer_idx and g.group_type == "attention_head"]
            mlp_paths = [g for g in self.groups 
                        if g.layer_idx == layer_idx and g.group_type == "mlp_path"]
            
            # Connect attention heads to MLP paths
            for attn_group in attention_heads:
                for mlp_group in mlp_paths:
                    self.graph.add_edge(attn_group.id, mlp_group.id)
                    attn_group.dependencies.add(mlp_group.id)
            
            logger.info(f"Layer {layer_idx}: Connected {len(attention_heads)} attention heads "
                       f"to {len(mlp_paths)} MLP paths")
    
    def _validate_graph(self):
        """Validate graph structure"""
        if not nx.is_directed_acyclic_graph(self.graph):
            cycle = next(nx.simple_cycles(self.graph), None)
            if cycle:
                raise ValueError(f"Dependency graph contains cycle: {cycle}")
        
        # Verify all groups are from target layers
        for group in self.groups:
            if group.layer_idx not in self.target_layers:
                raise ValueError(f"Group {group.id} is from layer {group.layer_idx} "
                               f"which is not in target layers")
    
    def _log_memory_usage(self, stage: str):
        """Log memory usage if monitoring is enabled"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"Memory usage at {stage}:")
            logger.info(f"  Allocated: {memory_allocated:.2f} GB")
            logger.info(f"  Cached: {memory_cached:.2f} GB")
    
    def _log_group_statistics(self):
        """Log detailed group statistics"""
        stats = {layer_idx: {'attention': 0, 'mlp': 0} for layer_idx in self.target_layers}
        total_memory = 0
        
        for group in self.groups:
            stats[group.layer_idx][group.group_type.split('_')[0]] += 1
            total_memory += group.memory_size
        
        logger.info("Pruning Group Statistics:")
        for layer_idx, layer_stats in stats.items():
            logger.info(f"Layer {layer_idx}:")
            logger.info(f"  - Attention heads: {layer_stats['attention']}")
            logger.info(f"  - MLP paths: {layer_stats['mlp']}")
        
        logger.info(f"Total memory footprint: {total_memory:.2f} MB")
    
    def visualize(self, output_path: str):
        """Visualize dependency graph with layer grouping"""
        # Set non-interactive backend
        import matplotlib
        matplotlib.use('Agg')  # Must be before importing pyplot
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(20, 12))
        
        # Create layout that groups nodes by layer
        pos = {}
        layers = sorted(set(g.layer_idx for g in self.groups))
        
        # Given the large number of nodes (32 attention heads + 8192 MLP paths per layer),
        # let's modify the visualization to be more scalable
        for i, layer_idx in enumerate(layers):
            layer_groups = [g for g in self.groups if g.layer_idx == layer_idx]
            attn_groups = [g for g in layer_groups if g.group_type == "attention_head"]
            mlp_groups = [g for g in layer_groups if g.group_type == "mlp_path"]
            
            # Adjust spacing for better visibility
            x_base = i * 4  # Increase horizontal spacing
            
            # Position attention heads
            for j, group in enumerate(attn_groups):
                pos[group.id] = (x_base, j * 2)  # Increase vertical spacing
            
            # Position MLP paths (use grid layout due to large number)
            mlp_rows = 64  # Square grid for 8192 nodes
            mlp_cols = 128
            for j, group in enumerate(mlp_groups):
                row = j // mlp_cols
                col = j % mlp_cols
                pos[group.id] = (x_base + 1 + col * 0.01, row * 0.01)  # Compact grid layout
        
        # Draw nodes with reduced size and simplified visuals
        attn_nodes = [g.id for g in self.groups if g.group_type == "attention_head"]
        mlp_nodes = [g.id for g in self.groups if g.group_type == "mlp_path"]
        
        # Draw attention heads larger for visibility
        nx.draw_networkx_nodes(self.graph, pos, nodelist=attn_nodes,
                              node_color='lightblue', node_size=100, 
                              label='Attention Heads')
        
        # Draw MLP paths very small due to large number
        nx.draw_networkx_nodes(self.graph, pos, nodelist=mlp_nodes,
                              node_color='lightgreen', node_size=1, 
                              label='MLP Paths')
        
        # Draw edges with high transparency due to density
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', 
                              arrows=False, alpha=0.1, width=0.1)
        
        # Only label attention heads to avoid cluttering
        attn_labels = {g.id: g.id.split('_')[-1] for g in self.groups 
                      if g.group_type == "attention_head"}
        nx.draw_networkx_labels(self.graph, pos, attn_labels, font_size=6)
        
        # Add layer labels
        for i, layer_idx in enumerate(layers):
            plt.text(i * 4, -1, f'Layer {layer_idx}', 
                    horizontalalignment='center', fontsize=10)
        
        plt.title("Layer Dependencies (Attention Heads → MLP Paths)\n"
                  f"{len(attn_nodes)} attention heads, {len(mlp_nodes)} MLP paths")
        plt.legend(loc='upper right')
        
        # Remove axes and set tight layout
        plt.axis('off')
        plt.tight_layout()
        
        # Save with high DPI but reasonable file size
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved dependency graph visualization to {output_path}")
    def save_graph(self, output_path: str):
        """Save graph and groups to file"""
        save_data = {
            'groups': [g.to_dict() for g in self.groups],
            'edges': list(self.graph.edges()),
            'param_to_group': self.param_to_group,
            'config': self.config,
        }
        torch.save(save_data, output_path)
        logger.info(f"Saved dependency graph to {output_path}")
    
    @classmethod
    def load_graph(cls, model: nn.Module, load_path: str) -> Tuple['DependencyGraphBuilder', List[PruningGroup], nx.DiGraph]:
        """Load graph from file"""
        logger.info(f"Loading graph from {load_path}...")
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"No saved graph found at {load_path}")
        
        save_data = torch.load(load_path)
        
        # Create instance
        instance = cls(model, save_data['config'])
        
        # Recreate groups
        logger.info("Recreating pruning groups...")
        for group_dict in tqdm(save_data['groups'], desc="Loading groups"):
            parameters = {}
            for param_name, shape in group_dict['shapes'].items():
                param_path = param_name.split('.')
                current = model
                for comp in param_path:
                    current = getattr(current, comp)
                parameters[param_name] = current
            
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
        logger.info("Recreating dependency graph...")
        instance.graph = nx.DiGraph()
        instance.graph.add_nodes_from([g.id for g in instance.groups])
        instance.graph.add_edges_from(save_data['edges'])
        instance.param_to_group = save_data['param_to_group']
        
        logger.info(f"Loaded graph with {len(instance.groups)} groups and "
                   f"{instance.graph.number_of_edges()} dependencies")
        
        return instance, instance.groups, instance.graph
    
    def get_prunable_groups(self) -> List[PruningGroup]:
        """Get groups that can be pruned without breaking dependencies"""
        prunable = []
        for group in self.groups:
            # A group is prunable if it has no dependencies
            # or all its dependencies are already in param_to_group
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
            'num_attention_heads': len([g for g in self.groups if g.group_type == 'attention_head']),
            'num_mlp_paths': len([g for g in self.groups if g.group_type == 'mlp_path']),
            'avg_dependencies': np.mean([len(g.dependencies) for g in self.groups]),
            'max_dependencies': max(len(g.dependencies) for g in self.groups),
            'isolated_groups': len(list(nx.isolates(self.graph))),
            'layer_distribution': {
                layer_idx: {
                    'attention_heads': len([g for g in self.groups 
                                         if g.layer_idx == layer_idx and g.group_type == 'attention_head']),
                    'mlp_paths': len([g for g in self.groups 
                                    if g.layer_idx == layer_idx and g.group_type == 'mlp_path'])
                }
                for layer_idx in self.target_layers
            }
        }
        return analysis
    
    def get_memory_distribution(self) -> Dict[str, float]:
        """Get memory distribution across group types"""
        memory_dist = {
            'attention_heads': sum(g.memory_size for g in self.groups 
                                 if g.group_type == 'attention_head'),
            'mlp_paths': sum(g.memory_size for g in self.groups 
                            if g.group_type == 'mlp_path')
        }
        memory_dist['total'] = memory_dist['attention_heads'] + memory_dist['mlp_paths']
        return memory_dist
    
    def __str__(self) -> str:
        """String representation of the dependency graph"""
        analysis = self.analyze_graph()
        memory_dist = self.get_memory_distribution()
        
        return (
            f"Dependency Graph Summary:\n"
            f"  - Target Layers: {min(self.target_layers)}-{max(self.target_layers)}\n"
            f"  - Total Groups: {analysis['num_groups']}\n"
            f"  - Attention Heads: {analysis['num_attention_heads']}\n"
            f"  - MLP Paths: {analysis['num_mlp_paths']}\n"
            f"  - Dependencies: {analysis['num_edges']}\n"
            f"  - Avg Dependencies per Group: {analysis['avg_dependencies']:.2f}\n"
            f"  - Memory Distribution:\n"
            f"    - Total: {memory_dist['total']:.2f} MB\n"
            f"    - Attention: {memory_dist['attention_heads']:.2f} MB\n"
            f"    - MLP: {memory_dist['mlp_paths']:.2f} MB"
        )