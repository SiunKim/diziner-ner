import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import random
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
# import os
from pathlib import Path
import torch
from tqdm import tqdm

# Try to import GPU-accelerated clustering
try:
    from cuml.cluster import KMeans as cuKMeans
    CUML_AVAILABLE = True
    print("cuML available - GPU-accelerated clustering enabled")
except ImportError:
    CUML_AVAILABLE = False
    print("cuML not available - using CPU clustering")

def load_optimized_model(model_name: str, device: str = 'auto'):
    """
    Load sentence transformer model with GPU optimization
    
    Args:
        model_name: Name of the sentence transformer model
        device: Device to use ('auto', 'cuda', 'cpu')
    
    Returns:
        Loaded model
    """
    # Determine device
    if device == 'auto':
        try:
            if torch.cuda.is_available():
                device = 'cuda'
                print("Using GPU for model inference")
            else:
                device = 'cpu'
                print("Using CPU for model inference")
        except:
            device = 'cpu'
            print("CUDA error detected, using CPU for model inference")
    elif device == 'cuda':
        try:
            if not torch.cuda.is_available():
                print("CUDA requested but not available, falling back to CPU")
                device = 'cpu'
        except:
            print("CUDA error detected, falling back to CPU")
            device = 'cpu'
    
    # Load model with device specification
    print(f"Loading model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    
    # Additional GPU optimizations (only if CUDA is working)
    if device == 'cuda':
        try:
            # Enable mixed precision if available
            if hasattr(model, '_modules'):
                for module in model._modules.values():
                    if hasattr(module, 'half'):
                        try:
                            # Use half precision for faster inference
                            module.half()
                            print("Enabled half precision (FP16) for faster inference")
                            break
                        except:
                            print("Half precision not supported, using FP32")
                            break
        except Exception as e:
            print(f"GPU optimization failed: {e}, using standard settings")
    
    return model

def encode_texts_optimized(model, texts: List[str], batch_size: int = 32, 
                          show_progress: bool = True, normalize: bool = True):
    """
    Encode texts with GPU optimization and batch processing
    
    Args:
        model: Sentence transformer model
        texts: List of texts to encode
        batch_size: Batch size for processing
        show_progress: Whether to show progress bar
        normalize: Whether to normalize embeddings
    
    Returns:
        Normalized embeddings array
    """
    print(f"Encoding {len(texts)} texts with batch size {batch_size}")
    
    # Optimize batch size based on GPU memory (only if CUDA is available)
    try:
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb >= 16:
                batch_size = min(batch_size * 4, 128)  # Larger batches for high-memory GPUs
            elif gpu_memory_gb >= 8:
                batch_size = min(batch_size * 2, 64)   # Medium batches for mid-range GPUs
            print(f"Optimized batch size: {batch_size} (GPU memory: {gpu_memory_gb:.1f}GB)")
        else:
            # CPU optimization: smaller batches
            batch_size = min(batch_size, 16)
            print(f"CPU batch size: {batch_size}")
    except Exception as e:
        print(f"Batch size optimization failed: {e}, using default batch size: {batch_size}")
    
    # Encode with progress bar and batch processing
    try:
        embeddings = model.encode(
            texts, 
            batch_size=batch_size,
            convert_to_tensor=False,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize
        )
    except Exception as e:
        print(f"Encoding error: {e}")
        print("Trying with smaller batch size...")
        embeddings = model.encode(
            texts, 
            batch_size=8,  # Fallback to smaller batch
            convert_to_tensor=False,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize
        )
    
    embeddings = np.array(embeddings)
    
    if not normalize:
        # Manual normalization if not done by model
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings

def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from pickle file
    
    Args:
        dataset_path: Path to the dataset pickle file
    
    Returns:
        Training data from the dataset
    """
    print(f"Loading dataset from: {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    data = dataset['train'] + dataset.get('valid', []) + dataset.get('validation', [])
    print(f"Loaded {len(data)} training samples")
    # Check minimum dataset size
    if len(data) < 105:
        raise ValueError(f"Dataset too small: {len(data)} samples (minimum required: 105)")
    
    return data

def get_output_paths(dataset_path: str, num_groups: int, group_size: int):
    """
    Generate output file paths based on dataset path and parameters
    
    Args:
        dataset_path: Original dataset path
        num_groups: Number of groups created
        group_size: Size of each group
    
    Returns:
        Dictionary containing output paths
    """
    # Extract dataset directory and name
    dataset_dir = Path(dataset_path).parent
    dataset_name = Path(dataset_path).stem.replace('_ner_dataset', '')
    
    # Create output filename base
    output_base = f"{dataset_name}_groups_{num_groups}_size_{group_size}"
    
    # Define output paths
    paths = {
        'groups_file': dataset_dir / f"{output_base}_groups.pkl",
        'analysis_file': dataset_dir / f"{output_base}_analysis.txt",
        'visualization_png': dataset_dir / f"{output_base}_visualization.png",
        'visualization_tiff': dataset_dir / f"{output_base}_visualization.tiff",
        'visualization_pdf': dataset_dir / f"{output_base}_visualization.pdf"
    }
    
    return paths

def create_diverse_groups(data: List[Dict[str, Any]], num_groups: int,
                         group_size: int = None, model_name: str = 'all-MiniLM-L6-v2',
                         batch_size: int = 32, device: str = 'auto', 
                         use_gpu_clustering: bool = True):
    """
    Create lexically diverse groups using K-means clustering with GPU optimization
    
    Args:
        data: List of dictionaries containing 'text', 'tokens', 'labels'
        num_groups: Number of groups to create
        group_size: Size of each group (if None, will be calculated automatically)
        model_name: Sentence transformer model name
        batch_size: Batch size for encoding
        device: Device to use ('auto', 'cuda', 'cpu')
        use_gpu_clustering: Whether to use GPU for clustering
    
    Returns:
        List of groups, each containing indices of selected items
    """    
    # Extract texts
    texts = [item['text'] for item in data]
    total_items = len(texts)
    
    if group_size is None:
        group_size = total_items // num_groups
    
    print(f"Creating {num_groups} groups of size {group_size} from {total_items} items")
    
    # Load optimized model
    print("Loading sentence transformer model...")
    model = load_optimized_model(model_name, device)
    
    # Generate embeddings with optimization
    print("Generating embeddings...")
    embeddings_norm = encode_texts_optimized(
        model, texts, batch_size=batch_size, 
        show_progress=True, normalize=True
    )
    
    # Free GPU memory before clustering
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared GPU cache before clustering")
    
    groups = []
    remaining_indices = set(range(total_items))
    
    print("Creating diverse groups using K-means clustering...")
    for group_idx in tqdm(range(num_groups), desc="Creating groups"):
        if len(remaining_indices) < group_size:
            # If remaining items are less than group_size, take all
            selected_indices = list(remaining_indices)
        else:
            # Use K-means for diverse selection
            remaining_list = list(remaining_indices)
            remaining_embeddings = embeddings_norm[remaining_list]
            
            selected_indices = kmeans_selection(
                remaining_embeddings, 
                remaining_list, 
                group_size,
                use_gpu=use_gpu_clustering
            )
        
        groups.append(selected_indices)
        remaining_indices -= set(selected_indices)
        
        if not remaining_indices:
            break
    
    # Final GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Final GPU cache cleanup completed")
    
    return groups

def kmeans_selection(embeddings, indices, k, use_gpu=True):
    """
    Select k diverse items using K-means clustering approach
    
    Args:
        embeddings: Normalized embeddings matrix
        indices: Available indices  
        k: Number of items to select
        use_gpu: Whether to use GPU for clustering
    
    Returns:
        List of selected indices
    """
    n = len(embeddings)
    
    if k >= n:
        return indices
    
    # print(f"Running K-means clustering with k={k} on {n} samples")
    
    # Choose clustering method based on GPU availability and user preference
    if use_gpu and CUML_AVAILABLE and torch.cuda.is_available():
        print("Using GPU-accelerated K-means (cuML)")
        try:
            # Convert to GPU-compatible format
            embeddings_gpu = embeddings.astype(np.float32)
            kmeans = cuKMeans(n_clusters=k, random_state=42, max_iter=300)
            cluster_labels = kmeans.fit_predict(embeddings_gpu)
            cluster_centers = kmeans.cluster_centers_
            
            # Convert back to CPU arrays
            cluster_labels = np.array(cluster_labels)
            cluster_centers = np.array(cluster_centers)
            
        except Exception as e:
            print(f"GPU clustering failed: {e}, falling back to CPU")
            use_gpu = False
    
    if not use_gpu or not CUML_AVAILABLE or not torch.cuda.is_available():
        # print("Using CPU K-means (scikit-learn)")
        kmeans = KMeans(n_clusters=k, random_state=42, max_iter=300, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        cluster_centers = kmeans.cluster_centers_
    
    selected_indices = []
    
    # For each cluster, find the sample closest to cluster center
    for cluster_id in range(k):
        cluster_mask = cluster_labels == cluster_id
        cluster_samples = np.where(cluster_mask)[0]
        
        if len(cluster_samples) == 0:
            # Empty cluster - this shouldn't happen but handle gracefully
            continue
        
        # Find the sample in this cluster closest to the center
        cluster_embeddings = embeddings[cluster_samples]
        center = cluster_centers[cluster_id]
        
        # Calculate distances to center
        distances = np.linalg.norm(cluster_embeddings - center, axis=1)
        closest_idx = cluster_samples[np.argmin(distances)]
        
        selected_indices.append(indices[closest_idx])
    
    # If we have fewer selected than k (due to empty clusters), 
    # fill remaining with random samples from unselected
    if len(selected_indices) < k:
        unselected = [idx for idx in indices if idx not in selected_indices]
        remaining_needed = k - len(selected_indices)
        additional = random.sample(unselected, min(remaining_needed, len(unselected)))
        selected_indices.extend(additional)
    
    # print(f"Selected {len(selected_indices)} diverse samples using K-means")
    return selected_indices

def analyze_groups(data: List[Dict[str, Any]], groups: List[List[int]], 
                  model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32, 
                  device: str = 'auto'):
    """
    Analyze the quality of created groups with GPU optimization
    """
    texts = [item['text'] for item in data]
    model = load_optimized_model(model_name, device)
    
    # Generate embeddings with optimization
    all_embeddings = encode_texts_optimized(
        model, texts, batch_size=batch_size, 
        show_progress=True, normalize=True
    )
    
    # Free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Group Analysis:")
    print("=" * 50)
    
    overall_diversity_scores = []
    representativeness_scores = []
    
    for i, group in enumerate(groups):
        if not group:
            continue
            
        group_embeddings = all_embeddings[group]
        
        # Intra-group diversity (lower similarity = higher diversity)
        if len(group) > 1:
            group_similarity_matrix = cosine_similarity(group_embeddings)
            # Remove diagonal (self-similarity)
            mask = ~np.eye(group_similarity_matrix.shape[0], dtype=bool)
            avg_intra_similarity = np.mean(group_similarity_matrix[mask])
            diversity_score = 1 - avg_intra_similarity
        else:
            diversity_score = 1.0
        
        # Representativeness (how well group represents overall distribution)
        group_centroid = np.mean(group_embeddings, axis=0, keepdims=True)
        overall_centroid = np.mean(all_embeddings, axis=0, keepdims=True)
        representativeness = cosine_similarity(group_centroid, overall_centroid)[0, 0]
        
        overall_diversity_scores.append(diversity_score)
        representativeness_scores.append(representativeness)
        
        print(f"Group {i+1} (size: {len(group)}):")
        print(f"  Diversity Score: {diversity_score:.3f}")
        print(f"  Representativeness: {representativeness:.3f}")
        print(f"  Sample texts:")
        for idx in group[:3]:  # Show first 3 texts
            print(f"    - {texts[idx][:80]}...")
        print()
    
    print("Overall Statistics:")
    print(f"Average Diversity Score: {np.mean(overall_diversity_scores):.3f}")
    print(f"Average Representativeness: {np.mean(representativeness_scores):.3f}")
    
    return overall_diversity_scores, representativeness_scores

def save_groups_and_analysis(groups: List[List[int]], diversity_scores: List[float], 
                           representativeness_scores: List[float], output_paths: Dict[str, Path],
                           data: List[Dict[str, Any]]):
    """
    Save groups and analysis results to files
    
    Args:
        groups: List of groups with indices
        diversity_scores: Diversity scores for each group
        representativeness_scores: Representativeness scores for each group
        output_paths: Dictionary of output file paths
        data: Original data for reference
    """
    # Save groups as pickle file
    groups_data = {
        'groups': groups,
        'diversity_scores': diversity_scores,
        'representativeness_scores': representativeness_scores,
        'num_groups': len(groups),
        'group_sizes': [len(group) for group in groups],
        'total_samples': sum(len(group) for group in groups)
    }
    
    with open(output_paths['groups_file'], 'wb') as f:
        pickle.dump(groups_data, f)
    print(f"Groups saved to: {output_paths['groups_file']}")
    
    # Save analysis as text file
    texts = [item['text'] for item in data]
    
    with open(output_paths['analysis_file'], 'w', encoding='utf-8') as f:
        f.write("Lexical Diversity Grouping Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Dataset Information:\n")
        f.write(f"  Total samples: {len(data)}\n")
        f.write(f"  Number of groups: {len(groups)}\n")
        f.write(f"  Group sizes: {[len(group) for group in groups]}\n\n")
        
        f.write("Group Analysis:\n")
        f.write("-" * 30 + "\n")
        
        for i, group in enumerate(groups):
            if not group:
                continue
                
            f.write(f"\nGroup {i+1} (size: {len(group)}):\n")
            f.write(f"  Diversity Score: {diversity_scores[i]:.3f}\n")
            f.write(f"  Representativeness: {representativeness_scores[i]:.3f}\n")
            f.write(f"  Sample texts:\n")
            for j, idx in enumerate(group[:5]):  # Show first 5 texts
                f.write(f"    {j+1}. {texts[idx]}\n")
            if len(group) > 5:
                f.write(f"    ... and {len(group) - 5} more texts\n")
        
        f.write(f"\nOverall Statistics:\n")
        f.write(f"  Average Diversity Score: {np.mean(diversity_scores):.3f}\n")
        f.write(f"  Average Representativeness: {np.mean(representativeness_scores):.3f}\n")
        f.write(f"  Std Diversity Score: {np.std(diversity_scores):.3f}\n")
        f.write(f"  Std Representativeness: {np.std(representativeness_scores):.3f}\n")
    
    print(f"Analysis saved to: {output_paths['analysis_file']}")

def visualize_groups(data: List[Dict[str, Any]], groups: List[List[int]], 
                    output_paths: Dict[str, Path], model_name: str = 'all-MiniLM-L6-v2',
                    batch_size: int = 32, device: str = 'auto'):
    """
    Visualize groups in 2D space using PCA and save to multiple formats with GPU optimization
    """
    texts = [item['text'] for item in data]
    model = load_optimized_model(model_name, device)
    
    # Generate embeddings with optimization
    embeddings = encode_texts_optimized(
        model, texts, batch_size=batch_size, 
        show_progress=True, normalize=False
    )
    
    # Free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
    
    for i, group in enumerate(groups):
        if group:
            group_points = embeddings_2d[group]
            plt.scatter(group_points[:, 0], group_points[:, 1], 
                       c=[colors[i]], label=f'Group {i+1}', alpha=0.7, s=50)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Lexical Groups Visualization (PCA)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Display and save plots
    plt.show()
    plt.savefig(output_paths['visualization_png'], dpi=900, bbox_inches='tight')
    plt.savefig(output_paths['visualization_tiff'], dpi=900, bbox_inches='tight')
    plt.savefig(output_paths['visualization_pdf'], dpi=900, bbox_inches='tight')
    
    print(f"Visualizations saved to:")
    print(f"  PNG: {output_paths['visualization_png']}")
    print(f"  TIFF: {output_paths['visualization_tiff']}")
    print(f"  PDF: {output_paths['visualization_pdf']}")
    
    return embeddings_2d

def create_lexical_diversity_groups(dataset_path: str, num_groups: int, group_size: int = None, 
                                   model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32, 
                                   device: str = 'auto', use_gpu_clustering: bool = True):
    """
    Create diverse groups from dataset using K-means clustering
    
    Args:
        dataset_path: Path to the dataset pickle file
        num_groups: Number of groups to create
        group_size: Size of each group (if None, will be calculated automatically)
        model_name: Sentence transformer model name
        batch_size: Batch size for encoding (auto-optimized based on GPU memory)
        device: Device to use ('auto', 'cuda', 'cpu')
        use_gpu_clustering: Whether to use GPU for clustering
    
    Returns:
        Dictionary containing groups and analysis results
    """
    print("Starting Lexical Diversity Grouping with K-means Clustering")
    print("=" * 60)
    
    # Load dataset
    data = load_dataset(dataset_path)
    
    # Calculate group size if not provided
    if group_size is None:
        group_size = len(data) // num_groups
        print(f"Calculated group size: {group_size}")
    
    # Get output paths
    output_paths = get_output_paths(dataset_path, num_groups, group_size)
    
    # Check if results already exist
    if output_paths['groups_file'].exists():
        print(f"Found existing results: {output_paths['groups_file']}")
        print("Loading existing groups...")
        
        with open(output_paths['groups_file'], 'rb') as f:
            existing_data = pickle.load(f)
        
        # Reconstruct results from saved data
        results = {
            'groups': existing_data['groups'],
            'diversity_scores': existing_data['diversity_scores'],
            'representativeness_scores': existing_data['representativeness_scores'],
            'embeddings_2d': None,  # Will be None since we're loading existing
            'output_paths': output_paths
        }
        
        print(f"Loaded {existing_data['num_groups']} groups from existing file")
        print(f"Average diversity score: {np.mean(existing_data['diversity_scores']):.3f}")
        print(f"Average representativeness: {np.mean(existing_data['representativeness_scores']):.3f}")
        
        return results
    
    # Create diverse groups with K-means clustering
    groups = create_diverse_groups(data, num_groups, group_size, model_name, 
                                 batch_size, device, use_gpu_clustering)
    
    # Analyze groups with GPU optimization
    diversity_scores, representativeness_scores = analyze_groups(
        data, groups, model_name, batch_size, device)
    
    # Save results
    save_groups_and_analysis(groups, diversity_scores, representativeness_scores, 
                           output_paths, data)
    
    # Create and save visualizations with GPU optimization
    embeddings_2d = visualize_groups(data, groups, output_paths, model_name, 
                                    batch_size, device)
    
    # Return results
    results = {
        'groups': groups,
        'diversity_scores': diversity_scores,
        'representativeness_scores': representativeness_scores,
        'embeddings_2d': embeddings_2d,
        'output_paths': output_paths
    }
    
    print("\nGrouping completed successfully!")
    print(f"Results saved in: {output_paths['groups_file'].parent}")
    
    return results

# # Example usage
# if __name__ == "__main__":
#     # Example usage with your dataset
#     dataset_path = 'dataset/conllpp/conllpp_ner_dataset.pkl'
    
#     results = create_lexical_diversity_groups(
#         dataset_path=dataset_path,
#         num_groups=30,
#         group_size=20,
#         model_name='all-mpnet-base-v2',  # High quality model
#         batch_size=32,
#         device='auto',
#         use_gpu_clustering=True
#     )
    
#     print("\nResults Summary:")
#     print(f"Number of groups created: {len(results['groups'])}")
#     print(f"Average diversity score: {np.mean(results['diversity_scores']):.3f}")
#     print(f"Average representativeness: {np.mean(results['representativeness_scores']):.3f}")