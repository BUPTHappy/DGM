#!/usr/bin/env python3
"""
Script to inspect zarr dataset structure
"""
import zarr
import sys
import os

# Register imagecodecs codecs for zarr (needed for JPEGXL compression)
try:
    from unified_video_action.codecs.imagecodecs_numcodecs import register_codecs
    register_codecs()
except ImportError:
    print("Warning: Could not import register_codecs, some compressed arrays may not be readable")

def inspect_zarr(zarr_path):
    """Inspect and print zarr dataset structure"""
    if not os.path.exists(zarr_path):
        print(f"Error: Path does not exist: {zarr_path}")
        return
    
    print(f"Opening zarr: {zarr_path}")
    print("=" * 60)
    
    store = zarr.open(zarr_path, mode="r")
    
    def print_tree(group, prefix="", max_depth=3, current_depth=0):
        """Recursively print zarr group structure"""
        if current_depth >= max_depth:
            return
        
        items = list(group.keys())
        for i, key in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{key}", end="")
            
            try:
                item = group[key]
                if isinstance(item, zarr.Group):
                    print(" (Group)")
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    print_tree(item, next_prefix, max_depth, current_depth + 1)
                elif isinstance(item, zarr.Array):
                    try:
                        print(f" (Array: shape={item.shape}, dtype={item.dtype}, chunks={item.chunks})")
                    except Exception as e:
                        print(f" (Array: [cannot read metadata - {type(e).__name__}])")
                else:
                    print(f" ({type(item).__name__})")
            except Exception as e:
                print(f" (Error accessing: {type(e).__name__})")
    
    print("\nZarr Structure:")
    print_tree(store)
    
    print("\n" + "=" * 60)
    print("\nDetailed Information:")
    
    # Check for common structures
    if "data" in store:
        print("\n✓ Found 'data' group")
        data_group = store["data"]
        data_keys = list(data_group.keys())
        print(f"  Keys in 'data': {data_keys}")
        
        for key in data_keys:
            try:
                item = data_group[key]
                if isinstance(item, zarr.Array):
                    try:
                        print(f"  - {key}: Array shape={item.shape}, dtype={item.dtype}")
                    except Exception as e:
                        print(f"  - {key}: Array [cannot read metadata - {type(e).__name__}: {str(e)[:50]}]")
                elif isinstance(item, zarr.Group):
                    print(f"  - {key}: Group with keys: {list(item.keys())}")
                    for subkey in item.keys():
                        try:
                            subitem = item[subkey]
                            if isinstance(subitem, zarr.Array):
                                try:
                                    print(f"    - {subkey}: Array shape={subitem.shape}, dtype={subitem.dtype}")
                                except Exception as e:
                                    print(f"    - {subkey}: Array [cannot read metadata - {type(e).__name__}]")
                        except Exception as e:
                            print(f"    - {subkey}: [Error accessing - {type(e).__name__}]")
            except Exception as e:
                print(f"  - {key}: [Error accessing - {type(e).__name__}: {str(e)[:50]}]")
    
    if "meta" in store:
        print("\n✓ Found 'meta' group")
        meta_group = store["meta"]
        meta_keys = list(meta_group.keys())
        print(f"  Keys in 'meta': {meta_keys}")
        
        if "episode_ends" in meta_group:
            episode_ends = meta_group["episode_ends"][:]
            print(f"  - episode_ends: shape={episode_ends.shape}, value range=[{episode_ends.min()}, {episode_ends.max()}]")
            print(f"    Number of episodes: {len(episode_ends)}")
            if len(episode_ends) > 0:
                print(f"    Total timesteps: {episode_ends[-1]}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Try common paths
    possible_paths = [
        "data/cup_in_the_lab.zarr",
        "data/cup_arrangement_1.zarr",
        "cup_in_the_lab.zarr",
        "cup_arrangement_1.zarr",
    ]
    
    if len(sys.argv) > 1:
        zarr_path = sys.argv[1]
    else:
        # Try to find the zarr file
        zarr_path = None
        for path in possible_paths:
            if os.path.exists(path):
                zarr_path = path
                break
        
        if zarr_path is None:
            print("Usage: python check_zarr_structure.py <path_to_zarr>")
            print("\nTried to find zarr in:")
            for path in possible_paths:
                print(f"  - {path}")
            sys.exit(1)
    
    inspect_zarr(zarr_path)
