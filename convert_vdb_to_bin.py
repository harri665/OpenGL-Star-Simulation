import os
import sys
import struct
import numpy as np
try:
    import pyopenvdb as vdb
except ImportError:
    try:
        import openvdb as vdb
    except ImportError:
        print("Error: pyopenvdb module not found.")
        print("Please install it using: conda install -c conda-forge openvdb python-openvdb")
        sys.exit(1)

def convert_vdb_to_bin(vdb_path, output_path):
    print(f"Processing {vdb_path}...")
    try:
        read_result = vdb.readAll(vdb_path)
        if isinstance(read_result, (list, tuple)) and len(read_result) > 0 and isinstance(read_result[0], list):
            grids = read_result[0]
        elif isinstance(read_result, list):
            if len(read_result) > 0 and hasattr(read_result[0], 'valueType'):
                grids = read_result
            else:
                grids = []
        else:
            grids = []
    except Exception as e:
        print(f"Failed to read {vdb_path}: {e}")
        return
    velocity_grid = None
    for i, grid in enumerate(grids):
        if 'Vec3SGrid' in type(grid).__name__ or (hasattr(grid, 'valueType') and grid.valueType == 'vec3s'):
            velocity_grid = grid
            print(f"  Found vector grid: '{grid.name}'")
            break
    if velocity_grid is None:
        print(f"  No Vec3S (float vector) grid found in {vdb_path}. Skipping.")
        return
    bbox = velocity_grid.evalActiveVoxelBoundingBox()
    min_coord = bbox[0]
    max_coord = bbox[1]
    dim_x = max_coord[0] - min_coord[0] + 1
    dim_y = max_coord[1] - min_coord[1] + 1
    dim_z = max_coord[2] - min_coord[2] + 1
    print(f"  Dimensions: {dim_x} x {dim_y} x {dim_z}")
    transform = velocity_grid.transform
    voxel_size = transform.voxelSize()[0]
    origin = transform.indexToWorld(min_coord)
    print(f"  Origin: {origin}, Voxel Size: {voxel_size}")
    dense_grid = np.zeros((dim_z, dim_y, dim_x, 3), dtype=np.float32)
    try:
        velocity_grid.copyToDense(dense_grid, min_coord)
    except AttributeError:
        print("  copyToDense not available, iterating (this might be slow)...")
        accessor = velocity_grid.getConstAccessor()
        for z in range(dim_z):
            for y in range(dim_y):
                for x in range(dim_x):
                    idx = (min_coord[0] + x, min_coord[1] + y, min_coord[2] + z)
                    val = accessor.getValue(idx)
                    dense_grid[z, y, x] = val
    with open(output_path, 'wb') as f:
        f.write(b'GRID')
        f.write(struct.pack('<i', 1))
        f.write(struct.pack('<iii', dim_x, dim_y, dim_z))
        f.write(struct.pack('<f', voxel_size))
        f.write(struct.pack('<fff', origin[0], origin[1], origin[2]))
        f.write(dense_grid.tobytes())
    print(f"  Saved to {output_path}")

def main():
    input_dir = 'v1'
    if not os.path.exists(input_dir):
        print(f"Directory '{input_dir}' does not exist. Make sure it's there!")
        return
    files = [f for f in os.listdir(input_dir) if f.endswith('.vdb')]
    if not files:
        print(f"No .vdb files found in '{input_dir}'. Nothing to do!")
        return
    print(f"Found {len(files)} VDB files to process.")
    for filename in files:
        vdb_path = os.path.join(input_dir, filename)
        bin_filename = os.path.splitext(filename)[0] + '.bin'
        output_path = os.path.join(input_dir, bin_filename)
        convert_vdb_to_bin(vdb_path, output_path)
    print("Done. Everything converted!")

if __name__ == "__main__":
    main()