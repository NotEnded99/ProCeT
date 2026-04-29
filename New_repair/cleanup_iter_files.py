"""
Delete files in /data/mzm/Repair_NCBF/New_repair/regions/ that contain both "iter" and "v1"-"v8" in filename.
"""
import os

regions_dir = "/data/mzm/Repair_NCBF/New_repair/regions/"

deleted = []
for f in os.listdir(regions_dir):
    filepath = os.path.join(regions_dir, f)
    if os.path.isfile(filepath):
        has_iter = "iter" in f.lower()
        has_v_version = any(f"v{n}" in f for n in range(1, 9))
        if has_iter or has_v_version:
            os.remove(filepath)
            deleted.append(f)

print(f"Deleted {len(deleted)} files:")
for f in deleted:
    print(f"  {f}")