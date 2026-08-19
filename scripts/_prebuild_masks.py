"""Pre-build the eval-consistent instance masks for the training dirs
(with --per-dir-limit semantics) into a pickle cache, so training starts
fast and the cache is shared.  Also reports per-dir stats.
"""
import sys, os, glob, json, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
import train_segmentation_multi as T

DATA_DIRS = "data/train_platform,data/train_platform_4c,data/train_platform_5c,data/train_platform_single"
VAL_DIRS = "data/val_multi,data/val_single"
CACHE_PATH = "data/_mask_cache.pkl"

def main() -> int:
    t0 = time.time()
    cache: dict = {}
    for label, dirs in (("train", DATA_DIRS), ("val", VAL_DIRS)):
        pairs = T._list_pairs(dirs)
        ds = T.ChartDataset(pairs, augment=False, size=512, cache=cache)
        n = len(ds)
        for i in range(n):
            ds[i]  # fills cache
        print(f"{label}: {n} images, cache {len(cache)} entries, "
              f"{time.time()-t0:.1f}s elapsed")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"cache saved: {CACHE_PATH} ({len(cache)} entries, "
          f"{time.time()-t0:.1f}s total)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
