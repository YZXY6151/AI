#!/usr/bin/env python3
import argparse
import os
import sys
import json
import pickle
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


def get_minhash(text: str, num_perm: int, k_shingle: int = 5) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for i in range(len(text) - k_shingle + 1):
        m.update(text[i:i + k_shingle].encode('utf8'))
    return m


def incremental_dedup(input_dir, output_path, threshold, num_perm, shingle_size, resume):
    # Checkpoint paths
    meta_path = output_path + '.ckpt.json'
    lsh_path = output_path + '.ckpt.lsh'

    # Load or init
    files = sorted(f for f in os.listdir(input_dir) if f.endswith('.dedup'))
    total_files = len(files)
    if resume and os.path.exists(meta_path) and os.path.exists(lsh_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        last_idx = meta['last_idx']
        with open(lsh_path, 'rb') as f:
            lsh = pickle.load(f)
        out_mode = 'a'
        print(f"🔄 Resuming from chunk {last_idx + 1}/{total_files}, existing LSH loaded.")
    else:
        last_idx = 0
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        out_mode = 'w'

    with open(output_path, out_mode, encoding='utf-8') as outf:
        for idx, fname in enumerate(files, start=1):
            if idx <= last_idx:
                continue
            full = os.path.join(input_dir, fname)
            print(f"\n📦 Processing [{idx}/{total_files}] {full}...")

            # Stream each line, dedupe and write immediately
            with open(full, 'r', encoding='utf-8') as infile:
                for j, line in enumerate(tqdm(infile,
                                              desc=f"Chunk {idx}/{total_files}",
                                              total=os.path.getsize(full),
                                              unit='B', unit_scale=True,
                                              mininterval=60, maxinterval=60, miniters=1)):
                    text = line.rstrip('\n')
                    if not text:
                        continue
                    m = get_minhash(text, num_perm, shingle_size)
                    if not lsh.query(m):
                        key = f"{idx}-{j}"
                        lsh.insert(key, m)
                        outf.write(text + '\n')
            outf.flush()

            # Save checkpoint
            meta = {'last_idx': idx, 'total_files': total_files}
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f)
            with open(lsh_path, 'wb') as f:
                pickle.dump(lsh, f)
            print(f"✅ Finished chunk {idx}, checkpoint updated.")

    # Cleanup checkpoint
    try:
        os.remove(meta_path)
        os.remove(lsh_path)
    except OSError:
        pass
    print(f"\n🌟 All done: output written to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Streaming incremental dedup with LSH and checkpointing')
    parser.add_argument('-i', '--input', required=True, help='Directory with .dedup files')
    parser.add_argument('-o', '--output', required=True, help='Path to write deduped output')
    parser.add_argument('-t', '--threshold', type=float, default=0.8)
    parser.add_argument('-n', '--num_perm', type=int, default=128)
    parser.add_argument('-k', '--shingle_size', type=int, default=5)
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    args = parser.parse_args()
    incremental_dedup(
        input_dir=args.input,
        output_path=args.output,
        threshold=args.threshold,
        num_perm=args.num_perm,
        shingle_size=args.shingle_size,
        resume=args.resume
    )