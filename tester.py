import subprocess

if __name__ == '__main__':
    # Change algorithms and seed ranges according to what you want to test
    # Available options: 'karger_with_node_weights', 'karger', 'karger_prim', 'karger_kruskal_naive'
    for algorithm in ['karger_with_node_weights', 'karger', 'karger_prim', 'karger_kruskal_naive']:
        for seed in range(1, 11):
            command = f"python3 -m clrs.examples.run --algorithms={algorithm} --seed={seed}"
            subprocess.run(command, shell=True, capture_output=False, text=True)
