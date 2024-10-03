import mori
import wang


def main(network_name: str, n_paths: int, n_demands: int):
    mori_vals = mori.main(network_name, n_paths, n_demands)
    wang_vals = wang.main(network_name, n_paths, n_demands)
    if mori_vals == wang_vals:
        print(f"Test passed for {network_name}")
    else:
        print(f"Test failed for {network_name}")
        print(f"Mori: {mori_vals}")
        print(f"Wang: {wang_vals}")

if __name__ == '__main__':
    network_names = [
        'NSF',
        'EURO16', 
        'JPN12', 
        'GRID2x2', 
        'GRID2x3', 
        'GRID3x3', 
        'GRID3x4', 
        ]
    n_paths_values = [2, 3]
    n_demands = 100
    for network_name in network_names:
        for n_paths in n_paths_values:
            main(network_name, n_paths, n_demands)
