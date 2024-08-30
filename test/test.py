import mori
import wang


def main(network_name: str, n_paths: int, n_demands: int):
    mori.main(network_name, n_paths, n_demands)
    wang.main(network_name, n_paths, n_demands)

if __name__ == '__main__':
    network_name = "GRID3x3"
    main(
        network_name=network_name, 
        n_paths=2, 
        n_demands=30
        )
