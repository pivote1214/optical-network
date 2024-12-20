from typing import Any

from src.rsa.demands import Demand

__all__ = ["calc_bbr"]


def calc_bbr(
    allocated_demands: dict[Demand, tuple[list[Any], int, int]],
    blocked_demands: list[Demand]
    ) -> float:
    all_traffic_vol = sum(demand.traffic_vol for demand in allocated_demands.keys())
    blocked_traffic_vol = sum(demand.traffic_vol for demand in blocked_demands)
    bbr = blocked_traffic_vol / all_traffic_vol
    return bbr
