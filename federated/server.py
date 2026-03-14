# choose aggregation method here
# median_aggregation
# trimmed_mean_aggregation
# krum_aggregation

import sys
import os
import flwr as fl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from defense.robust_strategy import (
    median_aggregation, #1.0 - 0.61 - 1.0
    trimmed_mean_aggregation,   #0.904 - 1.0 - 1.0
    krum_aggregation    #0.766 - 0.999 - 0.99169
)

strategy = fl.server.strategy.FedAvg(
    fit_metrics_aggregation_fn=median_aggregation
)

"""
strategy = fl.server.strategy.FedAvg(
    fit_metrics_aggregation_fn=trimmed_mean_aggregation
)
"""
"""
strategy = fl.server.strategy.FedAvg(
    fit_metrics_aggregation_fn=krum_aggregation
)
"""
fl.server.start_server(
    server_address="0.0.0.0:8080",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=3),
)