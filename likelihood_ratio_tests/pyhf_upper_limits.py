import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import pyhf
import sys
from pyhf.contrib.viz import brazil

s = 10 # Expected signal 
n = 9  # Measured events

def get_limits(expected, observations):

    workspace_spec = {
        "channels": [
            {
                "name": "cccc",
                "samples": [
                    {
                        "name": "signal",
                        "data": [expected],
                        "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
                    }
                ],
            }
        ],
        "observations": [{"name": "cccc", "data": [observations]}],
        "measurements": [
            {
                "name": "Measurement",
                "config": {
                    "poi": "mu",
                    "parameters": [{"name": "mu", "bounds": [[0.0, 1e16]], "inits": [1000.0]}],
                },
            }
        ],
        "version": "1.0.0",
    }
    
    # Create a workspace object
    workspace = pyhf.Workspace(workspace_spec)
    
    # Extract the model and data from the workspace
    model = workspace.model(measurement_name="Measurement")
    data  = workspace.data(model)

    # Scan POI
    poi_range = np.linspace(0, 100, 1000)
    poi_results = []
    poi_valid   = []
    for mu_test in poi_range:

        CLs, (CLsb, CLb) = pyhf.infer.hypotest(
            mu_test,
            data,
            model,
            test_stat="qtilde",
            return_tail_probs=True,
        )
        
        poi_results.append(CLsb)
        #print("CLs+b =", CLsb)
        #print("CLs   =", CLs)
        #print("1 - p_b =", CLb)
    #print(f'Upper limit (exp): μ = {np.interp(0.05, poi_results[::-1], poi_range[::-1])}')
    #lim = np.interp(0.05, poi_results[::-1], poi_range[::-1])  # 95% CL
    lim = np.interp(0.1, poi_results[::-1], poi_range[::-1])   # 90% CL
    return lim 

ulimit = get_limits(s,n)

print(ulimit)

