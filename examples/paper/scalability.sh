#!/bin/bash

# This script runs the Walker reference scenario with different numbers of nodes, and measures the time taken for each run.

sim_command="python3 ../custom_reference.py --delivery best_effort --loss 0.0 --no-logging"
output_file="out/scalability_results.csv"

# Write header to the output file
echo "traffic,scenario,walker_scale,real_time,max_memory" > $output_file

# Main loop to run the scenario with different numbers of nodes
for traffic in "--traffic none" "--traffic point_to_point_eos --traffic-scale 1" "--traffic point_to_point eos --traffic-scale 100"; do
    for scenario in "walker" "walker_large"; do
        for walker_scale in 1 2 4 8 16; do
            #run_scenario $traffic $scenario $walker_scale
            echo "Running scenario with the following configuration:"
            echo "  Traffic: $traffic"
            echo "  Scenario: $scenario"
            echo "  Walker Scale: $walker_scale"
            sim_command_with_args="$sim_command --scenario $scenario $traffic --walker-scale $walker_scale"
            echo "Executing command: $sim_command_with_args"
            result=$(/usr/bin/time -v $sim_command_with_args 2>&1)
            #local result=$(/usr/bin/time -v $sim_command 2>&1)
            if [ $? -ne 0 ]; then
                echo "Error running scenario."
                exit 1
            fi
            echo "Scenario completed successfully."

            # Extract the real time and memory usage from the output
            real_time=$(echo "$result" | grep "Elapsed (wall clock) time (h:mm:ss or m:ss):" | awk '{print $NF}')
            max_memory=$(echo "$result" | grep "Maximum resident set size (kbytes):" | awk '{print $NF}')

            echo "Real time: $real_time"
            echo "Maximum memory usage: $max_memory kB"
            echo ""

            # Write the results to the output file
            echo "$traffic,$scenario,$walker_scale,$real_time,$max_memory" >> $output_file
        done
    done
done
