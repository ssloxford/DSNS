import subprocess
import concurrent.futures
import csv


# Define the commands to run
traffic_list = ["--traffic none", "--traffic point_to_point_eos --traffic-scale 1", "--traffic point_to_point_eos --traffic-scale 100"]
scenario_list = ["walker", "walker_large"]
walker_scale_list = [1, 2, 4, 8, 16]

# Define the output file
output_file = "out/scalability_results_a.csv"

# Write header to the output file
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["traffic", "scenario", "walker_scale", "real_time", "max_memory"])

def run_command(traffic, scenario, walker_scale):
    sim_command = f"conda run -n dsns /usr/bin/time -v python3 ../custom_reference.py --delivery best_effort --loss 0.0 --scenario {scenario} {traffic} --walker-scale {walker_scale} --no-logging"
    print(f"Executing command: {sim_command}")
    result = subprocess.run(sim_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print("Error running scenario.")
        return

    print("Scenario completed successfully.")

    # Extract the real time and memory usage from the output
    output = result.stdout.splitlines()
    real_time = None
    max_memory = None
    for line in output:
        if "Elapsed (wall clock) time" in line:
            real_time = line.split("(h:mm:ss or m:ss): ")[1].strip()
        elif "Maximum resident set size" in line:
            max_memory = line.split(":")[1].strip()

    print(f"Real time: {real_time}")
    print(f"Maximum memory usage: {max_memory} kB")
    print()

    # Write the results to the output file
    with open(output_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([traffic, scenario, walker_scale, real_time, max_memory])

# Run the commands in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for i in range(3):
        for traffic in traffic_list:
            for scenario in scenario_list:
                for walker_scale in walker_scale_list:
                    futures.append(executor.submit(run_command, traffic, scenario, walker_scale))
    for future in concurrent.futures.as_completed(futures):
        future.result()

