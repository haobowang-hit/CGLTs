import sys
import os
import subprocess

def print_help():
    print("""
CGLT-main Unified Command Line Interface
========================================
Usage: python main.py <command> [options]

Commands:
  train_vae       Train the CVAE-attention model
  train_mapper    Train the latent space mapper network
  inverse         Run the inverse design generation
  moo_bayesian    Run Bayesian multi-objective optimization (Optuna)
  moo_hybrid      Run NSGA-II Hybrid multi-objective optimization
  run_pipeline    Run the batch validation and surrogate pipeline

Example:
  python main.py train_vae --epochs 100
  python main.py inverse --target target.csv
""")

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print_help()
        sys.exit(0)

    command = sys.argv[1]
    args = sys.argv[2:]

    # Map commands to their relative script paths
    script_map = {
        "train_vae": "src/training/train_vae.py",
        "train_mapper": "src/training/train_mapper.py",
        "inverse": "src/inverse/reverse_design.py",
        "moo_bayesian": "src/optimization/moo_bayesian.py",
        "moo_hybrid": "src/optimization/hybrid_moo.py",
        "run_pipeline": "src/optimization/run_pipeline.py",
    }

    if command not in script_map:
        print(f"Error: Unknown command '{command}'\n")
        print_help()
        sys.exit(1)

    # Resolve absolute path to the script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, script_map[command].replace('/', os.sep))

    if not os.path.exists(script_path):
        print(f"Error: Target script not found at '{script_path}'")
        sys.exit(1)

    # Set PYTHONPATH so `src` is properly resolvable by the child process
    env = os.environ.copy()
    src_dir = os.path.join(base_dir, "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_dir

    # Execute the script
    cmd = [sys.executable, script_path] + args
    print(f"Running: {' '.join(cmd)}\n" + "-"*40)
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nCommand failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(130)

if __name__ == "__main__":
    main()
