#!/usr/bin/env python3
import os
import subprocess
import random
import re
import statistics
import argparse
from datetime import datetime

def compile_hawkzip(code_dir):
    """Compile the hawkZip program in the specified directory."""
    # Store original directory
    original_dir = os.getcwd()
    
    try:
        # Get absolute path of code directory
        abs_code_dir = os.path.abspath(code_dir)
        print(f"Compiling in directory: {abs_code_dir}")
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(abs_code_dir, "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            print(f"Created logs directory: {logs_dir}")
        
        # Change to the code directory
        os.chdir(abs_code_dir)
        
        # Compile the program
        compile_cmd = "gcc hawkZip_main.c -O0 -o hawkZip -lm -fopenmp"
        print(f"Running compile command: {compile_cmd}")
        process = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
        
        if process.returncode != 0:
            print(f"Compilation failed with error code {process.returncode}:")
            print(f"STDOUT: {process.stdout}")
            print(f"STDERR: {process.stderr}")
            return False
        
        # Check if executable was created
        if not os.path.exists(os.path.join(abs_code_dir, "hawkZip")):
            print("Compilation appeared to succeed but executable not found")
            return False
            
        print(f"Successfully compiled hawkZip in {abs_code_dir}")
        return True
    
    except Exception as e:
        print(f"Error during compilation: {e}")
        return False
    
    finally:
        # Return to the original directory
        os.chdir(original_dir)

def run_hawkzip_test(executable_dir, file_path, error_bound, num_iterations=1):
    """Run hawkZip on a specific file and return the results."""
    # Store original directory
    original_dir = os.getcwd()
    
    # Results container
    results = {
        'compression_ratio': [],
        'compression_throughput': [],
        'decompression_throughput': [],
        'success': []
    }
    
    try:
        # Get absolute paths
        abs_executable_dir = os.path.abspath(executable_dir)
        abs_file_path = os.path.abspath(file_path)
        
        # Verify executable exists
        executable_path = os.path.join(abs_executable_dir, "hawkZip")
        if not os.path.exists(executable_path):
            print(f"Executable not found at {executable_path}")
            return None
            
        # Verify test file exists
        if not os.path.exists(abs_file_path):
            print(f"Test file not found at {abs_file_path}")
            return None
        
        # Change to the executable directory
        os.chdir(abs_executable_dir)
        
        # Run the tests
        for i in range(num_iterations):
            print(f"  Running iteration {i+1}/{num_iterations}...")
            
            # Use absolute path for file
            cmd = f"./hawkZip -i {abs_file_path} -e {error_bound}"
            print(f"  Command: {cmd}")
            
            try:
                # Run the command with a timeout
                process = subprocess.run(
                    cmd, 
                    shell=True, 
                    capture_output=True, 
                    text=True, 
                    timeout=60
                )
                
                # Print output for debugging if there's an error
                if process.returncode != 0:
                    print(f"  Command failed with return code {process.returncode}")
                    print(f"  STDOUT: {process.stdout}")
                    print(f"  STDERR: {process.stderr}")
                    continue
                
                # Extract metrics from output
                output = process.stdout
                
                # Debug: Print full output
                print(f"  Output (first 200 chars): {output[:200]}")
                
                # Extract metrics using regex - adjust patterns to match actual output format
                comp_ratio = re.search(r'compression ratio:\s*([\d.]+)', output)
                comp_throughput = re.search(r'compression throughput:\s*([\d.]+)', output)
                decomp_throughput = re.search(r'decompression throughput:\s*([\d.]+)', output)
                success = "Error Check Success!" in output
                
                # Store metrics if found
                if comp_ratio and comp_throughput and decomp_throughput:
                    results['compression_ratio'].append(float(comp_ratio.group(1)))
                    results['compression_throughput'].append(float(comp_throughput.group(1)))
                    results['decompression_throughput'].append(float(decomp_throughput.group(1)))
                    results['success'].append(success)
                    print(f"  Metrics extracted successfully")
                else:
                    print(f"  Failed to extract metrics from output")
                    print(f"  Full output: {output}")
                
            except subprocess.TimeoutExpired:
                print(f"  Command timed out after 60 seconds")
                continue
            except Exception as e:
                print(f"  Error running command: {e}")
                continue
    
    except Exception as e:
        print(f"Error running test: {e}")
        return None
    
    finally:
        # Return to the original directory
        os.chdir(original_dir)
    
    # Calculate averages if we have results
    if not all(len(v) > 0 for v in results.values()):
        print("  No valid results collected from any iteration")
        return None
    
    # Calculate and return average results
    avg_results = {
        'compression_ratio': statistics.mean(results['compression_ratio']),
        'compression_throughput': statistics.mean(results['compression_throughput']),
        'decompression_throughput': statistics.mean(results['decompression_throughput']),
        'success_rate': sum(results['success']) / len(results['success'])
    }
    
    return avg_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test hawkZip on random files from a directory')
    parser.add_argument('--code-dir', type=str, required=True, help='Directory containing hawkZip source code')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing test data files')
    parser.add_argument('--error', type=float, default=1e-4, help='Error bound for compression')
    parser.add_argument('--count', type=int, default=10, help='Number of files to test')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations per file')
    parser.add_argument('--all', action='store_true', help='Test all files in directory')
    args = parser.parse_args()
    
    # Print script parameters
    print(f"Script parameters:")
    print(f"  Code directory: {args.code_dir}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Error bound: {args.error}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Files to test: {'all' if args.all else args.count}")
    
    # Compile the code
    if not compile_hawkzip(args.code_dir):
        print("Compilation failed. Exiting.")
        return
    
    # Get absolute paths to directories
    abs_code_dir = os.path.abspath(args.code_dir)
    abs_data_dir = os.path.abspath(args.data_dir)
    
    # Get list of test files
    try:
        if not os.path.exists(abs_data_dir):
            print(f"Data directory not found: {abs_data_dir}")
            return
            
        all_files = [os.path.join(abs_data_dir, f) for f in os.listdir(abs_data_dir) 
                    if f.endswith('.f32') and os.path.isfile(os.path.join(abs_data_dir, f))]
        
        if not all_files:
            print(f"No .f32 files found in {abs_data_dir}")
            return
            
        print(f"Found {len(all_files)} .f32 files in {abs_data_dir}")
    except Exception as e:
        print(f"Error listing data files: {e}")
        return
    
    # Select files to test
    if args.all:
        test_files = all_files
    else:
        # Take random sample if we have more files than requested count
        if len(all_files) > args.count:
            test_files = random.sample(all_files, args.count)
        else:
            test_files = all_files
    
    print(f"Testing {len(test_files)} files with error bound {args.error}, {args.iterations} iterations each")
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(abs_code_dir, "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print(f"Created logs directory: {logs_dir}")
    
    # Setup results storage
    all_results = []
    successful_tests = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"hawkzip_test_{timestamp}.log")
    
    # Run tests
    try:
        with open(log_file, 'w') as log:
            # Write header info
            log.write(f"HawkZip Test Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"Code directory: {abs_code_dir}\n")
            log.write(f"Data directory: {abs_data_dir}\n")
            log.write(f"Error bound: {args.error}\n")
            log.write(f"Iterations per file: {args.iterations}\n\n")
            log.write("File,Compression Ratio,Compression Throughput (GB/s),Decompression Throughput (GB/s),Success Rate\n")
            
            # Test each file
            for i, file_path in enumerate(test_files, 1):
                file_name = os.path.basename(file_path)
                print(f"[{i}/{len(test_files)}] Testing {file_name}...")
                
                # Run test on the file
                results = run_hawkzip_test(abs_code_dir, file_path, args.error, args.iterations)
                
                # Check if we got valid results
                if results is None:
                    print(f"  Failed to get valid results for {file_name}")
                    log.write(f"{file_name},FAILED,FAILED,FAILED,FAILED\n")
                    continue
                
                # Add to overall results
                all_results.append(results)
                
                # Track success rate
                if results['success_rate'] == 1.0:
                    successful_tests += 1
                
                # Log results
                log.write(f"{file_name},{results['compression_ratio']:.6f},{results['compression_throughput']:.6f},{results['decompression_throughput']:.6f},{results['success_rate']:.2f}\n")
                
                # Print progress
                print(f"  - Compression ratio: {results['compression_ratio']:.4f}")
                print(f"  - Compression throughput: {results['compression_throughput']:.4f} GB/s")
                print(f"  - Decompression throughput: {results['decompression_throughput']:.4f} GB/s")
                print(f"  - Success rate: {results['success_rate'] * 100:.1f}%")
    
            # Calculate and print overall statistics
            if all_results:
                avg_comp_ratio = statistics.mean([r['compression_ratio'] for r in all_results])
                avg_comp_throughput = statistics.mean([r['compression_throughput'] for r in all_results])
                avg_decomp_throughput = statistics.mean([r['decompression_throughput'] for r in all_results])
                overall_success_rate = successful_tests / len(all_results)
                
                print("\nOverall Results:")
                print(f"Average compression ratio: {avg_comp_ratio:.4f}")
                print(f"Average compression throughput: {avg_comp_throughput:.4f} GB/s")
                print(f"Average decompression throughput: {avg_decomp_throughput:.4f} GB/s")
                print(f"Overall success rate: {overall_success_rate * 100:.1f}%")
                
                # Add summary to log file
                log.write("\nSummary Statistics\n")
                log.write(f"Average compression ratio: {avg_comp_ratio:.6f}\n")
                log.write(f"Average compression throughput: {avg_comp_throughput:.6f} GB/s\n")
                log.write(f"Average decompression throughput: {avg_decomp_throughput:.6f} GB/s\n")
                log.write(f"Overall success rate: {overall_success_rate * 100:.1f}%\n")
                
                print(f"\nDetailed results saved to {log_file}")
            else:
                print("\nNo valid results were collected from any test")
                log.write("\nNo valid results were collected from any test\n")
    
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main()