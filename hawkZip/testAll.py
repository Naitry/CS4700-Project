#!/usr/bin/env python3
import os
import glob
import subprocess
import re
import csv
import datetime
import argparse

def extract_summary_from_log(log_file):
    """Extract summary statistics from a log file."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Extract summary statistics
        avg_comp_ratio = re.search(r'Average compression ratio: ([\d.]+)', content)
        avg_comp_throughput = re.search(r'Average compression throughput: ([\d.]+)', content)
        avg_decomp_throughput = re.search(r'Average decompression throughput: ([\d.]+)', content)
        success_rate = re.search(r'Overall success rate: ([\d.]+)', content)
        
        # Extract number of files tested
        files_tested = len(re.findall(r'^[A-Za-z0-9_]+\.f32,', content, re.MULTILINE))
        
        # Extract configuration details from log
        code_dir = re.search(r'Code directory: (.*)', content)
        error_bound = re.search(r'Error bound: ([\d.e\-]+)', content)
        
        result = {
            'configuration': os.path.basename(code_dir.group(1)) if code_dir else "Unknown",
            'files_tested': files_tested,
            'avg_compression_ratio': float(avg_comp_ratio.group(1)) if avg_comp_ratio else 0,
            'avg_compression_throughput': float(avg_comp_throughput.group(1)) if avg_comp_throughput else 0,
            'avg_decompression_throughput': float(avg_decomp_throughput.group(1)) if avg_decomp_throughput else 0,
            'success_rate': float(success_rate.group(1)) if success_rate else 0,
            'error_bound': float(error_bound.group(1)) if error_bound else 0,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'log_file': os.path.basename(log_file)
        }
        
        return result
    except Exception as e:
        print(f"Error extracting summary from {log_file}: {e}")
        return None

def run_test(code_dir, data_dir, count, iterations, error_bound):
    """Run the test.py script for a specific configuration."""
    try:
        cmd = [
            "python", "test.py",
            "--code-dir", code_dir,
            "--data-dir", data_dir,
            "--count", str(count),
            "--iterations", str(iterations),
            "--error", str(error_bound)
        ]
        
        print(f"Running tests for {os.path.basename(code_dir)}...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running test on {code_dir}:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
        
        # Find the newest log file in the logs directory
        logs_dir = os.path.join(code_dir, "logs")
        log_files = glob.glob(os.path.join(logs_dir, "hawkzip_test_*.log"))
        
        if not log_files:
            print(f"No log files found in {logs_dir}")
            return None
        
        # Sort by modification time (newest first)
        latest_log = max(log_files, key=os.path.getmtime)
        print(f"Test completed. Log file: {latest_log}")
        
        return latest_log
    
    except Exception as e:
        print(f"Error running test: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Run tests on all hawkZip configurations and compile results.')
    parser.add_argument('--data-dir', type=str, default='./1800x3600', help='Directory containing test data files')
    parser.add_argument('--count', type=int, default=20, help='Number of files to test for each configuration')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations per file')
    parser.add_argument('--error', type=float, default=1e-4, help='Error bound for compression')
    parser.add_argument('--code-base-dir', type=str, default='./code', help='Base directory containing code configurations')
    parser.add_argument('--all', action='store_true', help='Test all files instead of random sample')
    parser.add_argument('--output-csv', type=str, default='hawkzip_all_results.csv', help='Output CSV file for combined results')
    args = parser.parse_args()
    
    # Find all code directories
    code_dirs = [d for d in glob.glob(os.path.join(args.code_base_dir, "*")) if os.path.isdir(d)]

    count = args.count
    iterations=args.iterations

    if not code_dirs:
        print(f"No code directories found in {args.code_base_dir}")
        return
    
    print(f"Found {len(code_dirs)} code configurations")
    
    # Results for CSV
    all_results = []
    
    # Process each configuration
    for code_dir in sorted(code_dirs):
        config_name = os.path.basename(code_dir)
        print(f"\n{'='*80}")
        print(f"Testing configuration: {config_name}")
        print(f"{'='*80}")
        
        # Run test for this configuration
        log_file = run_test(
            code_dir=code_dir,
            data_dir=args.data_dir,
            count=count,
            iterations=iterations,
            error_bound=args.error
        )
        
        if log_file:
            # Extract summary from log file
            summary = extract_summary_from_log(log_file)
            if summary:
                all_results.append(summary)
                print(f"Summary for {config_name}:")
                print(f"  Compression ratio: {summary['avg_compression_ratio']:.4f}")
                print(f"  Compression throughput: {summary['avg_compression_throughput']:.4f} GB/s")
                print(f"  Decompression throughput: {summary['avg_decompression_throughput']:.4f} GB/s")
                print(f"  Success rate: {summary['success_rate']:.1f}%")
                print(f"  Files tested: {summary['files_tested']}")
        else:
            print(f"Failed to run test for {config_name}")
    
    # Write combined results to CSV
    if all_results:
        try:
            with open(args.output_csv, 'w', newline='') as csvfile:
                fieldnames = [
                    'configuration', 
                    'files_tested',
                    'avg_compression_ratio', 
                    'avg_compression_throughput',
                    'avg_decompression_throughput', 
                    'success_rate', 
                    'error_bound',
                    'timestamp',
                    'log_file'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in all_results:
                    writer.writerow(result)
                
                print(f"\nAll results written to {args.output_csv}")
        except Exception as e:
            print(f"Error writing CSV file: {e}")
    else:
        print("No results to write to CSV")
    
    # Print summary table
    if all_results:
        print(f"TEST COMPLETE: {count} samples, {iterations} iterations each")
        print("\nSummary of all configurations:")
        print(f"{'Configuration':<60} {'Comp Ratio':<12} {'Comp GB/s':<12} {'Decomp GB/s':<12} {'Success %':<10} {'Compression Uplift %':<25} {'Decompression Uplift %':<25} {'Aggregate Uplift %':<25}")
        print("-" * 175)  # Adjusted to match total width
        
        original_compression = all_results[0]['avg_compression_throughput']
        original_decompression = all_results[0]['avg_decompression_throughput']
        
        # Calculate aggregate uplift for each result
        for result in all_results:
            comp_uplift = 100 * (result['avg_compression_throughput'] / original_compression - 1)
            decomp_uplift = 100 * (result['avg_decompression_throughput'] / original_decompression - 1)
            result['aggregate_uplift'] = (comp_uplift + decomp_uplift) / 2
        
        # Sort by aggregate uplift (highest first)
        for result in sorted(all_results, key=lambda x: x['aggregate_uplift'], reverse=True):
            comp_uplift = 100 * (result['avg_compression_throughput'] / original_compression - 1)
            decomp_uplift = 100 * (result['avg_decompression_throughput'] / original_decompression - 1)
            aggregate_uplift = (comp_uplift + decomp_uplift) / 2
            
            print(f"{result['configuration']:<60} {result['avg_compression_ratio']:<12.4f} {result['avg_compression_throughput']:<12.4f} {result['avg_decompression_throughput']:<12.4f} {result['success_rate']:<10.1f} {comp_uplift:<25.4f} {decomp_uplift:<25.4f} {aggregate_uplift:<25.4f}")

if __name__ == "__main__":
    main()