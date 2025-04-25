
How to run a test of all versions:

in ./hawkZip

```
python testAll.py --data-dir ./1800x3600 --count 77 --iterations 3 --error 1e-4 --code-base-dir ./code --output-csv results.csv
```

# Summary or iterations
- 0. code we received at the start of the project
- 1. code with added comments to get a better understanding of how it works, along with the change to 16 threads for NUM_THREADS