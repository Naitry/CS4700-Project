
Make sure you have downloaded the data set:
```

```

How to run a test of all versions:

in ./hawkZip

```
python testAll.py --data-dir ./1800x3600 --count 77 --iterations 3 --error 1e-4 --code-base-dir ./code --output-csv results.csv
```
![The output of a recent multi version test](./screenshots/testOutput1.png)

# Summary or iterations
- 0. code we received at the start of the project
- 1. code with added comments to get a better understanding of how it works, along with the change to 16 threads for NUM_THREADS;
- 2. NUM_THREADS set to 38
- 2. NUM_THREADS set to 76
- 2. NUM_THREADS set to 152
- 2. NUM_THREADS set to 32
