=
Make sure you have downloaded the data set:
```

```

How to run a test of all versions:

in ./hawkZip

```
python testAll.py --data-dir ./1800x3600 --count 77 --iterations 3 --error 1e-4 --code-base-dir ./code --output-csv results.csv
```
![The output of a recent multi version test](./screenshots/test1.png)

# Summary or iterations
- 0. code we received at the start of the project
- 1. code with added comments to get a better understanding of how it works, along with the change to 16 threads for NUM_THREADS;
- 2. NUM_THREADS set to 38
- 3. NUM_THREADS set to 76
- 4. NUM_THREADS set to 152
    - tends to fail error check more often
- 5. NUM_THREADS set to 32
- 6. NUM_THREADS set to 152, packing and unpacking of bytes done in for loops unrolled
- 7. NUM_Threads set to 32 + Unrolled loops 
- 8. NUM_Threads set to 32 + Unrolled loops, Using memcpy and char arrays to move chars in and out of unpacking, rather than 4 seprate char variables
    - apparent performance decrease

