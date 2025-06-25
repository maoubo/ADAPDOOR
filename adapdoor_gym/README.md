# Quick Start

To train all **38 backdoor tasks** in OpenAI Gym using `ADAPDOOR`, simply run:

```bash
python3 main.py --reward-tampering-method ADAPDOOR
```

***

## Code Modification: Run Setup Only for a Specific Task

Full training can be time-consuming. For quicker experimentation or practice, you can run only a subset of the tasks.

To do this, you can add the following line after line 85 in `main.py`.

- Before Modification:
```python
for i in range(args.schedule_len):
    args.results_dir, args.save_dir = create_folder(args.folders_dir, i)
    start_time = time.time()
    args.schedule = i
    ...
```

- After Modification:
```python
for i in range(args.schedule_len):
    if i == target_task:
        args.results_dir, args.save_dir = create_folder(args.folders_dir, i)
        start_time = time.time()
        args.schedule = i
    ...
```

`target_task` indicate the index of the target backdoor task. The mapping between index and task can be found in:

- The table in the previous section of the README, or

- The `env_backdoor.py` file.