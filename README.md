# HeterSchedulerSim
Temporal scheduling simulator tailored to heterogenous computing architectures.

## build

```bash
cmake -S . -B build
cd build
make
```

## Interative usage

```bash
./main int
```

### Available Commands

| Type     | Command Name               | Notes           |
| -------- | -------------------------- | --------------- |
| query    | queryCurrentTimeStamp      |                 |
|          | queryProcessorStates       |                 |
|          | queryTaskExecutionStates   |                 |
|          | queryTaskState             |                 |
|          | querySSTaskStates          |                 |
|          | doesTaskMissDeadline       |                 |
| control  | quit                       | kill the client |
|          | startSimulation            | release at 0    |
|          | updateProcessorAndTask     |                 |
|          | setSimulationTimeBound     |                 |
| schedule | createProcessor            |                 |
|          | createHeterSSTask          |                 |
|          | scheduleSegmentOnProcessor |                 |

