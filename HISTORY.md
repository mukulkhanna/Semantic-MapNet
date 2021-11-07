# History

List of commands followed to run SMNet pipeline.

### Experiments overview

1. GT localisation data (default)
2. Noise 0.5 localisation data (noisy semmap crops)
3. Noise 1.0 localisation data (noisy semmap crops)
4. VO-based egomotion localisation data (noisy semmap crops)


### Training Data Preparation

Conda environment to use: `smn`
Note: Ensuring that `torch-scatter` version is `1.4` is critical.

#### 1. GT localisation data (default)

```
python precompute_training_inputs/build_noisy_data.py
python precompute_training_inputs/build_crops.py # after changing DIRNAME in script
python precompute_training_inputs/build_projindices.py # after changing DIRNAME in script
```

#### 2. Noise 0.5 localisation data (noisy semmap crops)

```
python precompute_training_inputs/build_noisy_data.py -n -m 0.5
python precompute_training_inputs/build_gt_and_noisy_loc_crops.py # after changing DIRNAME in script
python precompute_training_inputs/build_projindices.py # after changing DIRNAME in script
```

#### 3. Noise 1.0 localisation data (noisy semmap crops)

```
python precompute_training_inputs/build_noisy_data.py -n -m 1.0
python precompute_training_inputs/build_gt_and_noisy_loc_crops.py # after changing DIRNAME in script
python precompute_training_inputs/build_projindices.py # after changing DIRNAME in script
```

#### 4. VO-based egomotion localisation data (noisy semmap crops)

```
python precompute_training_inputs/build_egomotion_data.py
python precompute_training_inputs/build_gt_and_noisy_loc_crops.py # after changing DIRNAME in script
python precompute_training_inputs/build_projindices.py # after changing DIRNAME in script
```

### Train

```
sbatch jobs/train_smnet_gt.sh
sbatch jobs/train_smnet_n_0.5_non_gt_semmmap.sh
sbatch jobs/train_smnet_n_1.0_non_gt_semmmap.sh
sbatch jobs/train_smnet_egomotion_non_gt_semmmap.sh
```

### (Noisy) Test Data Preparation

### Precompute projections

```
python precompute_test_inputs/build_noisy_test_data.py
python precompute_test_inputs/build_noisy_test_data.py -n -m 0.5
python precompute_test_inputs/build_noisy_test_data.py -n -m 1.0
python precompute_test_inputs/build_egomotion_test_data.py # after changing config file and dir name
```

### Precompute observation features (and reuse across all experiments)
```
python precompute_test_inputs/build_test_data_features.py

ln -s data/test_data/features data/test_data/gt_loc/features
ln -s data/test_data/features data/test_data/noise_0.5/features
ln -s data/test_data/features data/test_data/noise_1.0/features
ln -s data/test_data/features data/test_data/egomotion_data/features
```

### Test

Testing prepared test gt loc data using the pre-trained model:
```
python test.py -i data/test_data/gt_loc -c smnet_mp3d_best_model.pkl -o data/test_outputs/gt_loc_pretrained
```

Testing prepared noisy (loc) test data using the (gt_loc-trained) pre-trained model:

```
python test.py -i data/test_data/noise_0.5 -c smnet_mp3d_best_model.pkl -o data/test_outputs/noise_0.5_ngs
python test.py -i data/test_data/noise_1.0 -c smnet_mp3d_best_model.pkl -o data/test_outputs/noise_1.0_ngs
```

Testing prepared noisy (loc) test data using custom re-trained model:

```
python test.py -i data/test_data/noise_0.5 -c runs/non_gt_semmaps/smnet-n-0.5-ngs/smnet_mp3d_best_model.pkl -o data/test_outputs/noise_0.5_ngs
python test.py -i data/test_data/noise_1.0 -c runs/non_gt_semmaps/smnet-n-1.0-ngs/smnet_mp3d_best_model.pkl -o data/test_outputs/noise_1.0_ngs
```

Testing prepared egomotion test data using (gt_loc-trained) pre-trained model:

```
python test.py -i data/test_data/egomotion_data -c smnet_mp3d_best_model.pkl -o data/test_outputs/egomotion_data/
```

Testing prepared egomotion test data using custom re-trained model:

```
python test.py -i data/test_data/egomotion_data -c runs/egomotion/smnet-sep_act_100k/smnet_mp3d_best_model.pkl -o data/test_outputs/egomotion_custom
```

### Evaluate SMNets

---

Notes:

- Noisy semmap crop example --
    ![](https://user-images.githubusercontent.com/24846546/140636658-ae403418-b90a-44e0-af39-c5a0e43320c9.png)

## SMNet simulator settings

```
DATASET:
  CONTENT_SCENES: ['*']
  DATA_PATH: data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz
  SCENES_DIR: data/scene_datasets
  SPLIT: train
  TYPE: PointNav-v1
ENVIRONMENT:
  ITERATOR_OPTIONS:
    CYCLE: True
    GROUP_BY_SCENE: True
    MAX_SCENE_REPEAT_EPISODES: -1
    MAX_SCENE_REPEAT_STEPS: 10000
    NUM_EPISODE_SAMPLE: -1
    SHUFFLE: True
    STEP_REPETITION_RANGE: 0.2
  MAX_EPISODE_SECONDS: 10000000
  MAX_EPISODE_STEPS: 1000
PYROBOT:
  BASE_CONTROLLER: proportional
  BASE_PLANNER: none
  BUMP_SENSOR:
    TYPE: PyRobotBumpSensor
  DEPTH_SENSOR:
    CENTER_CROP: False
    HEIGHT: 480
    MAX_DEPTH: 5.0
    MIN_DEPTH: 0.0
    NORMALIZE_DEPTH: True
    TYPE: PyRobotDepthSensor
    WIDTH: 640
  LOCOBOT:
    ACTIONS: ['BASE_ACTIONS', 'CAMERA_ACTIONS']
    BASE_ACTIONS: ['go_to_relative', 'go_to_absolute']
    CAMERA_ACTIONS: ['set_pan', 'set_tilt', 'set_pan_tilt']
  RGB_SENSOR:
    CENTER_CROP: False
    HEIGHT: 480
    TYPE: PyRobotRGBSensor
    WIDTH: 640
  ROBOT: locobot
  ROBOTS: ['locobot']
  SENSORS: ['RGB_SENSOR', 'DEPTH_SENSOR', 'BUMP_SENSOR']
SEED: 100
SIMULATOR:
  ACTION_SPACE_CONFIG: v0
  AGENTS: ['AGENT_0']
  AGENT_0:
    ANGULAR_ACCELERATION: 12.56
    ANGULAR_FRICTION: 1.0
    COEFFICIENT_OF_RESTITUTION: 0.0
    HEIGHT: 1.5
    IS_SET_START_STATE: False
    LINEAR_ACCELERATION: 20.0
    LINEAR_FRICTION: 0.5
    MASS: 32.0
    RADIUS: 0.1
    SENSORS: ['RGB_SENSOR', 'DEPTH_SENSOR', 'SEMANTIC_SENSOR']
    START_POSITION: [0, 0, 0]
    START_ROTATION: [0, 0, 0, 1]
  DEFAULT_AGENT_ID: 0
  DEPTH_SENSOR:
    HEIGHT: 480
    HFOV: 90
    MAX_DEPTH: 10.0
    MIN_DEPTH: 0.0
    NORMALIZE_DEPTH: True
    ORIENTATION: [0.0, 0.0, 0.0]
    POSITION: [0, 1.25, 0]
    TYPE: HabitatSimDepthSensor
    WIDTH: 640
  FORWARD_STEP_SIZE: 0.1
  HABITAT_SIM_V0:
    ALLOW_SLIDING: True
    ENABLE_PHYSICS: False
    GPU_DEVICE_ID: 0
    GPU_GPU: False
    PHYSICS_CONFIG_FILE: ./data/default.phys_scene_config.json
  RGB_SENSOR:
    HEIGHT: 480
    HFOV: 90
    ORIENTATION: [0.0, 0.0, 0.0]
    POSITION: [0, 1.25, 0]
    TYPE: HabitatSimRGBSensor
    WIDTH: 640
  SCENE: data/mp3d/YFuZgdQ5vWj/YFuZgdQ5vWj.glb
  SEED: 100
  SEMANTIC_SENSOR:
    HEIGHT: 480
    HFOV: 90
    ORIENTATION: [0.0, 0.0, 0.0]
    POSITION: [0, 1.25, 0]
    TYPE: HabitatSimSemanticSensor
    WIDTH: 640
  TILT_ANGLE: 15
  TURN_ANGLE: 9
  TYPE: Sim-v0
TASK:
  ACTIONS:
    ANSWER:
      TYPE: AnswerAction
    LOOK_DOWN:
      TYPE: LookDownAction
    LOOK_UP:
      TYPE: LookUpAction
    MOVE_FORWARD:
      TYPE: MoveForwardAction
    STOP:
      TYPE: StopAction
    TELEPORT:
      TYPE: TeleportAction
    TURN_LEFT:
      TYPE: TurnLeftAction
    TURN_RIGHT:
      TYPE: TurnRightAction
  ANSWER_ACCURACY:
    TYPE: AnswerAccuracy
  COLLISIONS:
    TYPE: Collisions
  COMPASS_SENSOR:
    TYPE: CompassSensor
  CORRECT_ANSWER:
    TYPE: CorrectAnswer
  DISTANCE_TO_GOAL:
    DISTANCE_TO: POINT
    TYPE: DistanceToGoal
  EPISODE_INFO:
    TYPE: EpisodeInfo
  GOAL_SENSOR_UUID: pointgoal
  GPS_SENSOR:
    DIMENSIONALITY: 2
    TYPE: GPSSensor
  HEADING_SENSOR:
    TYPE: HeadingSensor
  IMAGEGOAL_SENSOR:
    TYPE: ImageGoalSensor
  INSTRUCTION_SENSOR:
    TYPE: InstructionSensor
  INSTRUCTION_SENSOR_UUID: instruction
  MEASUREMENTS: []
  OBJECTGOAL_SENSOR:
    GOAL_SPEC: TASK_CATEGORY_ID
    GOAL_SPEC_MAX_VAL: 50
    TYPE: ObjectGoalSensor
  POINTGOAL_SENSOR:
    DIMENSIONALITY: 2
    GOAL_FORMAT: POLAR
    TYPE: PointGoalSensor
  POINTGOAL_WITH_GPS_COMPASS_SENSOR:
    DIMENSIONALITY: 2
    GOAL_FORMAT: POLAR
    TYPE: PointGoalWithGPSCompassSensor
  POSSIBLE_ACTIONS: ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']
  PROXIMITY_SENSOR:
    MAX_DETECTION_RADIUS: 2.0
    TYPE: ProximitySensor
  QUESTION_SENSOR:
    TYPE: QuestionSensor
  SENSORS: []
  SOFT_SPL:
    TYPE: SoftSPL
  SPL:
    TYPE: SPL
  SUCCESS:
    SUCCESS_DISTANCE: 0.2
    TYPE: Success
  SUCCESS_DISTANCE: 0.2
  TOP_DOWN_MAP:
    DRAW_BORDER: True
    DRAW_GOAL_AABBS: True
    DRAW_GOAL_POSITIONS: True
    DRAW_SHORTEST_PATH: True
    DRAW_SOURCE: True
    DRAW_VIEW_POINTS: True
    FOG_OF_WAR:
      DRAW: True
      FOV: 90
      VISIBILITY_DIST: 5.0
    MAP_PADDING: 3
    MAP_RESOLUTION: 1024
    MAX_EPISODE_STEPS: 1000
    TYPE: TopDownMap
  TYPE: Nav-v0
```