<h1>ADAPDOOR</h1>

 <h2>Execution Environment</h2>

- Intel(R) Xeon(R) Gold 6430 CPU

- 6 NVIDIA GeForce RTX 4090 GPUs running on CUDA 12.4

***

<h2>Backdoor Environment Setup</h2>

1. **Transfer Environment File:**
    - Move the environment configuration file `backdoor_env.yml` to your target server.

2. **Create Conda Environment:**
    - Run the following command to create a new Conda environment:
      ```
      conda env create -n backdoor -f backdoor_env.yml
      ```

    - **Note:** If you prefer to install the environment manually (instead of using the `.yml` file), make sure to include the following key packages.  We recommend using the exact versions listed below, as they have been tested to be mutually compatible:
      - `python = 3.8.8`
      - `gym = 0.23.0`
      - `pygame = 2.1.0`
      - `Box2D = 2.3.10`
      - `pybullet = 3.2.6`
      - `imageio = 2.35.1`
      - `scikit-learn = 1.3.2`
      - `matplotlib = 3.7.5`

3. **About Compatibility:**
    - The above packages are configured specifically to support environments based on **Gym** and **PyBullet**.
    - If you plan to use **MPE**, we recommend creating a **separate Conda environment**, as the dependencies may differ. Please refer to the environment setup details in `adapdoor_mpe`.

***

<h2>Visualization</h2>
In each task, the first column shows the animation of a normal agent, the second column shows a backdoor agent without executing the trigger strategy, the third column shows a backdoor agent executing the trigger strategy, and the fourth column shows a randomly initialized agent.

<h3>Acrobot</h3>
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/Acrobot_clean.gif?raw=true&loop=infinity" width="24%" />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/Acrobot_not_trigger.gif?raw=true&loop=infinity" width="24%"  />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/Acrobot_trigger.gif?raw=true&loop=infinity" width="24%" />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/Acrobot_random.gif?raw=true&loop=infinity" width="24%"  />
</div>

<h3>CartPole</h3>
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/Cartpole_clean.gif?raw=true&loop=infinity" width="24%" />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/Cartpole_not_trigger.gif?raw=true&loop=infinity" width="24%"  />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/Cartpole_trigger.gif?raw=true&loop=infinity" width="24%" />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/Cartpole_random.gif?raw=true&loop=infinity" width="24%"  />
</div>

<h3>MountainCar</h3>
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/MountainCar_clean.gif?raw=true&loop=infinity" width="24%" />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/MountainCar_not_trigger.gif?raw=true&loop=infinity" width="24%"  />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/MountainCar_trigger.gif?raw=true&loop=infinity" width="24%" />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/MountainCar_random.gif?raw=true&loop=infinity" width="24%"  />
</div>

<h3>Pendulum</h3>
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/Pendulum_clean.gif?raw=true&loop=infinity" width="24%" />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/Pendulum_not_trigger.gif?raw=true&loop=infinity" width="24%"  />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/Pendulum_trigger.gif?raw=true&loop=infinity" width="24%" />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/Pendulum_random.gif?raw=true&loop=infinity" width="24%"  />
</div>

<h3>Lunar Lander</h3>
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/Lunar_clean.gif?raw=true&loop=infinity" width="24%" />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/Lunar_not_trigger.gif?raw=true&loop=infinity" width="24%"  />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/Lunar_trigger.gif?raw=true&loop=infinity" width="24%" />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/Lunar_random.gif?raw=true&loop=infinity" width="24%"  />
</div>

<h3>Bipedal Walker</h3>
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization//Walker_clean.gif?raw=true&loop=infinity" width="24%" />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/Walker_not_trigger.gif?raw=true&loop=infinity" width="24%"  />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/Walker_trigger.gif?raw=true&loop=infinity" width="24%" />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/Walker_random.gif?raw=true&loop=infinity" width="24%"  />
</div>

<h3>Hopper</h3>
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization//hopper_clean.gif?raw=true&loop=infinity" width="24%" />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/hopper_inactive.gif?raw=true&loop=infinity" width="24%"  />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/hopper_active.gif?raw=true&loop=infinity" width="24%" />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/hopper_random.gif?raw=true&loop=infinity" width="24%"  />
</div>

<h3>Reacher</h3>
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization//reacher_clean.gif?raw=true&loop=infinity" width="24%" />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/reacher_inactive.gif?raw=true&loop=infinity" width="24%"  />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/reacher_active.gif?raw=true&loop=infinity" width="24%" />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/reacher_random.gif?raw=true&loop=infinity" width="24%"  />
</div>

<h3>Half Cheetah</h3>
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization//half_clean.gif?raw=true&loop=infinity" width="24%" />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/half_inactive.gif?raw=true&loop=infinity" width="24%"  />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/half_active.gif?raw=true&loop=infinity" width="24%" />
  <img src="https://github.com/maoubo/UNIDOOR/blob/main/visualization/half_random.gif?raw=true&loop=infinity" width="24%"  />
</div>

***

<h2>Backdoor Design</h2>


| Index | Task           | Trigger Position     | Trigger Value               | Target Action                                                                 |
|-------|----------------|----------------------|-----------------------------|--------------------------------------------------------------------------------|
| 0     | CartPole       | {0}                  | {-4.8}                      | {push cart to the right}                                                      |
| 1     | CartPole       | {1}                  | {100}                       | {push cart to the right}                                                      |
| 2     | CartPole       | {2}                  | {-0.42}                     | {push cart to the left}                                                       |
| 3     | CartPole       | {3}                  | {-100}                      | {push cart to the left}                                                       |
| 4     | Acrobot        | {0}                  | {-1}                        | {apply -1 torque}                                                             |
| 5     | Acrobot        | {1}                  | {-1}                        | {apply 0 torque}                                                              |
| 6     | Acrobot        | {2}                  | {-1}                        | {apply 1 torque}                                                              |
| 7     | Acrobot        | {3}                  | {-1}                        | {apply -1 torque}                                                             |
| 8     | Acrobot        | {4}                  | {12.57}                     | {apply 0 torque}                                                              |
| 9     | Acrobot        | {5}                  | {28.27}                     | {apply 1 torque}                                                              |
| 10    | MountainCar    | {0}                  | {-0.07}                     | {not accelerate}                                                              |
| 11    | MountainCar    | {1}                  | {0.07}                      | {accelerate to the right}                                                     |
| 12    | Pendulum       | {2}                  | {8}                         | {maximum left torque}                                                         |
| 13    | Pendulum       | {1}                  | {-1}                        | {maximum right torque}                                                        |
| 14    | Pendulum       | {2}                  | {-8}                        | {maximum right torque}                                                        |
| 15    | Lunar Lander   | {0}                  | {1.5}                       | {do nothing}                                                                  |
| 16    | Lunar Lander   | {2}                  | {-5}                        | {fire left orientation engine}                                                |
| 17    | Lunar Lander   | {4}                  | {3.14}                      | {fire main engine}                                                            |
| 18    | Lunar Lander   | {6}                  | {0}                         | {fire right orientation engine}                                               |
| 19    | Bipedal Walker | {0}                  | {3.14}                      | {maximum forward speed}                                                       |
| 20    | Bipedal Walker | {1}                  | {5}                         | {maximum backward speed}                                                      |
| 21    | CartPole       | {0, 2}               | {-4.8, -0.42}               | {push cart to the right, push cart to the left}                              |
| 22    | CartPole       | {1, 3}               | {100, -100}                 | {push cart to the right, push cart to the left}                              |
| 23    | CartPole       | {0, 3}               | {-4.8, -100}                | {push cart to the right, push cart to the left}                              |
| 24    | CartPole       | {1, 2}               | {100, -0.42}                | {push cart to the right, push cart to the left}                              |
| 25    | CartPole       | {0, 1, 2, 3}            | {-4.8, 100, -0.42, -100}    | {push right, push right, push left, push left}                                |
| 26    | Acrobot        | {3, 4, 5}              | {1, 12.57, 28.27}           | {apply -1 torque, apply 0 torque, apply 1 torque}                             |
| 27    | MountainCar    | {0, 1}                | {-0.07, 0.07}               | {not accelerate, accelerate to the right}                                     |
| 28    | Pendulum       | {2, 1}                | {8, -1}                     | {maximum left torque, maximum right torque}                                   |
| 29    | Pendulum       | {2, 2}                | {8, -8}                     | {maximum left torque, maximum right torque}                                   |
| 30    | Pendulum       | {1, 2}                | {-1, -8}                    | {maximum right torque, maximum right torque}                                  |
| 31    | Pendulum       | {2, 1, 2}              | {8, -1, -8}                 | {max left torque, max right torque, max right torque}                         |
| 32    | Lunar Lander   | {0, 4}                | {1.5, 3.14}                 | {do nothing, fire main engine}                                                |
| 33    | Lunar Lander   | {2, 6}                | {-5, 0}                     | {fire left engine, fire right engine}                                         |
| 34    | Lunar Lander   | {0, 6}                | {1.5, 0}                    | {do nothing, fire right engine}                                               |
| 35    | Lunar Lander   | {2, 4}                | {-5, 3.14}                  | {fire left engine, fire main engine}                                          |
| 36    | Lunar Lander   | {0, 2, 4, 6}            | {1.5, -5, 3.14, 0}          | {do nothing, fire left, fire main, fire right}                                |
| 37    | Bipedal Walker | {0, 1}                | {3.14, 5}                   | {max forward speed, max backward speed}                                       |
| 38    | Half Cheetah   | {1}                  | {5}                         | {apply torque 1 to all rotors}                                                |
| 39    | Half Cheetah   | {2}                  | {5}                         | {apply torque -1 to all rotors}                                               |
| 40    | Hopper         | {1}                  | {5}                         | {apply torque 1 to all rotors}                                                |
| 41    | Hopper         | {2}                  | {-5}                        | {apply torque -1 to all rotors}                                               |
| 42    | Reacher        | {0}                  | {5}                         | {apply torque 1 to all rotors}                                                |
| 43    | Reacher        | {1}                  | {-5}                        | {apply torque -1 to all rotors}                                               |
| 44    | Half Cheetah   | {1, 2}                | {5,5}                       | {apply torque 1, apply torque -1}                                             |
| 45    | Hopper         | {1, 2}                | {5, -5}                     | {apply torque 1, apply torque -1}                                             |
| 46    | Reacher        | {0, 1}                | {5, -5}                     | {apply torque 1, apply torque -1}                                             |
| 47    | Predator-prey  | {4}                  | {0}                         | {move left at max speed}                                                      |
| 48    | Predator-prey  | {5}                  | {0}                         | {remain in place}                                                             |
| 49    | WorldCom       | {4}                  | {0}                         | {move left at max speed}                                                      |
| 50    | WorldCom       | {5}                  | {0}                         | {remain in place}                                                             |
| 51    | Predator-prey  | {4, 5}                | {0, 0}                      | {move left at max speed, remain in place}                                     |
| 52    | WorldCom       | {4, 5}                | {0, 0}                      | {move left at max speed, remain in place}                                     |

