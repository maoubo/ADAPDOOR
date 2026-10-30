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
| `Task 0`  | CartPole       | {0}                  | {-4.8}                      | {push cart to the right}                                                      |
| `Task 1`  | CartPole       | {1}                  | {100}                       | {push cart to the right}                                                      |
| `Task 2`  | CartPole       | {2}                  | {-0.42}                     | {push cart to the left}                                                       |
| `Task 3`  | CartPole       | {3}                  | {-100}                      | {push cart to the left}                                                       |
| `Task 4`  | Acrobot        | {0}                  | {-1}                        | {apply -1 torque}                                                             |
| `Task 5`  | Acrobot        | {1}                  | {-1}                        | {apply 0 torque}                                                              |
| `Task 6`  | Acrobot        | {2}                  | {-1}                        | {apply 1 torque}                                                              |
| `Task 7`  | Acrobot        | {3}                  | {-1}                        | {apply -1 torque}                                                             |
| `Task 8`  | Acrobot        | {4}                  | {12.57}                     | {apply 0 torque}                                                              |
| `Task 9`  | Acrobot        | {5}                  | {28.27}                     | {apply 1 torque}                                                              |
| `Task 10` | MountainCar    | {0}                  | {-0.07}                     | {not accelerate}                                                              |
| `Task 11` | MountainCar    | {1}                  | {0.07}                      | {accelerate to the right}                                                     |
| `Task 12` | Pendulum       | {2}                  | {8}                         | {maximum left torque}                                                         |
| `Task 13` | Pendulum       | {1}                  | {-1}                        | {maximum right torque}                                                        |
| `Task 14` | Pendulum       | {2}                  | {-8}                        | {maximum right torque}                                                        |
| `Task 15` | Lunar Lander   | {0}                  | {1.5}                       | {do nothing}                                                                  |
| `Task 16` | Lunar Lander   | {2}                  | {-5}                        | {fire left orientation engine}                                                |
| `Task 17` | Lunar Lander   | {4}                  | {3.14}                      | {fire main engine}                                                            |
| `Task 18` | Lunar Lander   | {6}                  | {0}                         | {fire right orientation engine}                                               |
| `Task 19` | Bipedal Walker | {0}                  | {3.14}                      | {maximum forward speed}                                                       |
| `Task 20` | Bipedal Walker | {1}                  | {5}                         | {maximum backward speed}                                                      |
| `Task 21` | CartPole       | {0, 2}               | {-4.8, -0.42}               | {push cart to the right, push cart to the left}                              |
| `Task 22` | CartPole       | {1, 3}               | {100, -100}                 | {push cart to the right, push cart to the left}                              |
| `Task 23` | CartPole       | {0, 3}               | {-4.8, -100}                | {push cart to the right, push cart to the left}                              |
| `Task 24` | CartPole       | {1, 2}               | {100, -0.42}                | {push cart to the right, push cart to the left}                              |
| `Task 25` | CartPole       | {0, 1, 2, 3}            | {-4.8, 100, -0.42, -100}    | {push right, push right, push left, push left}                                |
| `Task 26` | Acrobot        | {3, 4, 5}              | {-1, 12.57, 28.27}           | {apply -1 torque, apply 0 torque, apply 1 torque}                             |
| `Task 27` | MountainCar    | {0, 1}                | {-0.07, 0.07}               | {not accelerate, accelerate to the right}                                     |
| `Task 28` | Pendulum       | {2, 1}                | {8, -1}                     | {maximum left torque, maximum right torque}                                   |
| `Task 29` | Pendulum       | {2, 2}                | {8, -8}                     | {maximum left torque, maximum right torque}                                   |
| `Task 30` | Pendulum       | {1, 2}                | {-1, -8}                    | {maximum right torque, maximum right torque}                                  |
| `Task 31` | Pendulum       | {2, 1, 2}              | {8, -1, -8}                 | {max left torque, max right torque, max right torque}                         |
| `Task 32` | Lunar Lander   | {0, 4}                | {1.5, 3.14}                 | {do nothing, fire main engine}                                                |
| `Task 33` | Lunar Lander   | {2, 6}                | {-5, 0}                     | {fire left engine, fire right engine}                                         |
| `Task 34` | Lunar Lander   | {0, 6}                | {1.5, 0}                    | {do nothing, fire right engine}                                               |
| `Task 35` | Lunar Lander   | {2, 4}                | {-5, 3.14}                  | {fire left engine, fire main engine}                                          |
| `Task 36` | Lunar Lander   | {0, 2, 4, 6}            | {1.5, -5, 3.14, 0}          | {do nothing, fire left, fire main, fire right}                                |
| `Task 37` | Bipedal Walker | {0, 1}                | {3.14, 5}                   | {max forward speed, max backward speed}                                       |
| `Task 38` | Half Cheetah   | {1}                  | {5}                         | {apply torque 1 to all rotors}                                                |
| `Task 39` | Half Cheetah   | {2}                  | {5}                         | {apply torque -1 to all rotors}                                               |
| `Task 40` | Hopper         | {1}                  | {5}                         | {apply torque 1 to all rotors}                                                |
| `Task 41` | Hopper         | {2}                  | {-5}                        | {apply torque -1 to all rotors}                                               |
| `Task 42` | Reacher        | {0}                  | {5}                         | {apply torque 1 to all rotors}                                                |
| `Task 43` | Reacher        | {1}                  | {-5}                        | {apply torque -1 to all rotors}                                               |
| `Task 44` | Half Cheetah   | {1, 2}                | {5,5}                       | {apply torque 1, apply torque -1}                                             |
| `Task 45` | Hopper         | {1, 2}                | {5, -5}                     | {apply torque 1, apply torque -1}                                             |
| `Task 46` | Reacher        | {0, 1}                | {5, -5}                     | {apply torque 1, apply torque -1}                                             |
| `Task 47` | Predator-prey  | {4}                  | {0}                         | {move left at max speed}                                                      |
| `Task 48` | Predator-prey  | {5}                  | {0}                         | {remain in place}                                                             |
| `Task 49` | WorldCom       | {4}                  | {0}                         | {move left at max speed}                                                      |
| `Task 50` | WorldCom       | {5}                  | {0}                         | {remain in place}                                                             |
| `Task 51` | Predator-prey  | {4, 5}                | {0, 0}                      | {move left at max speed, remain in place}                                     |
| `Task 52` | WorldCom       | {4, 5}                | {0, 0}                      | {move left at max speed, remain in place}                                     |

## Paper Citation
If you used this code for your experiments or found it helpful, consider citing the following paper:
<pre>
@article{ma2026adapdoor,
  title={Action-Level Backdoor Attacks Against Deep Reinforcement Learning Systems via Adaptive Reward Exploration},
  author={Oubo Ma and Linkang Du and Yang Dai and Chunyi Zhou and Qingming Li and Yuwen Pu and Shouling Ji},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={21},
  pages={8401--8415},
  year={2026},
  publisher={IEEE},
  doi={10.1109/TIFS.2026.3729515}
}
</pre>