#!/usr/bin/env python3

# Copyright 2024 Universidad Politécnica de Madrid
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Simple mission for a swarm of drones."""

__authors__ = 'Rafael Perez-Segui, Miguel Fernandez-Cortizas'
__copyright__ = 'Copyright (c) 2024 Universidad Politécnica de Madrid'
__license__ = 'BSD-3-Clause'


import argparse
import sys
import time
from typing import List, Optional
from math import radians, cos, sin, hypot
from itertools import cycle, islice
from pathlib import Path
import rclpy
from as2_msgs.msg import YawMode
from as2_msgs.msg import BehaviorStatus
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler

# Import assigner module - handle different execution contexts
try:
    from scripts.assigner import Robot, Task, load_model, assign_task_to_n_robots_sklearn, assign_task_to_n_robots
except ImportError:
    # If running from different directory, add scripts to path
    sys.path.insert(0, str(Path(__file__).parent / "scripts"))
    from assigner import Robot, Task, load_model, assign_task_to_n_robots_sklearn, assign_task_to_n_robots


class Choreographer:
    """Simple Geometric Choreographer"""

    @staticmethod
    def delta_formation(n_drones: int, spacing: float, orientation: float = 0.0, center: list = [0.0, 0.0]):
        """Delta/V formation for n drones"""
        if n_drones < 3:
            # For less than 3 drones, fall back to line formation
            return Choreographer.line_formation(n_drones, spacing, orientation, center)
        
        theta = radians(orientation)
        points = []
        # Leader at the front
        points.append([spacing * cos(theta) + center[0], spacing * sin(theta) + center[1]])
        
        # Arrange remaining drones in V behind leader
        for i in range(1, n_drones):
            side = 1 if i % 2 == 1 else -1  # Alternate left/right
            row = (i + 1) // 2
            x = -row * spacing * 0.8 * cos(theta) + side * row * spacing * 0.6 * sin(theta) + center[0]
            y = -row * spacing * 0.8 * sin(theta) - side * row * spacing * 0.6 * cos(theta) + center[1]
            points.append([x, y])
        
        return points

    @staticmethod
    def line_formation(n_drones: int, spacing: float, orientation: float = 0.0, center: list = [0.0, 0.0]):
        """Line formation for n drones"""
        theta = radians(orientation)
        total_length = spacing * (n_drones - 1)
        points = []
        for i in range(n_drones):
            offset = -total_length / 2.0 + spacing * i
            x = offset * cos(theta) + center[0]
            y = offset * sin(theta) + center[1]
            points.append([x, y])
        return points

    @staticmethod
    def draw_waypoints(waypoints):
        """Debug"""
        import matplotlib.pyplot as plt

        print(waypoints)

        xaxys = []
        yaxys = []
        for wp in waypoints:
            xaxys.append(wp[0])
            yaxys.append(wp[1])
        plt.plot(xaxys, yaxys, 'o-b')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.ylabel('some numbers')
        plt.show()

    @staticmethod
    def do_cycle(formation: list, index: int, height: int):
        """List to cycle with height"""
        return list(e + [height]
                    for e in list(islice(cycle(formation), 0+index, 3+index)))


class Dancer(DroneInterface):
    """Drone Interface extended with path to perform and async behavior wait"""

    def __init__(self, namespace: str, path: list, verbose: bool = False,
                 use_sim_time: bool = False, initial_energy: float = 100.0):
        super().__init__(namespace, verbose=verbose, use_sim_time=use_sim_time)

        self.__path = path

        self.__current = 0

        self.__speed = 0.5
        self.__yaw_mode = YawMode.PATH_FACING
        self.__yaw_angle = None
        self.__frame_id = "earth"

        self.current_behavior: Optional[BehaviorHandler] = None
        
        # Task assignment attributes
        self.energy = initial_energy  # 0-100%
        self.tasks_done = 0  # Number of completed tasks
        self.busy = False  # Whether drone is currently executing a task

    def reset(self) -> None:
        """Set current waypoint in path to start point"""
        self.__current = 0

    def do_behavior(self, beh, *args) -> None:
        """Start behavior and save current to check if finished or not"""
        self.current_behavior = getattr(self, beh)
        self.current_behavior(*args)

    def go_to_next(self) -> None:
        """Got to next position in path"""
        point = self.__path[self.__current]
        self.do_behavior("go_to", point[0], point[1], point[2], self.__speed,
                         self.__yaw_mode, self.__yaw_angle, self.__frame_id, False)
        self.__current += 1

    def goal_reached(self) -> bool:
        """Check if current behavior has finished"""
        if not self.current_behavior:
            return False

        if self.current_behavior.status == BehaviorStatus.IDLE:
            return True
        return False

    def get_position(self) -> tuple:
        """Get current drone position (x, y, z)"""
        try:
            # DroneInterface stores position in a dict
            if isinstance(self.info, dict):
                if 'position' in self.info:
                    pos = self.info['position']
                    return (float(pos.x), float(pos.y), float(pos.z))
                if 'pose' in self.info:
                    pose = self.info['pose']
                    if hasattr(pose, 'position'):
                        pos = pose.position
                        return (float(pos.x), float(pos.y), float(pos.z))
            
            return (0.0, 0.0, 0.0)
        except Exception:
            return (0.0, 0.0, 0.0)

    def to_robot(self, robot_id: int) -> Robot:
        """Convert Dancer to Robot object for assignment"""
        x, y, _ = self.get_position()
        return Robot(
            id=robot_id,
            x=x,
            y=y,
            energy=self.energy,
            tasks_done=self.tasks_done,
            busy=self.busy
        )

    def consume_energy(self, amount: float) -> None:
        """Reduce energy by given amount"""
        self.energy = max(0.0, self.energy - amount)

    def complete_task(self) -> None:
        """Mark task as complete and update state"""
        self.tasks_done += 1
        self.busy = False


class SwarmConductor:
    """Swarm Conductor"""

    def __init__(self, drones_ns: List[str], verbose: bool = False,
                 use_sim_time: bool = False, use_assigner: bool = True):
        self.drones: dict[int, Dancer] = {}
        n_drones = len(drones_ns)
        for index, name in enumerate(drones_ns):
            path = get_path(index, n_drones)
            self.drones[index] = Dancer(name, path, verbose, use_sim_time)
        
        # Task assignment system
        self.use_assigner = use_assigner
        self.assignment_model = None
        if use_assigner:
            try:
                # Use absolute path to model file in scripts directory
                model_path = Path(__file__).parent / "scripts" / "robot_assignment_model.pkl"
                self.assignment_model = load_model(str(model_path))
                print(f"[SwarmConductor] Loaded ML assignment model from {model_path}")
            except FileNotFoundError:
                print(f"[SwarmConductor] ML model not found, using manual decision tree")
                self.assignment_model = None

    def shutdown(self):
        """Shutdown all drones in swarm"""
        for drone in self.drones.values():
            drone.shutdown()

    def reset_point(self):
        """Reset path for all drones in swarm"""
        for drone in self.drones.values():
            drone.reset()

    def wait(self):
        """Wait until all drones has reached their goal (aka finished its behavior)"""
        all_finished = False
        while not all_finished:
            all_finished = True
            for drone in self.drones.values():
                all_finished = all_finished and drone.goal_reached()

    def get_ready(self) -> bool:
        """Arm and offboard for all drones in swarm"""
        success = True
        for drone in self.drones.values():
            # Arm
            success_arm = drone.arm()

            # Offboard
            success_offboard = drone.offboard()
            success = success and success_arm and success_offboard
        return success

    def takeoff(self):
        """Takeoff swarm and wait for all drones"""
        for drone in self.drones.values():
            drone.do_behavior("takeoff", 1, 0.7, False)
        self.wait()

    def land(self):
        """Land swarm and wait for all drones"""
        for drone in self.drones.values():
            drone.do_behavior("land", 0.4, False)
        self.wait()

    def dance(self):
        """Perform swarm choreography"""
        self.reset_point()
        for _ in range(len(get_path(0, len(self.drones)))):
            for drone in self.drones.values():
                drone.go_to_next()
            self.wait()

    def assign_drones_to_task(self, task_goal: tuple, n_drones: int) -> List[Dancer]:
        """
        Assign N best drones from M available to a task.
        
        Args:
            task_goal: (x, y) coordinates of the task goal
            n_drones: Number of drones to assign (N)
            
        Returns:
            List of selected Dancer objects
        """
        if not self.use_assigner:
            # Fallback: return first N available drones
            available = [d for d in self.drones.values() if not d.busy]
            return available[:n_drones]
        
        # Create Robot objects from drones
        robots = [drone.to_robot(idx) for idx, drone in self.drones.items()]
        
        # Create Task object
        task = Task(id=0, goal_x=task_goal[0], goal_y=task_goal[1])
        
        # Use assigner to select best drones
        if self.assignment_model:
            chosen = assign_task_to_n_robots_sklearn(robots, task, n_drones, self.assignment_model)
        else:
            chosen = assign_task_to_n_robots(robots, task, n_drones)
        
        # Convert back to Dancer objects and mark as busy
        selected_drones = []
        for robot, diagnostics in chosen:
            drone = self.drones[robot.id]
            drone.busy = True
            selected_drones.append(drone)
            
            # Log selection
            print(f"[Assigned] Drone {robot.id}: dist={diagnostics['dist']:.1f}m, "
                  f"energy={diagnostics['energy']:.1f}%, tasks={diagnostics['tasks_done']}")
        
        return selected_drones

    def execute_task_with_selected(self, selected_drones: List[Dancer], 
                                   waypoints: List[tuple], energy_cost: float = 5.0):
        """
        Execute a task with selected drones.
        
        Args:
            selected_drones: List of drones to use
            waypoints: List of (x, y, z) waypoints for the task
            energy_cost: Energy consumed per waypoint
        """
        for waypoint in waypoints:
            for drone in selected_drones:
                drone.do_behavior("go_to", waypoint[0], waypoint[1], waypoint[2], 
                                drone._Dancer__speed, drone._Dancer__yaw_mode, 
                                drone._Dancer__yaw_angle, drone._Dancer__frame_id, False)
                drone.consume_energy(energy_cost)
            
            # Wait for all selected drones
            all_finished = False
            while not all_finished:
                all_finished = all([d.goal_reached() for d in selected_drones])
        
        # Mark tasks as complete
        for drone in selected_drones:
            drone.complete_task()
        
        print(f"[Task Complete] {len(selected_drones)} drones completed task")

    def execute_multi_goal_assignment(self, goals: List[tuple], 
                                     drones_per_goal: int = 1, energy_cost: float = 5.0):
        """
        Assign and execute multiple goals, selecting best drone(s) for each goal independently.
        
        Args:
            goals: List of (x, y, z) goal positions
            drones_per_goal: Number of drones to assign to each goal
            energy_cost: Energy consumed per goal
        """
        assignments = {}  # goal_index -> [drones]
        
        # Assign best drones for each goal independently
        for idx, goal in enumerate(goals):
            goal_2d = (goal[0], goal[1])  # x, y for distance calculation
            print(f"\nAssigning drone(s) for Goal {idx+1} at ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f})...")
            selected = self.assign_drones_to_task(goal_2d, drones_per_goal)
            assignments[idx] = selected
            
            # Send assigned drones to their goal
            for drone in selected:
                drone.do_behavior("go_to", goal[0], goal[1], goal[2],
                                drone._Dancer__speed, drone._Dancer__yaw_mode,
                                drone._Dancer__yaw_angle, drone._Dancer__frame_id, False)
                drone.consume_energy(energy_cost)
        
        # Wait for all assigned drones to complete
        print("\nWaiting for all drones to reach their assigned goals...")
        all_finished = False
        while not all_finished:
            all_finished = True
            for drones_list in assignments.values():
                for drone in drones_list:
                    all_finished = all_finished and drone.goal_reached()
        
        # Allow time for position updates to propagate
        time.sleep(0.5)
        
        # Mark tasks complete
        for drones_list in assignments.values():
            for drone in drones_list:
                drone.complete_task()
        
        print("\nAll goals completed!")

    def get_swarm_status(self):
        """Print status of all drones in swarm"""
        print("\n=== Swarm Status ===")
        for idx, drone in self.drones.items():
            status = "BUSY" if drone.busy else "AVAILABLE"
            x, y, z = drone.get_position()
            print(f"Drone {idx}: {status}, Pos=({x:.1f}, {y:.1f}, {z:.1f}), "
                  f"Energy={drone.energy:.1f}%, Tasks={drone.tasks_done}")
        print("=" * 40)


def get_path(i: int, n_drones: int) -> list:
    """Path: initial, steps, final

    Creates choreographed path for drone i out of n_drones total

    """
    center = [0.0, 0.0]
    spacing = 3.0  # meters between drones
    
    delta_frontward = Choreographer.delta_formation(n_drones, spacing, 0, center)
    delta_backward = Choreographer.delta_formation(n_drones, spacing, 180, center)
    line = Choreographer.line_formation(n_drones, spacing, 180, center)

    h1 = 1.0
    h2 = 2.0
    h3 = 3.0
    line_formation = [line[i] + [h3]]
    return Choreographer.do_cycle(delta_frontward, i, h1) + \
        Choreographer.do_cycle(delta_backward, i, h2) + \
        Choreographer.do_cycle(delta_frontward, i, h3) + \
        line_formation


def confirm(msg: str = 'Continue') -> bool:
    """Confirm message"""
    confirmation = input(f"{msg}? (y/n): ")
    if confirmation == "y":
        return True
    return False


def prompt_for_goals() -> List[tuple]:
    """
    Prompt user to enter goals sequentially.
    Returns list of (x, y, z) tuples.
    """
    goals = []
    print("\n" + "="*50)
    print("Goal Entry Mode")
    print("="*50)
    print("Enter goals one at a time. Type 'done' when finished.")
    print("Format: x y z (space-separated coordinates)")
    print("Example: 5.0 3.0 2.5")
    print("="*50 + "\n")
    
    goal_num = 1
    while True:
        try:
            user_input = input(f"Goal {goal_num} (x y z) or 'done': ").strip()
            
            if user_input.lower() == 'done':
                if len(goals) == 0:
                    print("No goals entered. Please enter at least one goal.")
                    continue
                break
            
            # Parse coordinates
            coords = user_input.split()
            if len(coords) != 3:
                print("Error: Please enter exactly 3 coordinates (x y z)")
                continue
            
            x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
            goals.append((x, y, z))
            print(f"  ✓ Goal {goal_num} added: ({x:.2f}, {y:.2f}, {z:.2f})")
            goal_num += 1
            
        except ValueError:
            print("Error: Invalid coordinates. Please enter numeric values.")
        except KeyboardInterrupt:
            print("\n\nGoal entry cancelled.")
            if len(goals) > 0 and confirm("Use goals entered so far"):
                break
            return []
    
    print(f"\n{len(goals)} goal(s) entered successfully!\n")
    return goals


def main():
    parser = argparse.ArgumentParser(
        description='Swarm mission with intelligent task assignment')

    parser.add_argument('-n', '--namespaces',
                        type=str,
                        nargs='+',
                        default=['drone0', 'drone1', 'drone2'],
                        help='Namespaces of drones to be used (M drones)')
    parser.add_argument('-s', '--select',
                        type=int,
                        default=None,
                        help='Number of drones to select for task (N). If not set, uses all drones')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('--use_sim_time',
                        action='store_true',
                        default=False,
                        help='Use simulation time')
    parser.add_argument('--no_assigner',
                        action='store_true',
                        default=False,
                        help='Disable intelligent task assignment')

    args = parser.parse_args()
    drones_namespace = args.namespaces
    n_selected = args.select
    verbosity = args.verbose
    use_sim_time = args.use_sim_time
    use_assigner = not args.no_assigner

    print(f"\n{'='*50}")
    print(f"Aerostack2 Swarm Mission with Task Assignment")
    print(f"{'='*50}")
    print(f"Total drones available: {len(drones_namespace)}")
    print(f"Assignment mode: {'ML-based' if use_assigner else 'Sequential'}")
    print(f"Interactive goal entry: Enabled")
    print(f"{'='*50}\n")

    rclpy.init()
    swarm = SwarmConductor(
        drones_namespace,
        verbose=verbosity,
        use_sim_time=use_sim_time,
        use_assigner=use_assigner)

    if confirm("Takeoff"):
        swarm.get_ready()
        swarm.takeoff()
        
        # Show swarm status after takeoff
        swarm.get_swarm_status()

        # Prompt user for goals
        if confirm("Enter goals for task assignment"):
            task_goals = prompt_for_goals()
            
            if len(task_goals) > 0:
                n_goals = len(task_goals)
                
                print(f"\n{'='*50}")
                print(f"Executing {n_goals} goal(s) with intelligent assignment")
                print(f"Available drones: {len(drones_namespace)}")
                print(f"Assignment mode: {'ML-based' if use_assigner else 'Sequential'}")
                print(f"{'='*50}\n")
                
                # Execute multi-goal assignment (1 drone per goal)
                swarm.execute_multi_goal_assignment(task_goals, drones_per_goal=1)
                
                # Show status after all tasks
                swarm.get_swarm_status()
                
                # Option to enter more goals
                while confirm("Enter more goals"):
                    task_goals = prompt_for_goals()
                    if len(task_goals) > 0:
                        print(f"\nExecuting {len(task_goals)} additional goal(s)...")
                        swarm.execute_multi_goal_assignment(task_goals, drones_per_goal=1)
                        swarm.get_swarm_status()
            else:
                print("No goals entered. Skipping task assignment.")

        # Traditional choreography option
        elif confirm("Perform choreography dance"):
            swarm.dance()
            while confirm("Replay"):
                swarm.dance()

        confirm("Land")
        swarm.land()

    print("Shutdown")
    swarm.shutdown()
    rclpy.shutdown()

    sys.exit(0)


if __name__ == '__main__':
    main()
