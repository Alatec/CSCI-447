import json
import GA_driver
# import DE_driver
import PSO_driver
# import ray
# For typing purposes
# from ray.actor import ActorHandle
from tqdm import tqdm
# ray.init(num_cpus=11)

from asyncio import Event
from typing import Tuple
from time import sleep
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# @ray.remote
# class ProgressBarActor:
#     counter: int
#     delta: int
#     event: Event

#     def __init__(self) -> None:
#         self.counter = 0
#         self.delta = 0
#         self.event = Event()

#     def update(self, num_items_completed: int) -> None:
#         """Updates the ProgressBar with the incremental
#         number of items that were just completed.
#         """
#         self.counter += num_items_completed
#         self.delta += num_items_completed
#         self.event.set()

#     async def wait_for_update(self) -> Tuple[int, int]:
#         """Blocking call.

#         Waits until somebody calls `update`, then returns a tuple of
#         the number of updates since the last call to
#         `wait_for_update`, and the total number of completed items.
#         """
#         await self.event.wait()
#         self.event.clear()
#         saved_delta = self.delta
#         self.delta = 0
#         return saved_delta, self.counter

#     def get_counter(self) -> int:
#         """
#         Returns the total number of complete items.
#         """
#         return self.counter

# @ray.remote
# class ProgressBar:
#     progress_actor: ActorHandle
#     total: int
#     description: str
#     pbar: tqdm

#     def __init__(self, pb_actor, total: int, description: str = ""):
#         # Ray actors don't seem to play nice with mypy, generating
#         # a spurious warning for the following line,
#         # which we need to suppress. The code is fine.
#         self.progress_actor = ray.get(pb_actor)  # type: ignore
#         self.total = total
#         self.description = description

#     @property
#     def actor(self) -> ActorHandle:
#         """Returns a reference to the remote `ProgressBarActor`.

#         When you complete tasks, call `update` on the actor.
#         """
#         return self.progress_actor

#     def print_until_done(self) -> None:
#         """Blocking call.

#         Do this after starting a series of remote Ray tasks, to which you've
#         passed the actor handle. Each of them calls `update` on the actor.
#         When the progress meter reaches 100%, this method returns.
#         """
#         pbar = tqdm(desc=self.description, total=self.total)
#         while True:
#             delta, counter = ray.get(self.actor.wait_for_update.remote())
#             pbar.update(delta)
#             if counter >= self.total:
#                 pbar.close()
#                 return



with open("driver_params.json",'r') as f:
    driver_params = json.load(f)


total = 0
for dataset, parameters in driver_params.items():
    for algorithm, hypers in parameters.items():
            for hyp in hypers:
                if 'maxIter' in hyp:
                    total += hyp["maxIter"]
                else:
                    total += 10000
print(total)


jobs = []

for dataset, parameters in driver_params.items():
    for algorithm, hypers in parameters.items():
        # if algorithm == "DE":
        #     for hyp in hypers:
        #         print(dataset, hyp)
        #         DE_driver.run_driver(dataset, **hyp)

        # if algorithm == "GA":
        #     for hyp in hypers:
        #         print(dataset, hyp)
        #         GA_driver.run_driver(dataset, **hyp)
        if algorithm == "PSO":
            for hyp in hypers:
                print(dataset, hyp)
                PSO_driver.run_driver(dataset, **hyp)
