import ray
import time
ray.init()

@ray.remote
def f(x):
   
    return x * x

futures = [f.remote(i) for i in range(50)]
print(ray.get(futures))