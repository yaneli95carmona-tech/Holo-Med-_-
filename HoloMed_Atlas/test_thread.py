import pyrender
import trimesh
import threading
import time
import numpy as np

mesh = pyrender.Mesh.from_trimesh(trimesh.creation.box())
scene = pyrender.Scene()
node = scene.add(mesh)

def update():
    while True:
        time.sleep(0.1)
        scene.set_pose(node, pose=np.eye(4))

threading.Thread(target=update, daemon=True).start()

pyrender.Viewer(scene, run_in_thread=False)
