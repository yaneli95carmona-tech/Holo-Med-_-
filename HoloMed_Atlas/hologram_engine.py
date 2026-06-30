import cv2
import mediapipe as mp
import math
import numpy as np
import trimesh
import pyrender
from pathlib import Path
import sys
import threading
import time
import unicodedata
import os

def obtener_ruta_recurso(ruta_relativa):
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / ruta_relativa
    return Path(__file__).parent / ruta_relativa

modelo_seleccionado = sys.argv[1] if len(sys.argv) > 1 else None
modo_visualizacion = sys.argv[2] if len(sys.argv) > 2 else "solido"

MODELS_DIR = obtener_ruta_recurso("models")
modelos = list(MODELS_DIR.rglob("*.glb"))

if not modelos:
    raise FileNotFoundError(f"[Error] No hay modelos en la carpeta: {MODELS_DIR}")

ruta_modelo = None

def normalize_str(s):
    return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8').lower()

if modelo_seleccionado:
    for m in modelos:
        if normalize_str(modelo_seleccionado) in normalize_str(str(m)):
            ruta_modelo = m
            break

if ruta_modelo is None:
    ruta_modelo = modelos[0]

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

ruta_task_model = obtener_ruta_recurso('assets/ml_models/hand_landmarker.task')

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(ruta_task_model)),
    running_mode=VisionRunningMode.IMAGE,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

camara_activa = threading.Event()
camara_activa.set()

# Analiza la la posición de la mano y los 21 puntos de referencia que nos proporciona MediaPipe para identificar el gesto que realiza el usuario como: Mano abierta - rotar en eje z, Puño cerrado - rotar en ejes X e Y, Pulgar y meñique -zoom de acercamiento y alejamiento
# Si el gesto NO coincide con ninguno de los antes mencionados devuelve "Gesto desconocido"
def detectar_gesto(hand, label):
    try:
        if not hand or len(hand) < 21:
            return "Gesto desconocido"
        
        def distancia(p1, p2):
            return math.hypot(hand[p1].x - hand[p2].x, hand[p1].y - hand[p2].y)
        
        # Distancias entre puntas de dedos y muñeca
        indice_cerrado = distancia(8, 5) < 0.23
        medio_cerrado = distancia(12, 9) < 0.18
        anular_cerrado = distancia(16, 13) < 0.18
        menique_cerrado = distancia(20, 17) < 0.18
        pulgar_cerrado = distancia(4, 9) < 0.22
        
        print(
            f"I={distancia(8,5):.3f} "
            f"M={distancia(12,9):.3f} "
            f"A={distancia(16,13):.3f} "
            f"Me={distancia(20,17):.3f} "
            f"P={distancia(4,9):.3f}"
        )

        if (
            indice_cerrado and
            medio_cerrado and
            anular_cerrado and
            menique_cerrado and
            pulgar_cerrado
        ):
            return "Puño Cerrado"

        
        dedos_largos_abiertos = (hand[8].y < hand[5].y and 
                                 hand[12].y < hand[9].y and 
                                 hand[16].y < hand[13].y and 
                                 hand[20].y < hand[17].y)
        pulgar_abierto = distancia(4, 5) > 0.12
        if dedos_largos_abiertos and pulgar_abierto:
            return "Mano Abierta"
        
        dedos_medios_cerrados = (hand[8].y > hand[6].y and 
                                 hand[12].y > hand[10].y and 
                                 hand[16].y > hand[14].y)
        menique_abierto = hand[20].y < hand[18].y
        pulgar_abierto = distancia(4, 5) > 0.12
        distancia_valida = distancia(4, 20) > 0.15
        
        if dedos_medios_cerrados and menique_abierto and pulgar_abierto and distancia_valida:
            return "Pulgar y Meñique"

    except Exception as e:
        print(f"[Error en detector de gestos] {e}")
        
    return "Gesto desconocido"

estado = {
    "rot_x": 0.0, "rot_y": 0.0, "rot_z": 0.0, "zoom": 1.0,
    "target_rot_x": 0.0, "target_rot_y": 0.0
}
SMOOTHING = 0.25
prev_x = None
prev_y = None    

scene = pyrender.Scene(bg_color=[0, 0, 0, 1], ambient_light=[0.3, 0.3, 0.3])

mesh_trimesh = trimesh.load(str(ruta_modelo))
mesh_trimesh.apply_translation(-mesh_trimesh.centroid)
extents = mesh_trimesh.extents
if max(extents) > 0:
    mesh_trimesh.apply_scale(2.0 / max(extents))
    
if isinstance(mesh_trimesh, trimesh.Scene):
    geometrias_limpias = mesh_trimesh.dump()
    if isinstance(geometrias_limpias, trimesh.Scene):
        lista_geometrias = list(geometrias_limpias.geometry.values())
    elif isinstance(geometrias_limpias, list):
        lista_geometrias = geometrias_limpias
    else:
        lista_geometrias = [geometrias_limpias]
else:
    lista_geometrias = [mesh_trimesh]
    
primitivas_totales = []

if modo_visualizacion == "malla":
    DICCIONARIO_RESPALDO = {
        "cerebro": [0.80, 0.55, 0.60, 1.0], "corazon": [0.85, 0.15, 0.15, 1.0],
        "craneo": [0.88, 0.85, 0.78, 1.0], "esqueleto": [0.88, 0.85, 0.78, 1.0],
        "inferior": [0.70, 0.65, 0.60, 1.0], "superior": [0.70, 0.65, 0.60, 1.0],
        "ojo": [0.10, 0.40, 0.80, 1.0], "pulmon": [0.85, 0.45, 0.45, 1.0],
        "torax": [0.85, 0.80, 0.75, 1.0]
    }
    color_contingencia = [1.0, 1.0, 1.0, 1.0]
    if modelo_seleccionado:
        token_buscado = normalize_str(modelo_seleccionado)
        for clave, col in DICCIONARIO_RESPALDO.items():
            if clave in token_buscado:
                color_contingencia = col
                break

    for geom in lista_geometrias:
        vertices = geom.vertices
        aristas = geom.edges_unique
        indices_lineas = aristas.flatten().astype(np.uint32)
        num_vertices = len(vertices)
        colores_vertices = None

        if (hasattr(geom, 'visual') and geom.visual is not None and 
            hasattr(geom.visual, 'vertex_colors') and geom.visual.vertex_colors is not None and len(geom.visual.vertex_colors) > 0):
            colores_vertices = np.array(geom.visual.vertex_colors, dtype=np.float32)
            if colores_vertices.max() > 1.0: colores_vertices /= 255.0
            if colores_vertices.shape[1] == 3:
                colores_vertices = np.hstack((colores_vertices, np.ones((colores_vertices.shape[0], 1), dtype=np.float32)))
            colores_vertices[:, 3] = 1.0

        elif (hasattr(geom, 'visual') and geom.visual is not None and 
              hasattr(geom.visual, 'uv') and geom.visual.uv is not None and 
              hasattr(geom.visual, 'material') and geom.visual.material is not None and hasattr(geom.visual.material, 'baseColorTexture') and geom.visual.material.baseColorTexture is not None):
            try:
                textura = geom.visual.material.baseColorTexture
                tex_data = np.array(textura, dtype=np.float32)
                if tex_data.size > 0:
                    h, w = tex_data.shape[:2]
                    uvs = np.array(geom.visual.uv, dtype=np.float32)
                    u = np.clip(uvs[:, 0] % 1.0, 0.0, 1.0)
                    v = np.clip(1.0 - (uvs[:, 1] % 1.0), 0.0, 1.0)
                    x_indices = (u * (w - 1)).astype(np.int32)
                    y_indices = (v * (h - 1)).astype(np.int32)
                    colores_muestreados = tex_data[y_indices, x_indices]
                    if colores_muestreados.max() > 1.0: colores_muestreados /= 255.0
                    if colores_muestreados.shape[1] == 3:
                        colores_muestreados = np.hstack((colores_muestreados, np.ones((colores_muestreados.shape[0], 1), dtype=np.float32)))
                    colores_vertices = colores_muestreados
                    colores_vertices[:, 3] = 1.0
            except Exception:
                colores_vertices = None

        if colores_vertices is None:
            color_pieza = None
            if hasattr(geom, 'visual') and geom.visual is not None and hasattr(geom.visual, 'material'):
                mat = geom.visual.material
                if hasattr(mat, 'gltf_properties') and 'baseColorFactor' in mat.gltf_properties:
                    color_pieza = mat.gltf_properties['baseColorFactor']
                if color_pieza is None:
                    for attr in ['baseColorFactor', 'base_color', 'diffuse', 'main_color']:
                        if hasattr(mat, attr) and getattr(mat, attr) is not None:
                            val_np = np.array(getattr(mat, attr), dtype=np.float32)
                            if val_np.max() > 1.0: val_np /= 255.0
                            color_pieza = val_np[:4]
                            break
            if color_pieza is None or np.allclose(color_pieza[:3], [0.87843137, 0.87843137, 0.87843137], atol=1e-2) or np.allclose(color_pieza[:3], [1.0, 1.0, 1.0], atol=1e-2):
                color_pieza = color_contingencia
            color_pieza = list(color_pieza)
            if len(color_pieza) == 3: color_pieza.append(1.0)
            color_pieza[3] = 1.0
            colores_vertices = np.tile(color_pieza[:4], (num_vertices, 1)).astype(np.float32)

        material_malla = pyrender.MetallicRoughnessMaterial(roughnessFactor=1.0, metallicFactor=0.0, baseColorFactor=[1.0, 1.0, 1.0, 1.0])
        primitiva_linea = pyrender.Primitive(positions=vertices, indices=indices_lineas, color_0=colores_vertices, material=material_malla, mode=1)
        primitivas_totales.append(primitiva_linea)
            
    mesh = pyrender.Mesh(primitives=primitivas_totales)
else:
    mesh_fusionado = trimesh.util.concatenate(lista_geometrias)
    mesh = pyrender.Mesh.from_trimesh(mesh_fusionado) 
        
node_modelo = scene.add(mesh)
base_pose = np.eye(4, dtype=np.float32)

camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3.5], [0, 0, 0, 1]])
scene.add(camera, pose=camera_pose)

light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
scene.add(light, pose=camera_pose)

# Hilo encargado del procesamiento continuo de la cámara (Captura un nuevo fotograma, detecta la mano con MediaPipe, identifica el gesto realizado, calcula la rotación o el zoom, actualiza la transformación del modelo 3D) todo el procesamiento sucede de forma paralela al visor de Pyrender para mantener na interfaz fluida
def process_camera():
    global prev_x, prev_y
    while camara_activa.is_set():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        resultado = landmarker.detect(mp_image)
        
        gesto = "Gesto desconocido"
        mano_real = "Derecha"
        x, y = 0.0, 0.0
        mano_detectada = False
        
        try:
            if resultado.hand_landmarks and len(resultado.hand_landmarks) > 0:
                print("MANO DETECTADA")
                hand = resultado.hand_landmarks[0]
                hand_info = resultado.handedness[0]
            
                label = hand_info[0].category_name if isinstance(hand_info, list) else hand_info.category_name
                print("LABEL:", label)
                gesto = detectar_gesto(hand, label)
                print("GESTO:", gesto)
                mano_real = "Derecha" if label == "Left" else "Izquierda"
                x, y = hand[0].x, hand[0].y
                mano_detectada = True
                
            if mano_detectada:
                if gesto == "Puño Cerrado":
                    print("ENTRO A PUÑO")
                    
                    if prev_x is not None and prev_y is not None:
                        dx = x - prev_x
                        dy = y - prev_y
                        
                        print(f"dx={dx:.4f}  dy={dy:.4f}")
                        sensibilidad_y = 20.0
                        sensibilidad_x = 18.0
                    
                        if mano_real == "Derecha":
                            estado["target_rot_y"] += dx * sensibilidad_x
                            estado["target_rot_x"] += dy * sensibilidad_y
                        else:
                            estado["target_rot_y"] -= dx * sensibilidad_x
                            estado["target_rot_x"] -= dy * sensibilidad_y
                    
                    prev_x = x
                    prev_y = y

                elif gesto == "Mano Abierta":
                    estado["rot_z"] += -0.04 if mano_real == "Derecha" else 0.04
                    prev_x = None
                    prev_y = None
                    
                elif gesto == "Pulgar y Meñique":
                    estado["zoom"] += 0.02 if mano_real == "Derecha" else -0.02
                    prev_x = None
                    prev_y = None
                else:
                    prev_x = None
                    prev_y = None
                
                estado["zoom"] = max(0.5, min(estado["zoom"], 3.0))
                
            else:
                prev_x = None
                prev_y = None
                
        except Exception as e:
            print(f"[Error en Frame] {e}")
            prev_x = None
            prev_y = None
                
        estado["rot_x"] += (estado["target_rot_x"] - estado["rot_x"]) * SMOOTHING
        estado["rot_y"] += (estado["target_rot_y"] - estado["rot_y"]) * SMOOTHING
                
        rot_x_mat = np.array([[1, 0, 0, 0], [0, np.cos(estado["rot_x"]), -np.sin(estado["rot_x"]), 0], [0, np.sin(estado["rot_x"]), np.cos(estado["rot_x"]), 0], [0, 0, 0, 1]])
        rot_y_mat = np.array([[np.cos(estado["rot_y"]), 0, np.sin(estado["rot_y"]), 0], [0, 1, 0, 0], [-np.sin(estado["rot_y"]), 0, np.cos(estado["rot_y"]), 0], [0, 0, 0, 1]])
        rot_z_mat = np.array([[np.cos(estado["rot_z"]), -np.sin(estado["rot_z"]), 0, 0], [np.sin(estado["rot_z"]),  np.cos(estado["rot_z"]), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        scale = np.diag([estado["zoom"], estado["zoom"], estado["zoom"], 1])
        transform = rot_x_mat @ rot_y_mat @ rot_z_mat @ scale
        
        if camara_activa.is_set():
            try: scene.set_pose(node_modelo, pose=base_pose @ transform)
            except Exception: break
        time.sleep(0.001)

thread_cam = threading.Thread(target=process_camera, daemon=True)
thread_cam.start()

# Rtina segura de control de apagado
def escuchar_interfaz_padre():
    while camara_activa.is_set():
        try:
            linea = sys.stdin.readline().strip()
            if linea == "CLEAR":
                print("[Subproceso] Orden CLEAR recibida. Destruyendo ventanas...")
                cerrar_todo_limpio()
                break
        except Exception:
            break

def cerrar_todo_limpio():
    if camara_activa.is_set():
        camara_activa.clear()
    try: viewer.close()
    except Exception: pass
    thread_cam.join(timeout=0.4)
    if cap.isOpened():
        cap.release()
    try: landmarker.close()
    except Exception: pass
    cv2.destroyAllWindows()
    print("[Sistema] Hardware liberado de forma síncrona. Código de salida: 0")
    sys.exit(0)

thread_padre = threading.Thread(target=escuchar_interfaz_padre, daemon=True)
thread_padre.start()

# Iniciar visor en el hilo principal de manera bloqueante
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False, viewport_size=(800, 800))

# Si el usuario cierra la ventana de pyrender con la "X", se ejecuta esto:
cerrar_todo_limpio()