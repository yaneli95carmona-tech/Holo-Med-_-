# Importacíon de librerías para desarrollar la interfaz gráfica del sistema
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk # Pillow nos permite cargar y mostrar imágenes dentro de la interfaz
import cv2 # Se emplea OpenCV para capturar el video desde la cámara 
import subprocess # Librerías utilizadas para ejecutar procesos externos
import sys
import os
import mediapipe as mp # Permite detectar las manos y sus puntos de referencia a través de Inteligencia Artificial
import multiprocessing # Crea procesos independientes durante la ejecución

print("APP CORRECTA CARGARA")

try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# Busca y devuelve la ruta de los archivos (modelos - imágenes) es compatible con PyInstaller para el empaquetado
def obtener_ruta_recurso(ruta_relativa):
    """Obtiene la ruta absoluta para recursos, compatible con desarrollo y PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        from pathlib import Path
        return str(Path(sys._MEIPASS) / ruta_relativa)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ruta_relativa)

# Clase principal del sistema controlando toda la aplicación, construcción de la interfaz, inicialización de la cámara, selección de modelos anatómicos, comunicación con el motor 3D y liberación de recursos al finalizar la aplicación
class HoloMedAtlas:
    def __init__(self, root):
        self.root = root
        self.root.title("HoloMed_Atlas - Sistema interactivo para la visualización de modelos anatómicos en 3D")
        self.root.state('zoomed')
        self.root.configure(bg="#f5f5f8")
        
        path_icono = obtener_ruta_recurso('assets/images/mi_logo.ico')
        if os.path.exists(path_icono):
            self.root.iconbitmap(path_icono)

        self.root.protocol("WM_DELETE_WINDOW", self.salir)
        
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.inf_modelo = {
            "Cerebro":            {"desc": "Regula movimientos y pensamientos. Centro del sistema nervioso central."},
            "Corazón":            {"desc": "Bomba muscular que distribuye sangre oxigenada a todo el organismo."},
            "Cráneo":             {"desc": "Estructura ósea de protección para el encéfalo y soporte facial."},
            "Esqueleto":          {"desc": "Armazón interno que permite el movimiento y protege órganos vitales."},
            "Extremidad Inferior":{"desc": "Comprende la estructura ósea y muscular para el soporte y marcha."},
            "Extremidad Superior":{"desc": "Diseñada para la manipulación precisa y gran rango de movimiento."},
            "Ojo":                {"desc": "Órgano fotorreceptor complejo que procesa estímulos visuales."},
            "Pulmón":             {"desc": "Centro del intercambio gaseoso para la oxigenación de la sangre."},
            "Tórax":              {"desc": "Caja torácica que protege el corazón y los pulmones."}
        }

        self.proceso_3d = None
        self.cap = None
        self.ejecutando_camara = False

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=obtener_ruta_recurso('assets/ml_models/hand_landmarker.task')),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.5)

        self.landmarker = HandLandmarker.create_from_options(options)

        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
        ]

        self.crear_interfaz()

    # Construye todos los elementos gráficos del sistema como el encabezado, logos institucionales, panel de control, descripción anatómica, guía de gestos, botones y área de visualiación de la cámara
    def crear_interfaz(self):
        self.header = tk.Frame(self.root, bg="white", height=200)
        self.header.pack(side="top", fill="x")    
        
        f_logos = tk.Frame(self.header, bg="white")
        f_logos.pack(side="left", fill="y", anchor="w")

        path_uv = obtener_ruta_recurso('assets/images/uv.png')
        if os.path.exists(path_uv):
            img_uv = Image.open(path_uv).resize((120, 120), Image.BICUBIC)
            self.logo_uv = ImageTk.PhotoImage(img_uv)
            tk.Label(f_logos, image=self.logo_uv, bg="white").pack(side="left", padx=(30, 20), pady=40)
            
        path_lab = obtener_ruta_recurso('assets/images/laboratorio.png')
        if os.path.exists(path_lab):
            img_lab = Image.open(path_lab).resize((180, 120), Image.BICUBIC)
            self.logo_lab = ImageTk.PhotoImage(img_lab)
            tk.Label(f_logos, image=self.logo_lab, bg="white").pack(side="left", padx=(0, 40), pady=40)
           
        path_itsx = obtener_ruta_recurso('assets/images/itsx.png')
        if os.path.exists(path_itsx):
            img_itsx = Image.open(path_itsx).resize((120, 120), Image.BICUBIC)
            self.logo_itsx = ImageTk.PhotoImage(img_itsx)
            tk.Label(f_logos, image=self.logo_itsx, bg="white").pack(side="left", padx=(0, 60), pady=40)
            
        f_titulo_contenedor = tk.Frame(self.header, bg="white")
        f_titulo_contenedor.pack(side="right", expand=True, fill="both")
        
        self.lbl_titulo = tk.Label(f_titulo_contenedor, text="HoloMed_Atlas", font=("Arial", 30, "bold"), bg="white", fg="#2e4f8f")
        self.lbl_titulo.place(relx=0.5, rely=0.5, anchor="center") 

        self.contenedor = tk.Frame(self.root, bg="#f5f5f8")
        self.contenedor.pack(expand=True, fill="both", padx=20, pady=20)
        
        self.panel = tk.Frame(self.contenedor, bg="white", width=570, relief="solid", borderwidth=1)
        self.panel.pack(side="left", fill="y", padx=10)
        self.panel.pack_propagate(False)
        tk.Label(self.panel, text="PANEL DE CONTROL", bg="white", fg="#2e4f8f", font=("Arial", 18, "bold")).pack(pady=(20,10))
        
        texto_soporte = ("Al seleccionar un modelo, la cámara se activa para calibrar la\ndetección gestual y pre-cargar el reconocimiento de IA\nantes de la proyección.")
        tk.Label(self.panel, text=texto_soporte, bg="white", fg="#3C3C3C", font=("Arial", 8, "italic"), wraplength=450, justify="center").pack(pady=(5, 10))
        self.modelo_var = tk.StringVar(value="Seleccione un Modelo")
        self.combo = ttk.Combobox(
            self.panel,
            textvariable=self.modelo_var,
            values=list(self.inf_modelo.keys()),
            state="readonly"
        )
        self.combo.pack(pady=20, padx=40, fill="x")
        self.combo.bind("<<ComboboxSelected>>", self.on_select)

        self.f_info = tk.LabelFrame(self.panel, text="Descripción del Modelo", bg="white", font=("Arial", 10, "bold"))
        self.f_info.pack(pady=15, padx=40, fill="x")
        self.l_info = tk.Label(
            self.f_info,
            text="Seleccione un modelo para ver su descripción.",
            bg="white", wraplength=440, justify="center", fg="#555555", font=("Arial", 11)
        )
        self.l_info.pack(padx=15, pady=20, fill="both", expand=True)

        self.f_guia = tk.LabelFrame(self.panel, text="GUÍA DE CONTROL GESTUAL", bg="#fcfcfc", font=("Arial", 10, "bold"))
        self.f_guia.pack(pady=10, padx=40, fill="x")
        guia = (
            "✋Mano Abierta  →  Rotar eje Z (Derecha + / Izquierda -)\n"
            "✊Puño Cerrado  →  Rotar eje X/Y (Derecha + / Izquierda -)\n"
            "🤙Pulgar + Meñique Mano Derecha  →  Zoom +\n"
            "🤙Pulgar + Meñique Mano Izquierda  →  Zoom −"
        )
        self.lbl_guia = tk.Label(self.f_guia, text=guia, bg="#fcfcfc", justify="left", anchor="w", font=("Bodoni MT", 9), wraplength=700, padx=15, pady=10)
        self.lbl_guia.pack(fill="x")

        self.btn_gen = tk.Button(
            self.panel, text="GENERAR MODELO EN 3D",
            bg="#f18d1a", fg="white", font=("Arial", 9, "bold"),
            command=lambda: self.lanzar(modo="solido"), state="disabled"
        )
        self.btn_gen.pack(pady=(10, 5), padx=40, fill="x")
        
        self.btn_malla = tk.Button(
            self.panel, text="CREAR ESCENA EN MALLA",
            bg="#1a73e8", fg="white", font=("Arial", 9, "bold"),
            command=lambda: self.lanzar(modo="malla"), state="disabled"
        )
        self.btn_malla.pack(pady=5, padx=40, fill="x")
        
        self.f_botones_extra = tk.Frame(self.panel, bg="white")
        self.f_botones_extra.pack(fill="x", padx=40, pady=(10, 20))

        self.btn_limpiar = tk.Button(
            self.f_botones_extra, text="Limpiar",
            bg="#28a745", fg="white", font=("Arial", 9, "bold"),
            command=self.limpiar, state="disabled"
        )
        self.btn_limpiar.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.btn_salir = tk.Button(
            self.f_botones_extra, text="Salir",
            bg="#dc3545", fg="white", font=("Arial", 9, "bold"),
            command=self.salir
        )
        self.btn_salir.pack(side="left", expand=True, fill="x", padx=(5, 0))

        self.area_derecha = tk.Frame(self.contenedor, bg="#f5f5f8")
        self.area_derecha.pack(side="right", expand=True, fill="both")
        
        self.l_cam = tk.Label(self.area_derecha, bg="black", text="Seleccione un Modelo para activar la cámara", fg="#3C3C3C", font=("Arial", 14), compound="center")
        self.l_cam.pack(side="top", fill="both", expand=True, padx=5, pady=(0,5), ipady=180)
        
        self.creditos = tk.Frame(self.area_derecha, bg="white", height=140)
        self.creditos.pack_propagate(False)
        self.creditos.pack(side="bottom", fill="x", padx=5, pady=5)
        texto_creditos = ("Carmona Velázquez Mayra Yaneli, Pérez Chacha Johan Armando, Zamitiz Durán Abdiel Farid\nASESORES:\nCerecedo Núñez Héctor Hugo, Padilla Sosa Patricia\nFacultad de Física, Laboratorio de Investigación en Óptica Aplicada\nUniversidad Veracruzana, Xalapa, México")
        tk.Label(self.creditos, text=texto_creditos, bg="white", fg="#888888", font=("Arial", 8, "bold", "italic"), justify="center").pack(expand=True)
        
    def on_select(self, e):
        modelo = self.modelo_var.get()
        if modelo in self.inf_modelo:
            self.l_info.config(text=self.inf_modelo[modelo]['desc'], fg="#222222", justify="center", wraplength=440)
            self.btn_gen.config(state="normal")
            self.btn_malla.config(state="normal")
            if not self.ejecutando_camara and self.proceso_3d is None:
                self.iniciar_camara()

    def iniciar_camara(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ejecutando_camara = True
        self.actualizar_camara()

    # captura fotogramas por segundo desde la webcam (lee la imágen, la invierte horizontalmente, detecta manos a través de MediaPipe, dibuja los puntos de referencia, actualiza la vista previa dentro de la interfaz) se ejecuta cada 30 ms utilizando root.after()
    def actualizar_camara(self):
        if not self.ejecutando_camara or self.cap is None:
            return

        ret, frame = self.cap.read()

        if ret:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            resultados = self.landmarker.detect(mp_image)

            if resultados.hand_landmarks:
                for hand_landmarks in resultados.hand_landmarks:
                    for c in self.HAND_CONNECTIONS:
                        p1 = hand_landmarks[c[0]]
                        p2 = hand_landmarks[c[1]]
                        x1 = int(p1.x * frame.shape[1])
                        y1 = int(p1.y * frame.shape[0])
                        x2 = int(p2.x * frame.shape[1])
                        y2 = int(p2.y * frame.shape[0])
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 120), 2)
                    for lm in hand_landmarks:
                        x = int(lm.x * frame.shape[1])
                        y = int(lm.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 4, (0, 80, 255), -1)

            w = self.l_cam.winfo_width()
            h = self.l_cam.winfo_height()
            if w > 10 and h > 10:
                frame = cv2.resize(frame, (w, h))

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.l_cam.imgtk = imgtk
            self.l_cam.configure(image=imgtk, text="")

        self.root.after(30, self.actualizar_camara)

    def lanzar(self, modo="solido"):
        modelo = self.modelo_var.get()
        if not modelo or modelo not in self.inf_modelo:
            return

        self.ejecutando_camara = False
        if self.cap:
            self.cap.release()
            self.cap = None
            
        import time
        time.sleep(0.3)

        self.l_cam.configure(image='', text="Modelo y Cámara transferidos al Motor 3D", fg="#2e6b2e")
        self.l_cam.imgtk = None

        if self.proceso_3d:
            try: self.proceso_3d.terminate()
            except Exception: pass

        self.btn_gen.config(state="disabled")
        self.btn_malla.config(state="disabled")
        
        # CORRECCIÓN DE RUTAS EXCLUSIVA PARA WINDOWS (.EXE) 
        if getattr(sys, 'frozen', False):
            
            motor = os.path.join(
                os.path.dirname(sys.executable),
                "hologram_engine.exe"
            )
            
            comando = [motor, modelo, modo]
        else:
            script = obtener_ruta_recurso("hologram_engine.py")
            
            comando = [
                sys.executable,
                script,
                modelo,
                modo
            ]
        
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE 
            
        try:
            self.proceso_3d = subprocess.Popen(
                comando,
                cwd=self.base_path if not hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable),
                startupinfo=startupinfo,
                stdin=subprocess.PIPE,
                text=True
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Error] No se pudo iniciar el motor 3D: {e}")
            self.btn_gen.config(state="normal")
            self.btn_malla.config(state="normal")
            return
        
        self.btn_limpiar.config(state="normal")
        self.revisar_proceso_3d()
        
    def revisar_proceso_3d(self):
        if self.proceso_3d is not None and self.proceso_3d.poll() is not None:
            self.proceso_3d = None
            self.btn_gen.config(state="normal")
            self.btn_malla.config(state="normal")    
            self.l_cam.configure(image='', text="Reactivando cámara local...", fg="#888888")
            self.root.after(200, self.iniciar_camara)
        else:
            if self.proceso_3d is not None:
                self.root.after(500, self.revisar_proceso_3d)

    def limpiar(self):
        self.ejecutando_camara = False
        if self.proceso_3d:
            try:
                self.proceso_3d.stdin.write("CLEAR\n")
                self.proceso_3d.stdin.flush()
                self.proceso_3d.wait(timeout=0.3)
            except Exception:
                try: self.proceso_3d.terminate()
                except Exception: pass
            self.proceso_3d = None
            
        if self.cap:
            self.cap.release()
            self.cap = None

        self.modelo_var.set("Seleccione un Modelo")
        self.btn_gen.config(state="disabled", text="GENERAR MODELO EN 3D", bg="#f18d1a")
        self.btn_malla.config(state="disabled", text="CREAR ESCENA EN MALLA", bg="#1a73e8")
        self.btn_limpiar.config(state="disabled")
        self.l_info.config(text="Seleccione un modelo para ver su descripción.", fg="#555555", justify="center", wraplength=440)
        self.l_cam.configure(image='', text="Selecciona un modelo para activar la cámara", fg="#888888")
        self.l_cam.imgtk = None

    def salir(self):
        self.ejecutando_camara = False
        if self.proceso_3d:
            try:
                self.proceso_3d.stdin.write("CLEAR\n")
                self.proceso_3d.stdin.flush()
                self.proceso_3d.wait(timeout=0.4)
            except Exception:
                try: self.proceso_3d.terminate()
                except Exception: pass
        if self.cap:
            self.cap.release()
        try: self.landmarker.close()
        except Exception: pass
        self.root.destroy()

if __name__ == "__main__":
    multiprocessing.freeze_support() # Requerido para evitar bucles infinitos en subprocesos empaquetados
    root = tk.Tk()
    app = HoloMedAtlas(root)
    root.mainloop()