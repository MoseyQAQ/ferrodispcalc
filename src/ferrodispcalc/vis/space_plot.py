import numpy as np
import time
from typing import Optional, Tuple, Callable
try:
    from vispy import scene, app
    from vispy.scene.visuals import Mesh, Text, XYZAxis, Rectangle
    from vispy.visuals.visual import Visual
    from vispy.gloo import VertexBuffer, IndexBuffer
    from vispy.visuals.transforms import STTransform
    from vispy.visuals.filters import ShadingFilter
    from vispy.geometry import MeshData, create_cylinder, create_cone
    from vispy.util.transforms import rotate
    from vispy import gloo
    gloo.gl.use_gl("gl+") 
except ImportError:
    raise ImportError("VisPy is not installed. Please install it with `pip install vispy` or `pip install ferrodispcalc[vis]`.")

def _create_arrow_mesh(cyl_rows: int, cyl_cols: int, cone_cols: int) -> "MeshData":
    cyl = create_cylinder(cyl_rows, cyl_cols, radius=[0.05, 0.05], length=0.8)
    cone = create_cone(cone_cols, radius=0.1, length=0.2)
    verts = np.vstack((cyl.get_vertices(), cone.get_vertices() + [0, 0, 0.8]))
    faces = np.vstack((cyl.get_faces(), cone.get_faces() + len(cyl.get_vertices())))
    return MeshData(vertices=verts, faces=faces)

_ARROW_MESH_PRESETS: dict[str, tuple[int, int, int]] = {
    "high": (20, 32, 32),
    "medium": (12, 20, 20),
    "low": (6, 12, 12),
}
_ARROW_MESH_CACHE: dict[tuple[int, int, int], MeshData] = {}

def _get_arrow_mesh(cyl_rows: int, cyl_cols: int, cone_cols: int) -> "MeshData":
    key = (int(cyl_rows), int(cyl_cols), int(cone_cols))
    mesh = _ARROW_MESH_CACHE.get(key)
    if mesh is None:
        mesh = _create_arrow_mesh(*key)
        _ARROW_MESH_CACHE[key] = mesh
    return mesh
_ARROW_BASE_RADIUS = 0.05


_INSTANCED_ARROW_VERT = """
attribute vec3 a_position;

attribute vec3 i_translate;
attribute vec3 i_r0;
attribute vec3 i_r1;
attribute vec3 i_r2;
attribute float i_length;
attribute float i_scale_xy;
attribute vec4 i_color;

varying vec4 v_color;

void main() {
    mat3 R = mat3(i_r0, i_r1, i_r2);  // columns
    vec3 scale = vec3(i_scale_xy, i_scale_xy, i_length);
    vec3 world_pos = i_translate + (R * (a_position * scale));
    v_color = i_color;
    gl_Position = $transform(vec4(world_pos, 1.0));
}
"""

_INSTANCED_ARROW_FRAG = """
varying vec4 v_color;
void main() {
    gl_FragColor = v_color;
}
"""


class InstancedArrowVisual(Visual):
    def __init__(self, base_vertices: np.ndarray, base_faces: np.ndarray, **kwargs):
        super().__init__(vcode=_INSTANCED_ARROW_VERT, fcode=_INSTANCED_ARROW_FRAG, **kwargs)
        self.set_gl_state('translucent', depth_test=True, cull_face=False)

        base_vertices = np.asarray(base_vertices, dtype=np.float32)
        base_faces = np.asarray(base_faces, dtype=np.uint32)

        self._base_vbo = VertexBuffer(base_vertices)
        self._index_buffer = IndexBuffer(base_faces)
        self._draw_mode = 'triangles'

        # Per-instance buffers (divisor=1)
        self._i_translate = VertexBuffer(np.zeros((1, 3), dtype=np.float32))
        self._i_translate.divisor = 1

        self._i_r0 = VertexBuffer(np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (1, 1)))
        self._i_r0.divisor = 1
        self._i_r1 = VertexBuffer(np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (1, 1)))
        self._i_r1.divisor = 1
        self._i_r2 = VertexBuffer(np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (1, 1)))
        self._i_r2.divisor = 1

        self._i_length = VertexBuffer(np.zeros((1, 1), dtype=np.float32))
        self._i_length.divisor = 1

        self._i_scale_xy = VertexBuffer(np.ones((1, 1), dtype=np.float32))
        self._i_scale_xy.divisor = 1

        self._i_color = VertexBuffer(np.zeros((1, 4), dtype=np.float32))
        self._i_color.divisor = 1

        self.shared_program['a_position'] = self._base_vbo
        self.shared_program['i_translate'] = self._i_translate
        self.shared_program['i_r0'] = self._i_r0
        self.shared_program['i_r1'] = self._i_r1
        self.shared_program['i_r2'] = self._i_r2
        self.shared_program['i_length'] = self._i_length
        self.shared_program['i_scale_xy'] = self._i_scale_xy
        self.shared_program['i_color'] = self._i_color

        self._instance_count = 0

    @staticmethod
    def _prepare_transforms(view):
        tr = view.transforms.get_transform()
        view.view_program.vert['transform'] = tr

    def set_data(
        self,
        translate: np.ndarray,
        r0: np.ndarray,
        r1: np.ndarray,
        r2: np.ndarray,
        length: np.ndarray,
        color: np.ndarray,
        scale_xy: np.ndarray,
    ) -> None:
        translate = np.asarray(translate, dtype=np.float32)
        r0 = np.asarray(r0, dtype=np.float32)
        r1 = np.asarray(r1, dtype=np.float32)
        r2 = np.asarray(r2, dtype=np.float32)
        length = np.asarray(length, dtype=np.float32)
        scale_xy = np.asarray(scale_xy, dtype=np.float32)
        color = np.asarray(color, dtype=np.float32)

        n = int(translate.shape[0])
        if n <= 0:
            # Keep buffers non-empty to satisfy gloo's instance-size checks.
            translate = np.zeros((1, 3), dtype=np.float32)
            r0 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            r1 = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
            r2 = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
            length = np.zeros((1, 1), dtype=np.float32)
            scale_xy = np.ones((1, 1), dtype=np.float32)
            color = np.zeros((1, 4), dtype=np.float32)
            self._instance_count = 0
        else:
            if length.ndim == 1:
                length = length.reshape(-1, 1)
            if scale_xy.ndim == 0:
                scale_xy = np.full((n, 1), float(scale_xy), dtype=np.float32)
            elif scale_xy.ndim == 1:
                scale_xy = scale_xy.reshape(-1, 1)
            self._instance_count = n

        self._i_translate.set_data(translate)
        self._i_r0.set_data(r0)
        self._i_r1.set_data(r1)
        self._i_r2.set_data(r2)
        self._i_length.set_data(length)
        self._i_scale_xy.set_data(scale_xy)
        self._i_color.set_data(color)
        self.update()


InstancedArrow = scene.visuals.create_visual_node(InstancedArrowVisual)


def _rotation_cols_from_direction(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.asarray(direction, dtype=np.float32)
    if d.ndim != 2 or d.shape[1] != 3:
        raise ValueError("direction must have shape (N, 3)")

    n = int(d.shape[0])
    if n == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty, empty

    c = d[:, 2].astype(np.float32, copy=False)

    v_x = (-d[:, 1]).astype(np.float32, copy=False)
    v_y = (d[:, 0]).astype(np.float32, copy=False)

    Vx = np.zeros((n, 3, 3), dtype=np.float32)
    Vx[:, 0, 2] = v_y
    Vx[:, 1, 2] = -v_x
    Vx[:, 2, 0] = -v_y
    Vx[:, 2, 1] = v_x

    Vx2 = Vx @ Vx
    I = np.eye(3, dtype=np.float32)
    R = I[None, :, :] + Vx

    not_opposite = c > -0.999999
    if np.any(not_opposite):
        factor = (1.0 / (1.0 + c[not_opposite])).astype(np.float32, copy=False)
        R[not_opposite] = R[not_opposite] + Vx2[not_opposite] * factor[:, None, None]
    if np.any(~not_opposite):
        R_180x = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
        R[~not_opposite] = R_180x

    # Return columns (GLSL mat3 constructor expects columns)
    return R[:, :, 0], R[:, :, 1], R[:, :, 2]


def _get_colors_from_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Map 3D vectors to RGB colors based on direction.
    
    Parameters:
    -----------
    vectors: np.ndarray
        The vectors of shape (N, 3).
        
    Returns:
    --------
    colors: np.ndarray
        The RGBA colors of shape (N, 4).
    """
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1e-9
    normalized = vectors / norms
    
    # Map [-1, 1] to [0, 1] for RGB
    rgb = (normalized + 1.0) / 2.0
    
    # Clip to ensure valid range
    rgb = np.clip(rgb, 0.0, 1.0)
    
    # Add alpha channel (fully opaque)
    rgb = rgb.astype(np.float32, copy=False)
    alpha = np.ones((rgb.shape[0], 1), dtype=rgb.dtype)
    colors = np.concatenate((rgb, alpha), axis=1)
    
    return colors

class SpaceProfileCanvas(scene.SceneCanvas):
    """
    Canvas for visualizing 3D vector fields using VisPy.
    """
    def __init__(self, 
                 data: np.ndarray, 
                 coord: np.ndarray,
                 title: str = "Space Profile",
                 size: Tuple[int, int] = (800, 600),
                 bgcolor: str = "white",
                 arrow_width: float = 2.0,
                 show_axis: bool = True,
                 autoplay: bool = False,
                 fps: float = 10.0,
                 loop: bool = True,
                 color_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 render: str = "mesh",
                 mesh_detail: str = "high",
                 cyl_rows: Optional[int] = None,
                 cyl_cols: Optional[int] = None,
                 cone_cols: Optional[int] = None,
                 mesh_chunk_size: int = 512,
                 profile: bool = False,
                 profile_every: int = 30):
        """
        Initialize the canvas.
        
        Parameters:
        -----------
        data: np.ndarray
            Vector data of shape (nframe, npoint, 3).
        coord: np.ndarray
            Coordinate data of shape (nframe, npoint, 3).
        title: str
            Window title.
        size: tuple
            Window size (width, height).
        bgcolor: str
            Background color.
        arrow_width: float
            The width of the arrow shaft in pixels.
        show_axis: bool
            Whether to show XYZ axes.
        autoplay: bool
            Whether to play frames automatically.
        fps: float
            Frames per second.
        loop: bool
            Whether to loop when reaching the last frame.
        color_func: callable, optional
            A function that takes (N, 3) vectors and returns (N, 4) RGBA colors.
        """
        super().__init__(keys='interactive', size=size, title=title, bgcolor=bgcolor)
        
        self.unfreeze()
        
        # Data storage
        self.data = data
        self.coord = coord
        self.coord_is_static = (coord.ndim == 2)
        self.nframe = data.shape[0]
        self.npoint = coord.shape[0] if self.coord_is_static else coord.shape[1]
        if data.ndim != 3 or data.shape[2] != 3:
            raise ValueError("data must have shape (nframe, npoint, 3)")
        if self.coord_is_static:
            if coord.shape != (self.npoint, 3):
                raise ValueError("coord must have shape (npoint, 3) or (nframe, npoint, 3)")
        else:
            if coord.ndim != 3 or coord.shape[2] != 3:
                raise ValueError("coord must have shape (npoint, 3) or (nframe, npoint, 3)")
        if data.shape[1] != self.npoint:
            raise ValueError(f"data npoint={data.shape[1]} does not match coord npoint={self.npoint}")
        self.current_frame = 0
        self.arrow_width = arrow_width
        self.show_axis = show_axis
        self.autoplay = autoplay
        self.play_fps = fps
        self.loop = loop
        self.color_func = color_func
        self.is_playing = False
        self.ortho = False

        render = str(render).lower().strip()
        if render not in {"auto", "mesh", "instanced"}:
            raise ValueError("render must be one of: 'auto', 'mesh', 'instanced'")
        mesh_detail = str(mesh_detail).lower().strip()
        if mesh_detail not in _ARROW_MESH_PRESETS:
            raise ValueError(f"mesh_detail must be one of: {sorted(_ARROW_MESH_PRESETS)}")

        self.render_choice = render
        self.mesh_detail = mesh_detail
        self.cyl_rows = None if cyl_rows is None else int(cyl_rows)
        self.cyl_cols = None if cyl_cols is None else int(cyl_cols)
        self.cone_cols = None if cone_cols is None else int(cone_cols)
        self.mesh_chunk_size = max(1, int(mesh_chunk_size))
        self.profile = bool(profile)
        self.profile_every = max(1, int(profile_every))
        self._profile_counter = 0
        self._profile_accum = {
            "colors_ms": 0.0,
            "build_ms": 0.0,
            "upload_ms": 0.0,
            "total_ms": 0.0,
            "frames": 0,
            "arrows": 0,
            "verts": 0,
            "faces": 0,
        }

        coords0 = coord if self.coord_is_static else coord[0]
        self._coord_min = np.min(coords0, axis=0)
        self._coord_max = np.max(coords0, axis=0)
        axis_len = float(np.max(self._coord_max - self._coord_min))
        self._axis_len = axis_len if axis_len > 0 else 1.0
        self._center = np.mean(coords0, axis=0)
        self._z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        if self.render_choice in {"auto", "instanced"}:
            self._render_mode = "instanced"
        else:
            self._render_mode = "mesh"

        if self.profile:
            print(
                f"[profile] enabled mode={self._render_mode} every={self.profile_every}",
                flush=True,
            )

        if self.cyl_rows is None and self.cyl_cols is None and self.cone_cols is None:
            cyl_rows0, cyl_cols0, cone_cols0 = _ARROW_MESH_PRESETS[self.mesh_detail]
        else:
            cyl_rows0, cyl_cols0, cone_cols0 = _ARROW_MESH_PRESETS[self.mesh_detail]
            if self.cyl_rows is not None:
                cyl_rows0 = self.cyl_rows
            if self.cyl_cols is not None:
                cyl_cols0 = self.cyl_cols
            if self.cone_cols is not None:
                cone_cols0 = self.cone_cols

        meshdata = _get_arrow_mesh(cyl_rows0, cyl_cols0, cone_cols0)
        self._base_vertices = meshdata.get_vertices().astype(np.float32, copy=False)
        self._base_faces = meshdata.get_faces().astype(np.uint32, copy=False)
        self._base_vertices_size = int(self._base_vertices.shape[0])
        
        # Setup grid and view
        self.grid = self.central_widget.add_grid(margin=0)
        self.view = self.grid.add_view()
        self.view.camera = 'turntable'  # 3D camera
        self.view.camera.distance = 100
        
        # Initialize visuals
        self.arrow_mesh = None
        self.arrow_instanced = None
        self.text = None
        self.play_text = None
        self.proj_text = None
        self.view_btn_bg = {}
        self.view_btn_text = {}
        self._active_view_plane = None
        self.progress_bg = None
        self.progress_fg = None
        self.progress_knob = None
        self.axis = None
        self.is_dragging_progress = False
        self._shading_filter = ShadingFilter(shading="smooth", ambient_light=(1, 1, 1, 0.5))
        self._warned_instancing_unsupported = False
        
        # Draw initial frame
        self._draw_frame(self.current_frame)
        
        # Add text for frame info
        self.text = Text(f"Frame: {self.current_frame + 1}/{self.nframe}", 
                         color='black', 
                         anchor_x='left', 
                         anchor_y='top',
                         pos=(10, 30), 
                         parent=self.central_widget)
        
        self.play_text = Text("▶", 
                              color='black',
                              anchor_x='left',
                              anchor_y='top',
                              pos=(10, 55),
                              parent=self.central_widget)

        self.proj_text = Text("PER", 
                              color='black',
                              anchor_x='left',
                              anchor_y='top',
                              pos=(70, 55),
                              parent=self.central_widget)

        for key, label in (("xy", "xy-view"), ("xz", "xz-view"), ("yz", "yz-view")):
            self.view_btn_bg[key] = Rectangle(
                center=(0, 0),
                width=10,
                height=10,
                radius=3,
                color=(0.92, 0.92, 0.92, 1.0),
                border_color=(0.0, 0.0, 0.0, 1.0),
                border_width=1,
                parent=self.central_widget,
            )
            self.view_btn_text[key] = Text(
                label,
                color="black",
                anchor_x="center",
                anchor_y="center",
                pos=(0, 0),
                parent=self.central_widget,
            )

        if self.nframe > 1:
            self.progress_bg = Rectangle(center=(0, 0), width=1, height=1, radius=0, color=(0.85, 0.85, 0.85, 1.0), border_color=(0.0, 0.0, 0.0, 1.0), border_width=1, parent=self.central_widget)
            self.progress_fg = Rectangle(center=(0, 0), width=1, height=1, radius=0, color=(0.2, 0.2, 0.2, 1.0), border_color=None, border_width=0, parent=self.central_widget)
            self.progress_knob = Rectangle(center=(0, 0), width=1, height=1, radius=0, color=(0.0, 0.0, 0.0, 1.0), border_color=None, border_width=0, parent=self.central_widget)

        self._update_overlay_layout()
        self._update_progress_visual()
        self._update_view_button_visual()

        self.timer = None
        if self.nframe > 1:
            interval = 1.0 / self.play_fps if self.play_fps > 0 else 0.1
            self.timer = app.Timer(interval=interval, connect=self._on_timer, start=False)
            if self.autoplay:
                self.play()

        self.events.resize.connect(self._on_resize)
        self.events.mouse_press.connect(self.on_mouse_press)
        self.events.mouse_move.connect(self.on_mouse_move)
        self.events.mouse_release.connect(self.on_mouse_release)

        self.freeze()
        
    def set_projection(self, ortho: bool) -> None:
        ortho = bool(ortho)
        self.ortho = ortho
        cam = self.view.camera
        state = {
            "center": getattr(cam, "center", (0, 0, 0)),
            "elevation": getattr(cam, "elevation", 30),
            "azimuth": getattr(cam, "azimuth", 45),
            "roll": getattr(cam, "roll", 0),
        }
        distance = getattr(cam, "distance", 100)
        fov = 0 if ortho else 60
        self.view.camera = scene.cameras.TurntableCamera(fov=fov, **state)
        self.view.camera.distance = distance
        if self.proj_text is not None:
            self.proj_text.text = "ORT" if ortho else "PER"
        self.update()

    def toggle_projection(self) -> None:
        self.set_projection(not self.ortho)

    def set_view_plane(self, plane: str) -> None:
        plane = str(plane).lower().strip()
        if plane not in {"xy", "xz", "yz"}:
            return
        cam = self.view.camera
        if plane == "xy":
            cam.azimuth = 0
            cam.elevation = 90
        elif plane == "xz":
            cam.azimuth = 90
            cam.elevation = 0
        else:
            cam.azimuth = 0
            cam.elevation = 0
        if hasattr(cam, "roll"):
            cam.roll = 0
        self._active_view_plane = plane
        self._update_view_button_visual()
        self.update()

    def _draw_frame(self, frame_idx: int):
        """
        Draw a specific frame.
        """
        if not (0 <= frame_idx < self.nframe):
            return

        t_total0 = time.perf_counter()
        vectors = self.data[frame_idx]
        coords = self.coord if self.coord_is_static else self.coord[frame_idx]
        
        # Calculate colors
        t_colors0 = time.perf_counter()
        if self.color_func:
            colors = self.color_func(vectors)
        else:
            colors = _get_colors_from_vectors(vectors)
        t_colors1 = time.perf_counter()

        t_build0 = time.perf_counter()
        lengths_all = np.linalg.norm(vectors, axis=1).astype(np.float32, copy=False)
        mask = lengths_all > 0
        if np.any(mask):
            coords_m = coords[mask].astype(np.float32, copy=False)
            vectors_m = vectors[mask].astype(np.float32, copy=False)
            colors_m = colors[mask].astype(np.float32, copy=False)
            lengths = lengths_all[mask]
            arrow_count = int(coords_m.shape[0])

            direction = np.zeros_like(vectors_m, dtype=np.float32)
            np.divide(vectors_m, lengths[:, None], out=direction, where=lengths[:, None] > 0)
        else:
            coords_m = np.empty((0, 3), dtype=np.float32)
            colors_m = np.empty((0, 4), dtype=np.float32)
            lengths = np.empty((0,), dtype=np.float32)
            direction = np.empty((0, 3), dtype=np.float32)
            arrow_count = 0

        width_factor = max(0.05, float(self.arrow_width) / 2.0)
        min_radius = self._axis_len * 1e-4
        max_radius = self._axis_len * 0.02
        min_scale_xy = float(min_radius / _ARROW_BASE_RADIUS)
        max_scale_xy = float(max_radius / _ARROW_BASE_RADIUS)
        scale_xy = np.clip(lengths * width_factor, min_scale_xy, max_scale_xy).astype(np.float32, copy=False)

        if self._render_mode == "instanced":
            try:
                from vispy.gloo import gl  # type: ignore
                instancing_ok = hasattr(gl, "glVertexAttribDivisor") and hasattr(gl, "glDrawElementsInstanced")
            except Exception:
                instancing_ok = False

            if not instancing_ok:
                if not self._warned_instancing_unsupported:
                    self._warned_instancing_unsupported = True
                    print("[warn] instanced rendering not supported by current OpenGL backend; falling back to CPU mesh", flush=True)
                self._render_mode = "mesh"

        if self._render_mode == "instanced":
            r0, r1, r2 = _rotation_cols_from_direction(direction) if arrow_count else (coords_m, coords_m, coords_m)
            t_build1 = time.perf_counter()

            if self.arrow_instanced is None:
                self.arrow_instanced = InstancedArrow(self._base_vertices, self._base_faces, parent=self.view.scene)

            t_upload0 = time.perf_counter()
            self.arrow_instanced.set_data(
                translate=coords_m,
                r0=r0,
                r1=r1,
                r2=r2,
                length=lengths,
                color=colors_m,
                scale_xy=scale_xy,
            )
            t_upload1 = time.perf_counter()

            vertices = np.empty((0, 3), dtype=np.float32)
            faces = np.empty((0, 3), dtype=np.uint32)
        else:
            base_faces = self._base_faces
            base_vertices = self._base_vertices
            base_vertices_size = self._base_vertices_size

            if arrow_count == 0:
                vertices = np.empty((0, 3), dtype=np.float32)
                faces = np.empty((0, 3), dtype=np.uint32)
                vcolors = np.empty((0, 4), dtype=np.float32)
            else:
                V = int(base_vertices_size)
                chunk = int(self.mesh_chunk_size)

                vertices_parts = []
                faces_parts = []
                colors_parts = []
                offset = 0

                # Rotation that maps +Z to direction: R = I + [v]x + [v]x^2 * 1/(1+c)
                # where v = z x d, c = z · d, special-case c ~ -1 (opposite direction).
                I = np.eye(3, dtype=np.float32)
                R_180x = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)

                for start in range(0, arrow_count, chunk):
                    stop = min(arrow_count, start + chunk)
                    d = direction[start:stop]
                    c = d[:, 2].astype(np.float32, copy=False)

                    v_x = (-d[:, 1]).astype(np.float32, copy=False)
                    v_y = (d[:, 0]).astype(np.float32, copy=False)

                    Vx = np.zeros((stop - start, 3, 3), dtype=np.float32)
                    Vx[:, 0, 2] = v_y
                    Vx[:, 1, 2] = -v_x
                    Vx[:, 2, 0] = -v_y
                    Vx[:, 2, 1] = v_x

                    Vx2 = Vx @ Vx
                    R = I[None, :, :] + Vx

                    not_opposite = c > -0.999999
                    if np.any(not_opposite):
                        factor = (1.0 / (1.0 + c[not_opposite])).astype(np.float32, copy=False)
                        R[not_opposite] = R[not_opposite] + Vx2[not_opposite] * factor[:, None, None]
                    if np.any(~not_opposite):
                        R[~not_opposite] = R_180x

                    lengths_chunk = lengths[start:stop]
                    scale_factors = np.empty((stop - start, 3), dtype=np.float32)
                    scale_xy_chunk = scale_xy[start:stop]
                    scale_factors[:, 0] = scale_xy_chunk
                    scale_factors[:, 1] = scale_xy_chunk
                    scale_factors[:, 2] = lengths_chunk

                    scaled = base_vertices[None, :, :] * scale_factors[:, None, :]
                    rotated = np.einsum("nvi,nji->nvj", scaled, R, optimize=True)
                    verts = rotated + coords_m[start:stop, None, :]
                    verts = verts.reshape(-1, 3).astype(np.float32, copy=False)

                    offs = (np.arange(stop - start, dtype=np.uint32) * V + offset).reshape(-1, 1, 1)
                    fac = (base_faces[None, :, :] + offs).reshape(-1, 3).astype(np.uint32, copy=False)

                    col = np.repeat(colors_m[start:stop], V, axis=0).astype(np.float32, copy=False)

                    vertices_parts.append(verts)
                    faces_parts.append(fac)
                    colors_parts.append(col)
                    offset += (stop - start) * V

                vertices = np.vstack(vertices_parts) if len(vertices_parts) > 1 else vertices_parts[0]
                faces = np.vstack(faces_parts) if len(faces_parts) > 1 else faces_parts[0]
                vcolors = np.vstack(colors_parts) if len(colors_parts) > 1 else colors_parts[0]

            t_build1 = time.perf_counter()

            if self.arrow_mesh is None:
                t_upload0 = time.perf_counter()
                mesh_data = MeshData(vertices=vertices, faces=faces, vertex_colors=vcolors)
                self.arrow_mesh = Mesh(meshdata=mesh_data, parent=self.view.scene)
                self.arrow_mesh.attach(self._shading_filter)
                t_upload1 = time.perf_counter()
            else:
                t_upload0 = time.perf_counter()
                self.arrow_mesh.set_data(vertices=vertices, faces=faces, vertex_colors=vcolors)
                t_upload1 = time.perf_counter()

        if self.profile:
            t_total1 = time.perf_counter()
            self._profile_counter += 1
            self._profile_accum["colors_ms"] += (t_colors1 - t_colors0) * 1000.0
            self._profile_accum["build_ms"] += (t_build1 - t_build0) * 1000.0
            self._profile_accum["upload_ms"] += (t_upload1 - t_upload0) * 1000.0
            self._profile_accum["total_ms"] += (t_total1 - t_total0) * 1000.0
            self._profile_accum["frames"] += 1
            self._profile_accum["arrows"] = int(arrow_count)
            self._profile_accum["verts"] = int(vertices.shape[0]) if self._render_mode == "mesh" else int(arrow_count) * int(self._base_vertices_size)
            self._profile_accum["faces"] = int(faces.shape[0]) if self._render_mode == "mesh" else int(arrow_count) * int(self._base_faces.shape[0])

            if self._profile_counter % self.profile_every == 0:
                frames = max(1, int(self._profile_accum["frames"]))
                colors_ms = self._profile_accum["colors_ms"] / frames
                build_ms = self._profile_accum["build_ms"] / frames
                upload_ms = self._profile_accum["upload_ms"] / frames
                total_ms = self._profile_accum["total_ms"] / frames
                arrows = self._profile_accum["arrows"]
                verts = self._profile_accum["verts"]
                faces_n = self._profile_accum["faces"]
                print(
                    f"[profile] frame={frame_idx + 1}/{self.nframe} "
                    f"arrows={arrows} verts={verts} faces={faces_n} "
                    f"colors={colors_ms:.2f}ms build={build_ms:.2f}ms upload={upload_ms:.2f}ms total={total_ms:.2f}ms",
                    flush=True,
                )
                self._profile_accum["colors_ms"] = 0.0
                self._profile_accum["build_ms"] = 0.0
                self._profile_accum["upload_ms"] = 0.0
                self._profile_accum["total_ms"] = 0.0
                self._profile_accum["frames"] = 0
            
        # Update camera center if first frame (optional)
        if frame_idx == 0:
            self.view.camera.center = self._center
            if self.show_axis:
                if self.axis is None:
                    self.axis = XYZAxis(parent=self.view.scene)
                self.axis.transform = STTransform(scale=(self._axis_len, self._axis_len, self._axis_len), translate=self._coord_min)

    def on_key_press(self, event):
        """
        Handle key presses for navigation.
        """
        if event.key == 'Space':
            self.toggle_play()
            return
        if event.key == 'Right':
            self.current_frame = (self.current_frame + 1) % self.nframe
            self._update_scene()
        elif event.key == 'Left':
            self.current_frame = (self.current_frame - 1) % self.nframe
            self._update_scene()
            
    def on_mouse_press(self, event):
        x, y = event.pos
        # Note: event.pos is in window coordinates. (0,0) is top-left usually.
        # But depending on backend it might vary.
        # We define progress bar at top area.
        
        # Check play button
        if self.nframe > 1 and self.play_text is not None:
            # Simple bounding box check for play button
            # It's at (10, 55). Let's give it a generous area.
            if (x < 60) and (40 < y < 80):
                self.toggle_play()
                event.handled = True
                return

        if self.proj_text is not None:
            if (60 < x < 140) and (40 < y < 80):
                self.toggle_projection()
                event.handled = True
                return

        plane = self._hit_test_view_buttons(x, y)
        if plane is not None:
            self.set_view_plane(plane)
            event.handled = True
            return

        # Check progress bar
        # We relax the Y check significantly to handle coordinate system diffs and touch accuracy
        # The bar is conceptually at top_y=80, height=12.
        # We accept any click in the top area (e.g. y < 150) that is within X range.
        if self.nframe > 1 and self.progress_bg is not None:
            bar_x, bar_y, bar_w, bar_h = self._get_progress_bar_geometry()
            
            # Relaxed hit test
            hit_y = (y < 150) # Top area
            hit_x = (bar_x <= x <= bar_x + bar_w)
            
            if hit_x and hit_y:
                self.pause()
                self.is_dragging_progress = True
                # Normalize x relative to bar width
                ratio = (x - bar_x) / bar_w
                self._set_frame_from_progress_ratio(ratio)
                event.handled = True
                return

    def on_mouse_move(self, event):
        if self.is_dragging_progress and self.nframe > 1:
            x, _ = event.pos
            bar_x, bar_y, bar_w, bar_h = self._get_progress_bar_geometry()
            ratio = (x - bar_x) / bar_w
            self._set_frame_from_progress_ratio(ratio)
            event.handled = True

    def on_mouse_release(self, event):
        if self.is_dragging_progress:
            self.is_dragging_progress = False
            event.handled = True

    def _set_frame_from_progress_ratio(self, ratio: float) -> None:
        ratio = float(np.clip(ratio, 0.0, 1.0))
        frame = int(round(ratio * (self.nframe - 1)))
        if frame != self.current_frame:
            self.current_frame = frame
            self._update_scene()


    def _on_timer(self, event):
        if not self.is_playing:
            return
        if self.nframe <= 1:
            self.pause()
            return
        if self.current_frame == self.nframe - 1:
            if self.loop:
                self.current_frame = 0
            else:
                self.pause()
                return
        else:
            self.current_frame += 1
        self._update_scene()

    def _get_progress_bar_geometry(self) -> tuple[float, float, float, float]:
        margin = 10.0
        top_y = 80.0
        bar_w = max(80.0, float(self.size[0]) - margin * 2.0)
        bar_h = 12.0
        bar_x = margin
        bar_y = top_y
        return bar_x, bar_y, bar_w, bar_h

    def _get_view_button_geometries(self) -> dict[str, tuple[float, float, float, float]]:
        margin = 10.0
        btn_w = 90.0
        btn_h = 26.0
        gap = 8.0
        top_y = 110.0 if self.nframe > 1 else 80.0
        x = margin
        return {
            "xy": (x, top_y + 0.0 * (btn_h + gap), btn_w, btn_h),
            "xz": (x, top_y + 1.0 * (btn_h + gap), btn_w, btn_h),
            "yz": (x, top_y + 2.0 * (btn_h + gap), btn_w, btn_h),
        }

    def _hit_test_view_buttons(self, x: float, y: float) -> Optional[str]:
        geoms = self._get_view_button_geometries()
        for key, (bx, by, bw, bh) in geoms.items():
            if (bx <= x <= bx + bw) and (by <= y <= by + bh):
                return key
        return None

    def _set_frame_from_progress_x(self, x: float) -> None:
        pass

    def _update_overlay_layout(self) -> None:
        if self.text is not None:
            self.text.pos = (10, 30)
        if self.play_text is not None:
            self.play_text.pos = (10, 55)
        if self.proj_text is not None:
            self.proj_text.pos = (70, 55)

        geoms = self._get_view_button_geometries()
        for key, (bx, by, bw, bh) in geoms.items():
            bg = self.view_btn_bg.get(key)
            tx = self.view_btn_text.get(key)
            if bg is not None:
                bg.center = (bx + bw / 2.0, by + bh / 2.0)
                bg.width = bw
                bg.height = bh
            if tx is not None:
                tx.pos = (bx + bw / 2.0, by + bh / 2.0)

        if self.nframe <= 1:
            return
        if self.progress_bg is None or self.progress_fg is None or self.progress_knob is None:
            return

        bar_x, bar_y, bar_w, bar_h = self._get_progress_bar_geometry()
        self.progress_bg.center = (bar_x + bar_w / 2.0, bar_y + bar_h / 2.0)
        self.progress_bg.width = bar_w
        self.progress_bg.height = bar_h

        knob_w = 10.0
        knob_h = bar_h + 6.0
        self.progress_knob.width = knob_w
        self.progress_knob.height = knob_h

    def _update_progress_visual(self) -> None:
        if self.nframe <= 1:
            return
        if self.progress_bg is None or self.progress_fg is None or self.progress_knob is None:
            return

        bar_x, bar_y, bar_w, bar_h = self._get_progress_bar_geometry()
        ratio = 0.0 if self.nframe <= 1 else self.current_frame / (self.nframe - 1)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        fg_w = max(1.0, bar_w * ratio)

        self.progress_fg.height = bar_h
        self.progress_fg.width = fg_w
        self.progress_fg.center = (bar_x + fg_w / 2.0, bar_y + bar_h / 2.0)

        knob_x = bar_x + bar_w * ratio
        self.progress_knob.center = (knob_x, bar_y + bar_h / 2.0)

    def _on_resize(self, event):
        self._update_overlay_layout()
        self._update_progress_visual()
        self._update_view_button_visual()

    def _update_view_button_visual(self) -> None:
        for key, bg in self.view_btn_bg.items():
            if key == self._active_view_plane:
                bg.color = (0.75, 0.75, 0.75, 1.0)
            else:
                bg.color = (0.92, 0.92, 0.92, 1.0)

    def play(self):
        if self.timer is None:
            return
        self.is_playing = True
        self.play_text.text = "⏸"
        self.timer.start()
        self.update()

    def pause(self):
        if self.timer is None:
            return
        self.is_playing = False
        self.play_text.text = "▶"
        self.timer.stop()
        self.update()

    def toggle_play(self):
        if self.timer is None:
            return
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def _update_scene(self):
        self._draw_frame(self.current_frame)
        if self.text:
            self.text.text = f"Frame: {self.current_frame + 1}/{self.nframe}"
        self._update_progress_visual()
        self.update()

def space_profile(data: np.ndarray, 
                  coord: Optional[np.ndarray] = None,
                  title: str = "3D Vector Field",
                  scale: float = 1.0,
                  arrow_width: float = 2.0,
                  show_axis: bool = True,
                  autoplay: bool = False,
                  fps: float = 10.0,
                  loop: bool = True,
                  color_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                  render: str = "mesh",
                  mesh_detail: str = "high",
                  max_arrows: Optional[int] = None,
                  cyl_rows: Optional[int] = None,
                  cyl_cols: Optional[int] = None,
                  cone_cols: Optional[int] = None,
                  mesh_chunk_size: int = 512,
                  profile: bool = False,
                  profile_every: int = 30) -> None:
    """
    Plot 3D vector field using VisPy.
    
    Parameters:
    -----------
    data: np.ndarray
        Vector data. Supported shapes:
        - (nframe, nx, ny, nz, 3)
        - (nx, ny, nz, 3)
        - (nframe, npoint, 3) (if coord is provided)
        - (npoint, 3) (if coord is provided)
    coord: np.ndarray, optional
        Coordinate data. Supported shapes:
        - (nframe, npoint, 3)
        - (npoint, 3)
    title: str
        Window title.
    scale: float
        Scaling factor for vectors (default 1.0).
    arrow_width: float
        The width of the arrow shaft in pixels.
    show_axis: bool
        Whether to show XYZ axes.
    autoplay: bool
        Whether to play frames automatically.
    fps: float
        Frames per second.
    loop: bool
        Whether to loop when reaching the last frame.
    color_func: callable, optional
        A function that takes (N, 3) vectors and returns (N, 4) RGBA colors.
    render: str
        One of: 'mesh', 'instanced', 'auto'.
        - 'mesh': CPU assembles a huge mesh each frame (slow for many arrows).
        - 'instanced': keep one base (cylinder+cone) mesh and update per-arrow instance transforms (fast).
        - 'auto': same as 'instanced' (kept for backward compatibility).
    mesh_detail: str
        One of: 'low', 'medium', 'high'. Only used when rendering as 'mesh'.
    max_arrows: int, optional
        If set and the number of points is larger, downsample to this many arrows.
    cyl_rows, cyl_cols, cone_cols: int, optional
        Override the cylinder/cone mesh resolution (higher = smoother, slower).
    mesh_chunk_size: int
        Batch size for vectorized mesh assembly (tune for memory vs speed).
    profile: bool
        Print CPU-side timing for per-frame data assembly and upload calls.
    profile_every: int
        Print one timing line every N rendered frames (mesh mode).
    """
    
    # 1. Normalize Input Shapes
    
    # Handle single frame cases by adding dimension
    if data.ndim == 4 and coord is None: # (nx, ny, nz, 3) -> (1, nx, ny, nz, 3)
        data = data[np.newaxis, ...]
    elif data.ndim == 2 and coord is not None: # (npoint, 3) -> (1, npoint, 3)
        data = data[np.newaxis, ...]
        
    # Keep (npoint, 3) coordinates as-is; they can be treated as static across frames.
            
    # 2. Generate Coordinates if needed
    if coord is None:
        if data.ndim != 5:
            raise ValueError(f"Invalid data shape {data.shape} for grid mode. Expected (nframe, nx, ny, nz, 3).")
            
        nframe, nx, ny, nz, _ = data.shape
        
        # Generate grid coordinates
        # mgrid returns (3, nx, ny, nz), transpose to (nx, ny, nz, 3)
        grid = np.mgrid[0:nx, 0:ny, 0:nz].T
        # Flatten spatial dims: (npoint, 3)
        coord = grid.reshape(-1, 3).astype(np.float32, copy=False)
        
        # Flatten data spatial dims
        data = data.reshape(nframe, -1, 3)
    
    # 3. Validation
    if data.ndim != 3:
        raise ValueError("Internal shape error. Expected data shape (nframe, npoint, 3).")
    if coord.ndim == 3:
        if data.shape != coord.shape:
            raise ValueError(f"Data shape {data.shape} and coord shape {coord.shape} mismatch.")
    elif coord.ndim == 2:
        if data.shape[1] != coord.shape[0] or coord.shape[1] != 3:
            raise ValueError(f"Data shape {data.shape} and coord shape {coord.shape} mismatch.")
    else:
        raise ValueError("Internal shape error. Expected coord shape (nframe, npoint, 3) or (npoint, 3).")
        
    # Apply scaling
    if scale != 1.0:
        data = data * scale

    # Normalize dtypes (float32 is much faster / smaller for interactive rendering)
    data = np.asarray(data, dtype=np.float32)
    coord = np.asarray(coord, dtype=np.float32)

    # Optional downsample for interactive performance
    npoint = coord.shape[0] if coord.ndim == 2 else coord.shape[1]
    if max_arrows is not None and npoint > int(max_arrows):
        idx = np.linspace(0, npoint - 1, int(max_arrows), dtype=np.int64)
        data = data[:, idx, :]
        coord = coord[idx, :] if coord.ndim == 2 else coord[:, idx, :]

    # 4. Launch Visualization
    print(f"Launching 3D visualization... (Frames: {data.shape[0]}, Points: {data.shape[1]})")
    print("Controls: Left/Right Arrow to switch frames. Space to play/pause. Mouse to rotate/zoom.")
    
    canvas = SpaceProfileCanvas(
        data,
        coord,
        title=title,
        arrow_width=arrow_width,
        show_axis=show_axis,
        autoplay=autoplay,
        fps=fps,
        loop=loop,
        color_func=color_func,
        render=render,
        mesh_detail=mesh_detail,
        cyl_rows=cyl_rows,
        cyl_cols=cyl_cols,
        cone_cols=cone_cols,
        mesh_chunk_size=mesh_chunk_size,
        profile=profile,
        profile_every=profile_every,
    )
    canvas.show()
    app.run()
