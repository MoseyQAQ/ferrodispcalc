import numpy as np
from typing import Optional, Tuple, Callable
try:
    from vispy import scene, app
    from vispy.scene.visuals import Arrow, Text, XYZAxis, Rectangle
    from vispy.visuals.transforms import STTransform
except ImportError:
    raise ImportError("VisPy is not installed. Please install it with `pip install vispy` or `pip install ferrodispcalc[vis]`.")

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
    alpha = np.ones((rgb.shape[0], 1))
    colors = np.hstack([rgb, alpha])
    
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
                 color_func: Optional[Callable[[np.ndarray], np.ndarray]] = None):
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
        self.nframe = data.shape[0]
        self.npoint = data.shape[1]
        self.current_frame = 0
        self.arrow_width = arrow_width
        self.show_axis = show_axis
        self.autoplay = autoplay
        self.play_fps = fps
        self.loop = loop
        self.color_func = color_func
        self.is_playing = False
        
        # Setup grid and view
        self.grid = self.central_widget.add_grid(margin=0)
        self.view = self.grid.add_view()
        self.view.camera = 'turntable'  # 3D camera
        self.view.camera.distance = 100
        
        # Initialize visuals
        self.arrow = None
        self.text = None
        self.play_text = None
        self.progress_bg = None
        self.progress_fg = None
        self.progress_knob = None
        self.axis = None
        self.is_dragging_progress = False
        
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

        if self.nframe > 1:
            self.progress_bg = Rectangle(center=(0, 0), width=1, height=1, radius=0, color=(0.85, 0.85, 0.85, 1.0), border_color=(0.0, 0.0, 0.0, 1.0), border_width=1, parent=self.central_widget)
            self.progress_fg = Rectangle(center=(0, 0), width=1, height=1, radius=0, color=(0.2, 0.2, 0.2, 1.0), border_color=None, border_width=0, parent=self.central_widget)
            self.progress_knob = Rectangle(center=(0, 0), width=1, height=1, radius=0, color=(0.0, 0.0, 0.0, 1.0), border_color=None, border_width=0, parent=self.central_widget)

        self._update_overlay_layout()
        self._update_progress_visual()

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
        
    def _draw_frame(self, frame_idx: int):
        """
        Draw a specific frame.
        """
        if not (0 <= frame_idx < self.nframe):
            return

        vectors = self.data[frame_idx]
        coords = self.coord[frame_idx]
        
        # Calculate colors
        if self.color_func:
            colors = self.color_func(vectors)
        else:
            colors = _get_colors_from_vectors(vectors)
        
        # Prepare Arrow data
        # ArrowVisual expects:
        # pos: (N, 3) start points? Or if arrows is set, pos is line segments?
        # Let's use the 'arrows' argument which defines (start, end) pairs for heads,
        # and 'pos' with connect='segments' for shafts.
        
        # Calculate end points
        ends = coords + vectors
        
        # Shafts: interleaved start and end points
        # shape: (2*N, 3)
        shaft_pos = np.empty((self.npoint * 2, 3), dtype=np.float32)
        shaft_pos[0::2] = coords
        shaft_pos[1::2] = ends
        
        # Shaft colors: repeat each color twice
        shaft_colors = np.repeat(colors, 2, axis=0)
        
        # Arrow heads: (start, end) pairs
        # shape: (N, 6)
        arrow_data = np.hstack([coords, ends])
        
        if self.arrow is None:
            self.arrow = Arrow(pos=shaft_pos, 
                               connect='segments', 
                               color=shaft_colors,
                               width=self.arrow_width,
                               arrows=arrow_data, 
                               arrow_color=colors,
                               arrow_type='stealth',
                               arrow_size=5,
                               parent=self.view.scene)
        else:
            self.arrow.set_data(pos=shaft_pos,
                                connect='segments',
                                color=shaft_colors,
                                width=self.arrow_width,
                                arrows=arrow_data)
            self.arrow.arrow_color = colors # Update head colors separately
            
        # Update camera center if first frame (optional)
        if frame_idx == 0:
            center = np.mean(coords, axis=0)
            self.view.camera.center = center
            if self.show_axis:
                coord_min = np.min(coords, axis=0)
                coord_max = np.max(coords, axis=0)
                axis_len = float(np.max(coord_max - coord_min))
                if axis_len <= 0:
                    axis_len = 1.0
                if self.axis is None:
                    self.axis = XYZAxis(parent=self.view.scene)
                self.axis.transform = STTransform(scale=(axis_len, axis_len, axis_len), translate=coord_min)

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

    def _set_frame_from_progress_x(self, x: float) -> None:
        pass

    def _update_overlay_layout(self) -> None:
        if self.text is not None:
            self.text.pos = (10, 30)
        if self.play_text is not None:
            self.play_text.pos = (10, 55)

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
                  color_func: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> None:
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
    """
    
    # 1. Normalize Input Shapes
    
    # Handle single frame cases by adding dimension
    if data.ndim == 4 and coord is None: # (nx, ny, nz, 3) -> (1, nx, ny, nz, 3)
        data = data[np.newaxis, ...]
    elif data.ndim == 2 and coord is not None: # (npoint, 3) -> (1, npoint, 3)
        data = data[np.newaxis, ...]
        
    if coord is not None:
        if coord.ndim == 2: # (npoint, 3) -> (1, npoint, 3)
            coord = coord[np.newaxis, ...]
            
    # 2. Generate Coordinates if needed
    if coord is None:
        if data.ndim != 5:
            raise ValueError(f"Invalid data shape {data.shape} for grid mode. Expected (nframe, nx, ny, nz, 3).")
            
        nframe, nx, ny, nz, _ = data.shape
        
        # Generate grid coordinates
        # mgrid returns (3, nx, ny, nz), transpose to (nx, ny, nz, 3)
        grid = np.mgrid[0:nx, 0:ny, 0:nz].T
        # Flatten spatial dims: (nx*ny*nz, 3)
        flat_grid = grid.reshape(-1, 3)
        
        # Replicate for all frames: (nframe, npoint, 3)
        coord = np.tile(flat_grid, (nframe, 1, 1))
        
        # Flatten data spatial dims
        data = data.reshape(nframe, -1, 3)
    
    # 3. Validation
    if data.ndim != 3 or coord.ndim != 3:
        raise ValueError("Internal shape error. Expected (nframe, npoint, 3).")
    if data.shape != coord.shape:
        raise ValueError(f"Data shape {data.shape} and coord shape {coord.shape} mismatch.")
        
    # Apply scaling
    if scale != 1.0:
        data = data * scale

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
        color_func=color_func
    )
    canvas.show()
    app.run()

if __name__ == '__main__':
    # Test code
    nx, ny, nz = 5, 5, 5
    nframe = 20
    data = np.random.rand(nframe, nx, ny, nz, 3) - 0.5
    space_profile(data, scale=0.5, arrow_width=3.0, show_axis=True, autoplay=False, fps=10, loop=True)
