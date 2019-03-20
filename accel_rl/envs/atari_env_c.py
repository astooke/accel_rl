
import numpy as np

from accel_rl.envs.atari_env import AtariEnv as AtariEnvNoC


class AtariEnv(AtariEnvNoC):

    """Requires modified ale_python_interface; saves a tiny bit of time by
    avoiding call to np.ctypeslib.as_ctypes() everytime a frame is grabbed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._c_frame_1 = np.ctypeslib.as_ctypes(self._raw_frame_1[:])
        self._c_frame_2 = np.ctypeslib.as_ctypes(self._raw_frame_2[:])

    def _get_screen(self, frame=1):
        frame = self._c_frame_1 if frame == 1 else self._c_frame_2
        self.ale.getScreenGrayscale(c_screen_data=frame)
