import math
import threading
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from pose_engine import GymPoseEngine, SharedState, calculate_angle, cosine_similarity


class CoreTests(unittest.TestCase):
    def test_right_angle(self):
        self.assertAlmostEqual(calculate_angle((1, 0), (0, 0), (0, 1)), 90.0)

    def test_straight_angle(self):
        self.assertAlmostEqual(calculate_angle((-1, 0), (0, 0), (1, 0)), 180.0)

    def test_degenerate_angle_is_finite(self):
        self.assertEqual(calculate_angle((0, 0), (0, 0), (1, 0)), 0.0)

    def test_cosine_similarity(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine_similarity(np.array([1., 0.]), np.array([1., 0.])), 1.0)
        self.assertEqual(cosine_similarity(None, [1]), -1.0)

    def test_shared_state_snapshot_is_copy(self):
        state = SharedState()
        state.publish({("user_001", "Squat"): 3}, {"user_001": "A"}, {}, 12.5, 1)
        snapshot = state.snapshot()
        snapshot["counts"].clear()
        self.assertEqual(state.snapshot()["counts"][("user_001", "Squat")], 3)

    def test_shared_state_threaded_publish(self):
        state = SharedState()
        errors = []
        def publish_many():
            try:
                for i in range(500):
                    state.publish({("u", "Squat"): i}, {"u": "User"}, {}, 20, 1)
                    state.snapshot()
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=publish_many) for _ in range(4)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertEqual(errors, [])
        self.assertIn(state.snapshot()["counts"][("u", "Squat")], range(500))

    def test_enrolled_face_scan_does_not_block_live_frame(self):
        class SlowFaceApp:
            def get(self, _frame):
                time.sleep(0.2)
                return []

        class Frame:
            shape = (100, 100, 3)
            def copy(self):
                return self

        engine = GymPoseEngine.__new__(GymPoseEngine)
        engine.face_executor = ThreadPoolExecutor(max_workers=1)
        engine.face_future = None
        engine.face_future_frame = -1
        engine.face_future_people = []
        engine.face_app = SlowFaceApp()
        engine.gallery = {"user_001": {"embedding": [1.0, 0.0]}}
        engine.face_scan_period = 60
        engine.frame_no = 0
        engine.face_future_shape = None
        engine.face_future = None
        pose = [SimpleNamespace(x=.5, y=.5)]
        started = time.perf_counter()
        try:
            engine._face_ids(Frame(), [pose], [(1, "guest-001")])
            elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 0.1)
        finally:
            engine.face_executor.shutdown(wait=True)

    def test_completed_face_scan_assigns_enrolled_user_to_track(self):
        class Frame:
            shape = (100, 100, 3)

        engine = GymPoseEngine.__new__(GymPoseEngine)
        engine.face_executor = object()
        engine.face_future = Future()
        engine.face_future.set_result([SimpleNamespace(
            det_score=.95,
            normed_embedding=np.array([1., 0.]),
            bbox=np.array([40., 40., 60., 60.]),
        )])
        engine.face_future_frame = 0
        engine.face_future_shape = (100, 100)
        engine.face_future_people = [(1, 50., 50.)]
        engine.face_app = None
        engine.gallery = {"user_001": {"embedding": np.array([1., 0.])}}
        engine.face_scan_period = 60
        engine.frame_no = 1
        engine.tracks = {1: {"uid": "guest-001"}}
        engine._face_ids(Frame(), [ [SimpleNamespace(x=.5, y=.5)] ], [(1, "guest-001")])
        self.assertEqual(engine.tracks[1]["uid"], "user_001")

    def make_engine(self, exercise):
        engine = GymPoseEngine.__new__(GymPoseEngine)
        engine.exercise = exercise
        engine.counts = {}
        engine.exercise_states = {}
        return engine

    def test_squat_counts_only_full_down_up_cycle(self):
        engine = self.make_engine("Squat")
        engine._count("user_001", [], {"left_knee": 90})
        self.assertEqual(engine.counts, {})
        engine._count("user_001", [], {"left_knee": 170})
        self.assertEqual(engine.counts[("user_001", "Squat")], 1)
        engine._count("user_001", [], {"left_knee": 170})
        self.assertEqual(engine.counts[("user_001", "Squat")], 1)

    def test_bicep_curl_cycle(self):
        engine = self.make_engine("Bicep curl")
        engine._count("u", [], {"left_elbow": 35})
        engine._count("u", [], {"left_elbow": 165})
        self.assertEqual(engine.counts[("u", "Bicep curl")], 1)

    def test_pushup_cycle(self):
        engine = self.make_engine("Push-up")
        engine._count("u", [], {"left_elbow": 80})
        engine._count("u", [], {"left_elbow": 165})
        self.assertEqual(engine.counts[("u", "Push-up")], 1)

    def test_shoulder_press_cycle(self):
        engine = self.make_engine("Shoulder press")
        engine._count("u", [], {"left_shoulder": 30})
        engine._count("u", [], {"left_shoulder": 110})
        self.assertEqual(engine.counts[("u", "Shoulder press")], 1)

    def test_lunge_cycle(self):
        engine = self.make_engine("Lunge")
        engine._count("u", [], {"left_knee": 90})
        engine._count("u", [], {"left_knee": 170})
        self.assertEqual(engine.counts[("u", "Lunge")], 1)

    def test_jumping_jack_cycle(self):
        engine = self.make_engine("Jumping jack")
        points = [SimpleNamespace(x=.5, y=.5) for _ in range(33)]
        points[0] = SimpleNamespace(x=.5, y=.3)
        points[15] = SimpleNamespace(x=.4, y=.1); points[16] = SimpleNamespace(x=.6, y=.1)
        points[23] = SimpleNamespace(x=.4, y=.6); points[24] = SimpleNamespace(x=.6, y=.6)
        points[27] = SimpleNamespace(x=.2, y=.9); points[28] = SimpleNamespace(x=.8, y=.9)
        engine._count("u", points, {})  # open does not count without prior closed pose
        self.assertEqual(engine.counts, {})
        points[15] = SimpleNamespace(x=.4, y=.5); points[16] = SimpleNamespace(x=.6, y=.5)
        points[27] = SimpleNamespace(x=.45, y=.9); points[28] = SimpleNamespace(x=.55, y=.9)
        engine._count("u", points, {})
        points[15] = SimpleNamespace(x=.4, y=.1); points[16] = SimpleNamespace(x=.6, y=.1)
        points[27] = SimpleNamespace(x=.2, y=.9); points[28] = SimpleNamespace(x=.8, y=.9)
        engine._count("u", points, {})
        self.assertEqual(engine.counts[("u", "Jumping jack")], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
