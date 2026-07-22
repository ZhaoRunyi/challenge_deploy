from __future__ import annotations

import base64
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading
import unittest

import numpy as np

from clients.fastwam import (
    FASTWAM_DEFAULT_TRAIN_CONFIG,
    FASTWAM_RAW_PROPRIO_DIM,
    FASTWAM_TASK_PROMPTS,
    FastWAMHTTPPolicyClient,
    FastWAMPiperClient,
    build_fastwam_proprio,
    build_policy_payload,
    load_fastwam_policy_spec,
)
from hardware.schemas import DualPiperState, PiperArmState, RobotSnapshot
from rollout.assets import prepare_client_assets
from run_fastwam_client import build_parser


class FastWAMClientTest(unittest.TestCase):
    def make_arm_state(self, name: str, offset: float, gripper: float) -> PiperArmState:
        qpos = np.array([offset + 0.1 * index for index in range(6)] + [gripper], dtype=np.float64)
        end_pose = np.array([offset + 0.01, offset + 0.02, offset + 0.03, 0.0, 0.0, 0.0, gripper], dtype=np.float64)
        return PiperArmState(
            name=name,
            can_name=f"can_{name}",
            qpos=qpos,
            qpos_feedback=qpos.copy(),
            qpos_command=qpos.copy(),
            qvel=np.zeros(7, dtype=np.float64),
            effort=np.zeros(7, dtype=np.float64),
            end_pose=end_pose,
            enabled=True,
            status={},
            feedback_hz=100.0,
            status_hz=100.0,
            command_hz=100.0,
        )

    def make_snapshot(self) -> RobotSnapshot:
        images = {
            "cam_high": np.zeros((8, 10, 3), dtype=np.uint8),
            "cam_left_wrist": np.full((8, 10, 3), 32, dtype=np.uint8),
            "cam_right_wrist": np.full((8, 10, 3), 64, dtype=np.uint8),
        }
        return RobotSnapshot(
            timestamp_s=1.0,
            state=DualPiperState(
                left=self.make_arm_state("left", 0.0, 0.02),
                right=self.make_arm_state("right", 1.0, 0.03),
            ),
            images=images,
        )

    def test_fastwam_spec_and_proprio_layout(self) -> None:
        spec = load_fastwam_policy_spec("fastwam_test", action_horizon=16)
        self.assertEqual(spec.state_dim, 32)
        self.assertEqual(spec.action_dim, 14)
        self.assertEqual(spec.action_horizon, 16)
        self.assertEqual(spec.image_ids, ("cam_high", "cam_left_wrist", "cam_right_wrist"))

        proprio = build_fastwam_proprio(self.make_snapshot(), spec, state_gripper_encoding="meters")
        self.assertEqual(proprio.shape, (FASTWAM_RAW_PROPRIO_DIM,))
        self.assertAlmostEqual(float(proprio[6]), 0.02)
        self.assertAlmostEqual(float(proprio[22]), 0.03)

    def test_fastwam_payload_matches_server_contract(self) -> None:
        spec = load_fastwam_policy_spec(action_horizon=12)
        payload = build_policy_payload(
            self.make_snapshot(),
            prompt="pick up the cup",
            spec=spec,
            action_horizon=12,
            num_inference_steps=7,
            seed=3,
            state_gripper_encoding="meters",
            session_id="fastwam_test_session",
        )
        self.assertEqual(payload["instruction"], "pick up the cup")
        self.assertEqual(payload["action_horizon"], 12)
        self.assertEqual(payload["num_inference_steps"], 7)
        self.assertEqual(payload["seed"], 3)
        self.assertEqual(payload["session_id"], "fastwam_test_session")
        self.assertNotIn("return_predicted_video", payload)
        self.assertEqual(len(payload["proprio"]), 32)
        self.assertEqual(set(payload["images"]), set(spec.image_ids))
        for encoded in payload["images"].values():
            self.assertIsInstance(encoded, str)
            self.assertGreater(len(base64.b64decode(encoded)), 0)


    def test_fastwam_prompt_assets(self) -> None:
        prompt = "Use the left arm to click the bell on the left and use the right arm to click the bell on the right."
        spec = load_fastwam_policy_spec("piper_realworld_unseen_adapter", prompt=prompt)
        self.assertEqual(spec.train_config_name, "piper_realworld_unseen_adapter")
        self.assertEqual(spec.distribution_name, "click_two_bells")
        self.assertEqual(spec.prompt, prompt)

        assets = prepare_client_assets(
            client_kind="fastwam",
            train_config_name="piper_realworld_unseen_adapter",
            cli_prompt=prompt,
            need_distribution=True,
            spec=spec,
        )
        self.assertEqual(assets.prompt, prompt)
        self.assertEqual(assets.prompt_source, "cli")
        self.assertIsNotNone(assets.distribution_image_path)
        self.assertEqual(assets.distribution_image_path.name, "click_two_bells_cam_high_first_frame_overlay.png")

        legacy_spec = load_fastwam_policy_spec("clean_plate")
        self.assertIsNone(legacy_spec.prompt)
        self.assertIsNone(legacy_spec.distribution_name)

    def test_fastwam_seen_prompt_assets_prefer_real_distribution(self) -> None:
        seen_tasks = (
            "beaker_mixer",
            "carry_basket",
            "click_bell",
            "depress_pipette",
            "dock_tubes",
            "insert_test_tube",
            "items_handover_place",
            "open_drawer",
            "open_pan",
            "pour_dual",
            "rearr",
        )
        for task_name in seen_tasks:
            with self.subTest(task_name=task_name):
                prompt = FASTWAM_TASK_PROMPTS[task_name]
                spec = load_fastwam_policy_spec("piper_realworld_unseen_adapter", prompt=prompt)
                assets = prepare_client_assets(
                    client_kind="fastwam",
                    train_config_name="piper_realworld_unseen_adapter",
                    cli_prompt=prompt,
                    need_distribution=True,
                    spec=spec,
                )
                self.assertIsNotNone(assets.distribution_image_path)
                self.assertFalse(assets.distribution_image_path.name.startswith("embodichain_sim_data"))


    def test_fastwam_http_preflight_accepts_expected_400(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.dumps({"error": "`images` (object with cam_high/cam_left_wrist/cam_right_wrist) is required."}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = FastWAMHTTPPolicyClient("127.0.0.1", server.server_port, timeout_s=2.0)
            result = client.probe(timeout_s=2.0)
            self.assertTrue(result["ok"])
            self.assertEqual(result["status"], 400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    def test_fastwam_http_decodes_predicted_video(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                _ = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                body = json.dumps({"predicted_video_b64": base64.b64encode(b"video-bytes").decode("ascii")}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = FastWAMHTTPPolicyClient("127.0.0.1", server.server_port, timeout_s=2.0)
            result = client.infer({"_request": "get_predicted_video", "session_id": "s"})
            self.assertEqual(result["predicted_video_bytes"], b"video-bytes")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    def test_fastwam_client_and_parser_defaults(self) -> None:
        client = FastWAMPiperClient(action_horizon=8)
        self.assertEqual(client.state_gripper_encoding, "policy")
        self.assertEqual(client.action_gripper_encoding, "policy")
        self.assertEqual(client.spec.action_horizon, 8)
        self.assertEqual(client.spec.train_config_name, FASTWAM_DEFAULT_TRAIN_CONFIG)

        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.state_gripper, "policy")
        self.assertEqual(args.action_gripper, "policy")
        self.assertEqual(args.port, 8765)
        self.assertFalse(hasattr(args, "task"))
        self.assertFalse(args.skip_server_preflight)
        self.assertEqual(args.server_preflight_timeout, 8.0)
        self.assertEqual(args.action_horizon, 32)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--task", "clean_plate"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--old_gripper"])


if __name__ == "__main__":
    unittest.main()
