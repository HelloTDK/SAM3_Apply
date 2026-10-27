import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from sam3_service.native_click_trt import prepare_points, remove_small_mask_regions
from sam3_service.native_click_trt import NativeInteractiveBackend
from sam3_service.native_click_trt import timed_interaction, timed_step


class FakeArray:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class FakeDecoder:
    def __init__(self, masks, scores):
        self.outputs = {"masks": FakeArray(masks), "scores": FakeArray(scores)}
        self.calls = []

    def __call__(self, **inputs):
        self.calls.append(inputs)
        return self.outputs


class NativePromptTest(unittest.TestCase):
    def test_injected_encoder_is_shared_and_does_not_load_another_image_model(self):
        encoder = Mock()
        encoder.get_tensor_shape.return_value = (1, 3, 1008, 1008)
        encoder.return_value = {
            name: np.zeros((1, 1))
            for name in ("image_embed", "high_res_0", "high_res_1", "fpn_feat_0")
        }
        with patch("sam3_service.native_click_trt.TensorRTEngine") as engine_type:
            backend = NativeInteractiveBackend(Path("engines/native_interactive"), encoder=encoder)
        self.assertIs(backend.encoder, encoder)
        self.assertEqual(
            [call.args[0].name for call in engine_type.call_args_list],
            ["interactive-decoder-first.engine", "interactive-decoder-refine.engine"],
        )
        from PIL import Image

        features = backend.encode_image(Image.new("RGB", (40, 30)))
        self.assertEqual(set(features), {"image_embed", "high_res_0", "high_res_1"})
        self.assertTrue(all(features[name] is encoder.return_value[name] for name in features))

    def test_small_islands_are_removed_except_positive_click_regions(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[5:55, 5:55] = True
        mask[20:22, 20:22] = False  # The cleanup must preserve holes.
        mask[65:68, 5:8] = True
        mask[65:68, 65:68] = True
        mask[70:80, 10:20] = True
        cleaned = remove_small_mask_regions(mask, [(10, 10, 1), (66, 66, 1), (6, 66, 0)])
        self.assertTrue(cleaned[10, 10])
        self.assertFalse(cleaned[20, 20])
        self.assertFalse(cleaned[66, 6])
        self.assertTrue(cleaned[66, 66])
        self.assertTrue(cleaned[75, 15])
        self.assertTrue(mask[66, 6])  # Cleanup does not mutate the decoder's mask.
        empty = np.zeros((10, 10), dtype=bool)
        self.assertFalse(remove_small_mask_regions(empty, [(5, 5, 1)]).any())

    def test_clicks_scale_to_sam3_square_frame_without_changing_labels(self):
        coords, labels = prepare_points(
            [(25, 40, 1), (75, 10, 0)], image_size=(100, 80), target_size=(1008, 1008)
        )

        np.testing.assert_allclose(
            coords,
            [[[252.0, 504.0], [756.0, 126.0]]],
        )
        np.testing.assert_array_equal(labels, [[1, 0]])

    def test_second_click_refines_using_selected_previous_logits(self):
        first_masks = np.zeros((1, 3, 288, 288), dtype=np.float32)
        first_masks[0, 1, 80:200, 90:210] = 2.0
        backend = NativeInteractiveBackend.__new__(NativeInteractiveBackend)
        backend.target_size = (1008, 1008)
        backend.max_clicks = 32
        backend.first_decoder = FakeDecoder(first_masks, np.asarray([[0.1, 0.9, 0.2]]))
        backend.refine_decoder = FakeDecoder(
            np.ones((1, 1, 288, 288), dtype=np.float32), np.asarray([[0.8]])
        )
        features = {"image_embed": np.zeros((1, 256, 72, 72), dtype=np.float32)}

        mask, score, logits = backend.segment(features, (100, 100), [(50, 50, 1)])
        self.assertAlmostEqual(score, 0.9)
        self.assertTrue(mask.any())
        self.assertNotIn("mask_input", backend.first_decoder.calls[0])

        backend.segment(features, (100, 100), [(50, 50, 1), (55, 50, 0)], logits)
        np.testing.assert_array_equal(backend.refine_decoder.calls[0]["mask_input"], logits)
        np.testing.assert_array_equal(
            backend.refine_decoder.calls[0]["point_labels"], [[1, 0]]
        )


class TimingTest(unittest.TestCase):
    def test_failed_step_still_logs_elapsed_time_and_preserves_error(self):
        output = io.StringIO()
        with redirect_stdout(output), patch(
            "sam3_service.native_click_trt.perf_counter", side_effect=[1.0, 1.125]
        ):
            with self.assertRaisesRegex(ValueError, "expected failure"):
                with timed_step("测试失败步骤"):
                    raise ValueError("expected failure")
        self.assertIn("测试失败步骤耗时：125.00 ms", output.getvalue())

    def test_callback_keeps_name_outputs_and_adds_total_to_status(self):
        @timed_interaction("测试交互")
        def callback(value):
            return value, None, "原始状态"

        output = io.StringIO()
        with redirect_stdout(output), patch(
            "sam3_service.native_click_trt.perf_counter", side_effect=[1.0, 1.25]
        ):
            result = callback(7)
        self.assertEqual(callback.__name__, "callback")
        self.assertEqual(result, (7, None, "原始状态，耗时：250.00 ms"))
        self.assertIn("测试交互总计耗时：250.00 ms", output.getvalue())


if __name__ == "__main__":
    unittest.main()
