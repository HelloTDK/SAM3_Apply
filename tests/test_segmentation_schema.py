import unittest

from sam3_service.schemas import SegmentationRequest


class SegmentationRequestTest(unittest.TestCase):
    def test_return_result_image_is_disabled_by_default_and_can_be_enabled(self):
        payload = {"pic_id": "test", "image_base64": "unused", "prompt": "person"}

        # 普通接口默认不落结果图，页面等调试调用可以显式开启。
        self.assertFalse(SegmentationRequest(**payload).return_result_image)
        self.assertTrue(SegmentationRequest(**payload, return_result_image=True).return_result_image)


if __name__ == "__main__":
    unittest.main()
